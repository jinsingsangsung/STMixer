import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List
from torch import nn, Tensor
from .util.box_ops import bbox_overlaps, box_cxcywh_to_xyxy, clip_boxes_tensor
from .util.msaq import SAMPLE4D
from .util.adaptive_mixing_operator import AdaptiveMixing
from .util.head_utils import _get_activation_layer, bias_init_with_prob, decode_box, position_embedding, make_interpolated_features, _get_clones
from .util.misc import inverse_sigmoid
from .util.head_utils import FFN, MLP, MultiheadAttention
from .util.attention import MultiheadAttention_
from .util.loss import SetCriterion, HungarianMatcher
# from .transformer.dab_transformer import build_transformer
from .util.misc import (NestedTensor, nested_tensor_from_tensor_list, inverse_sigmoid)
from einops import rearrange, repeat
import math

class AdaptiveSTSamplingMixing(nn.Module):

    def __init__(self, spatial_points=32,
                 temporal_points=4,
                 out_multiplier=4,
                 n_groups=4,
                 query_dim=256,
                 feat_channels=None):
        super(AdaptiveSTSamplingMixing, self).__init__()
        self.spatial_points = spatial_points
        self.temporal_points = temporal_points
        self.out_multiplier = out_multiplier
        self.n_groups = n_groups
        self.query_dim = query_dim
        self.feat_channels = feat_channels if feat_channels is not None else self.query_dim
        self.offset_generator = nn.Sequential(nn.Linear(query_dim, spatial_points * n_groups * 3))

        self.norm_s = nn.LayerNorm(query_dim)
        self.norm_t = nn.LayerNorm(query_dim)

        self.adaptive_mixing_s = AdaptiveMixing(
            self.feat_channels,
            query_dim=self.query_dim,
            in_points=self.spatial_points,
            out_points=self.spatial_points*self.out_multiplier,
            n_groups=self.n_groups,
        )

        self.adaptive_mixing_t = AdaptiveMixing(
            self.feat_channels,
            query_dim=self.query_dim,
            in_points=self.temporal_points,
            out_points=self.temporal_points*self.out_multiplier,
            n_groups=self.n_groups,
        )

        self.init_weights()

    def init_weights(self):
        nn.init.zeros_(self.offset_generator[-1].weight.data)
        nn.init.zeros_(self.offset_generator[-1].bias.data)

        bias = self.offset_generator[-1].bias.data.view(
            self.n_groups, self.spatial_points, 3)

        if int(self.spatial_points ** 0.5) ** 2 == self.spatial_points:
            h = int(self.in_points ** 0.5)
            y = torch.linspace(-0.5, 0.5, h + 1) + 0.5 / h
            yp = y[:-1]
            y = yp.view(-1, 1).repeat(1, h)
            x = yp.view(1, -1).repeat(h, 1)
            y = y.flatten(0, 1)[None, :, None]
            x = x.flatten(0, 1)[None, :, None]
            bias[:, :, 0:2] = torch.cat([y, x], dim=-1)
            # 格子采样
        else:
            bandwidth = 0.5 * 1.0
            nn.init.uniform_(bias, -bandwidth, bandwidth)
        bias[:, :, 2:3].mul_(0.0)

        self.adaptive_mixing_s._init_weights()
        self.adaptive_mixing_t._init_weights()

    def forward(self, features, proposal_boxes, spatial_queries, temporal_queries, featmap_strides):

        interpolated_features = make_interpolated_features(features)
        # B, C, n_groups, temporal_points, spatial_points, n_query, _ = sampled_feature.size()
        sampled_feature = sampled_feature.flatten(5, 6)                   # B, n_channels, n_groups, temporal_points, spatial_points, n_query
        sampled_feature = sampled_feature.permute(0, 5, 2, 3, 4, 1)       # B, n_query, n_groups, temporal_points, spatial_points, n_channels

        spatial_feats = torch.mean(sampled_feature, dim=3)                            # out_s has shape [B, n_query, n_groups, spatial_points, n_channels]
        spatial_queries = self.adaptive_mixing_s(spatial_feats, spatial_queries)
        spatial_queries = self.norm_s(spatial_queries)

        temporal_feats = torch.mean(sampled_feature, dim=4)                        # out_t has shape [B, n_query, n_groups, temporal_points, n_channels]
        temporal_queries = self.adaptive_mixing_t(temporal_feats, temporal_queries)
        temporal_queries = self.norm_t(temporal_queries)

        return spatial_queries, temporal_queries


class TransformerDecoder(nn.Module):

    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False, 
                    d_model=256, query_dim=2, keep_query_pos=False, query_scale_type='cond_elewise',
                    modulate_hw_attn=False,
                    bbox_embed_diff_each_layer=False,
                    ):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate
        assert return_intermediate
        self.query_dim = query_dim
        assert query_scale_type in ['cond_elewise', 'cond_scalar', 'fix_elewise']
        self.query_scale_type = query_scale_type
        if query_scale_type == 'cond_elewise':
            self.query_scale = MLP(d_model, d_model, d_model, 2)
        elif query_scale_type == 'cond_scalar':
            self.query_scale = MLP(d_model, d_model, 1, 2)
        elif query_scale_type == 'fix_elewise':
            self.query_scale = nn.Embedding(num_layers, d_model)
        else:
            raise NotImplementedError("Unknown query_scale_type: {}".format(query_scale_type))
        
        self.ref_point_head = MLP(query_dim // 2 * d_model, d_model, d_model, 2)
        
        self.bbox_embed = None
        self.d_model = d_model
        self.modulate_hw_attn = modulate_hw_attn
        self.bbox_embed_diff_each_layer = bbox_embed_diff_each_layer

        if modulate_hw_attn:
            self.ref_anchor_head = MLP(d_model, d_model, 2, 2)

        if not keep_query_pos:
            for layer_id in range(num_layers - 1):
                self.layers[layer_id + 1].ca_qpos_proj = None

        dim_feedforward = decoder_layer.dim_feedforward
        dropout = decoder_layer.dropout_rate
        self.activation = decoder_layer.activation
        self.cls_norm = nn.LayerNorm(d_model)
        self.cls_linear1 = nn.Linear(d_model, dim_feedforward)
        self.cls_linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout = nn.Dropout(dropout)
        self.conv_activation = _get_activation_layer(dict(type='RelU', inplace=True))

        self.conv1 = nn.Conv2d(2*d_model, d_model, kernel_size=1)
        self.conv2 = nn.Conv2d(d_model, d_model, kernel_size=1)
        self.q_proj = nn.Conv2d(d_model, d_model, kernel_size=1)
        self.k_proj = nn.Conv2d(d_model, d_model, kernel_size=1)
        self.v_proj = nn.Conv2d(d_model, d_model, kernel_size=1)
        
        # self.cls_params = nn.Linear(d_model, 80).weight
        # self.class_queries = nn.Embedding(80, 256).weight
        # self.conv2 = nn.Conv2d(d_model, 2*d_model, kernel_size=3, stride=2)
        # self.conv3 = nn.Conv2d(2*d_model, 2*d_model, kernel_size=3, stride=2)
        self.linear = nn.Linear(d_model, d_model)
        self.cls_norm_ = nn.LayerNorm(d_model)
        self.cls_norm__ = nn.LayerNorm(d_model)
        self.cls_linear1_ = nn.Linear(d_model, dim_feedforward)
        self.cls_linear2_ = nn.Linear(dim_feedforward, d_model)
        self.dropout_ = nn.Dropout(dropout)
        
        self.cls_norm2 = nn.LayerNorm(d_model)

    def gen_sineembed_for_position(self, pos_tensor):
            # n_query, bs, _ = pos_tensor.size()
            # sineembed_tensor = torch.zeros(n_query, bs, 256)
            scale = 2 * math.pi
            dim_t = torch.arange(128, dtype=torch.float32, device=pos_tensor.device)
            dim_t = 10000 ** (2 * (dim_t // 2) / 128)
            x_embed = pos_tensor[:, :, 0] * scale
            y_embed = pos_tensor[:, :, 1] * scale
            pos_x = x_embed[:, :, None] / dim_t
            pos_y = y_embed[:, :, None] / dim_t
            pos_x = torch.stack((pos_x[:, :, 0::2].sin(), pos_x[:, :, 1::2].cos()), dim=3).flatten(2)
            pos_y = torch.stack((pos_y[:, :, 0::2].sin(), pos_y[:, :, 1::2].cos()), dim=3).flatten(2)
            if pos_tensor.size(-1) == 2:
                pos = torch.cat((pos_y, pos_x), dim=2)
            elif pos_tensor.size(-1) == 4:
                w_embed = pos_tensor[:, :, 2] * scale
                pos_w = w_embed[:, :, None] / dim_t
                pos_w = torch.stack((pos_w[:, :, 0::2].sin(), pos_w[:, :, 1::2].cos()), dim=3).flatten(2)

                h_embed = pos_tensor[:, :, 3] * scale
                pos_h = h_embed[:, :, None] / dim_t
                pos_h = torch.stack((pos_h[:, :, 0::2].sin(), pos_h[:, :, 1::2].cos()), dim=3).flatten(2)

                pos = torch.cat((pos_y, pos_x, pos_w, pos_h), dim=2)
            else:
                raise ValueError("Unknown pos_tensor shape(-1):{}".format(pos_tensor.size(-1)))
            return pos

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                refpoints_unsigmoid: Optional[Tensor] = None, # num_queries, bs, 2
                class_queries: Optional[Tensor] = None,
                orig_res = None,
                ):
        output = tgt

        intermediate = []
        cls_intermediate = []
        reference_points = refpoints_unsigmoid.sigmoid()
        ref_points = [reference_points]

        # import ipdb; ipdb.set_trace()        

        for layer_id, layer in enumerate(self.layers):
            obj_center = reference_points[..., :self.query_dim]     # [num_queries, batch_size, 2]
            # get sine embedding for the query vector
            query_sine_embed = self.gen_sineembed_for_position(obj_center)  
            query_pos = self.ref_point_head(query_sine_embed) 

            # For the first decoder layer, we do not apply transformation over p_s
            if self.query_scale_type != 'fix_elewise':
                if layer_id == 0:
                    pos_transformation = 1
                else:
                    pos_transformation = self.query_scale(output)
            else:
                pos_transformation = self.query_scale.weight[layer_id]

            # apply transformation
            query_sine_embed = query_sine_embed[...,:self.d_model] * pos_transformation

            # modulated HW attentions
            if self.modulate_hw_attn:
                refHW_cond = self.ref_anchor_head(output).sigmoid() # nq, bs*t, 2
                query_sine_embed[..., self.d_model // 2:] *= (refHW_cond[..., 0] / obj_center[..., 2]).unsqueeze(-1)
                query_sine_embed[..., :self.d_model // 2] *= (refHW_cond[..., 1] / obj_center[..., 3]).unsqueeze(-1)


            output, actor_feature = layer(output, memory, tgt_mask=tgt_mask,
                           memory_mask=memory_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_mask=memory_key_padding_mask,
                           pos=pos, query_pos=query_pos, query_sine_embed=query_sine_embed,
                           is_first=(layer_id == 0))

            # separate classification branch from localization
            actor_feature = actor_feature.clone().detach()
            actor_feature2 = self.cls_linear2(self.dropout(self.activation(self.cls_linear1(actor_feature))))
            actor_feature = actor_feature + self.dropout(actor_feature2)
            actor_feature = self.cls_norm(actor_feature)

            # apply convolution
            h, w = orig_res
            actor_feature_expanded = actor_feature.flatten(0,1)[..., None, None].expand(-1, -1, h, w) # N_q*B, D, H, W
            encoded_feature_expanded = memory[:, None].expand(-1, len(tgt), -1, -1).flatten(1,2).view(h,w,-1,actor_feature.shape[-1]).permute(2,3,0,1) # N_q*B, D, H, W

            cls_feature = self.conv_activation(self.conv1(torch.cat([actor_feature_expanded, encoded_feature_expanded], dim=1)))
            cls_feature = self.conv_activation(self.conv2(cls_feature))
            # cls_feature = self.bn1(cls_feature)
            query = self.q_proj(cls_feature)
            query = query[:, None].expand(-1, 80, -1, -1, -1)
            key = class_queries[None, :, :, None, None].expand(actor_feature_expanded.shape[0], -1, -1, h, w)
            # key = self.cls_params[None, :, :, None, None].expand(actor_feature_expanded.shape[0], -1, -1, h, w)
            attn = (query*key).sum(dim=2).flatten(2).softmax(dim=2).reshape(actor_feature_expanded.shape[0], -1, h, w)[:, :, None]
            value = self.v_proj(encoded_feature_expanded)[:, None]
            cls_output = (attn * value).sum(dim=-1).sum(dim=-1).view(len(tgt), -1, 80, cls_feature.shape[1]) #N_q, B, N_c, D
            cls_output = self.linear(cls_output)
            cls_output2 = self.cls_linear2_(self.dropout_(self.activation(self.cls_linear1_(cls_output))))
            cls_output = cls_output + self.dropout_(cls_output2)
            cls_output = self.cls_norm_(cls_output)
            if layer_id != 0:
                cls_output = self.cls_norm__(cls_output + prev_output)
            prev_output = cls_output
            
            # iter update
            if self.bbox_embed is not None:
                if self.bbox_embed_diff_each_layer:
                    tmp = self.bbox_embed[layer_id](output)
                else:
                    tmp = self.bbox_embed(output)
                # import ipdb; ipdb.set_trace()
                tmp[..., :self.query_dim] += inverse_sigmoid(reference_points)
                new_reference_points = tmp[..., :self.query_dim].sigmoid()
                if layer_id != self.num_layers - 1:
                    ref_points.append(new_reference_points)
                reference_points = new_reference_points.detach()

            if self.return_intermediate:
                intermediate.append(self.norm(output))
                cls_intermediate.append(self.cls_norm2(cls_output))

        if self.norm is not None:
            output = self.norm(output)
            cls_output = self.cls_norm2(cls_output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)
                cls_intermediate.pop()
                cls_intermediate.append(cls_output)

        if self.return_intermediate:
            if self.bbox_embed is not None:
                return [
                    torch.stack(intermediate).transpose(1, 2),
                    torch.stack(cls_intermediate).transpose(1, 2),
                    torch.stack(ref_points).transpose(1, 2),
                ]
            else:
                return [
                    torch.stack(intermediate).transpose(1, 2), 
                    torch.stack(cls_intermediate).transpose(1, 2),
                    reference_points.unsqueeze(0).transpose(1, 2)
                ]

        return output.unsqueeze(0)


class DecoderStage(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False, keep_query_pos=False,
                 rm_self_attn_decoder=False):
        super().__init__()
        # Decoder Self-Attention
        if not rm_self_attn_decoder:
            self.sa_qcontent_proj = nn.Linear(d_model, d_model)
            self.sa_qpos_proj = nn.Linear(d_model, d_model)
            self.sa_kcontent_proj = nn.Linear(d_model, d_model)
            self.sa_kpos_proj = nn.Linear(d_model, d_model)
            self.sa_v_proj = nn.Linear(d_model, d_model)
            self.self_attn = MultiheadAttention_(d_model, nhead, dropout=dropout, vdim=d_model)

            self.norm1 = nn.LayerNorm(d_model)
            self.dropout1 = nn.Dropout(dropout)

        # Decoder Cross-Attention
        self.ca_qcontent_proj = nn.Linear(d_model, d_model)
        self.ca_qpos_proj = nn.Linear(d_model, d_model)
        self.ca_kcontent_proj = nn.Linear(d_model, d_model)
        self.ca_kpos_proj = nn.Linear(d_model, d_model)
        self.ca_v_proj = nn.Linear(d_model, d_model)
        self.ca_qpos_sine_proj = nn.Linear(d_model, d_model)
        self.cross_attn = MultiheadAttention_(d_model*2, nhead, dropout=dropout, vdim=d_model)

        self.nhead = nhead
        self.rm_self_attn_decoder = rm_self_attn_decoder

        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_layer(dict(type='RelU', inplace=True))
        self.normalize_before = normalize_before
        self.keep_query_pos = keep_query_pos

        self.dim_feedforward = dim_feedforward
        self.dropout_rate = dropout

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward(self, tgt, memory,
                     tgt_mask: Optional[Tensor] = None,
                     memory_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None,
                     query_sine_embed = None,
                     is_first = False):
                     
        # ========== Begin of Self-Attention =============
        if not self.rm_self_attn_decoder:
            # Apply projections here
            # shape: num_queries x batch_size x 256
            q_content = self.sa_qcontent_proj(tgt)      # target is the input of the first decoder layer. zero by default.
            q_pos = self.sa_qpos_proj(query_pos)
            k_content = self.sa_kcontent_proj(tgt)
            k_pos = self.sa_kpos_proj(query_pos)
            v = self.sa_v_proj(tgt)

            num_queries, bs, n_model = q_content.shape
            hw, _, _ = k_content.shape

            q = q_content + q_pos
            k = k_content + k_pos

            tgt2 = self.self_attn(q, k, value=v, attn_mask=tgt_mask,
                                key_padding_mask=tgt_key_padding_mask)[0]
            # ========== End of Self-Attention =============

            tgt = tgt + self.dropout1(tgt2)
            tgt = self.norm1(tgt)

        # ========== Begin of Cross-Attention =============
        # Apply projections here
        # shape: num_queries x batch_size x 256
        q_content = self.ca_qcontent_proj(tgt)
        k_content = self.ca_kcontent_proj(memory)
        v = self.ca_v_proj(memory)

        num_queries, bs, n_model = q_content.shape
        hw, _, _ = k_content.shape

        k_pos = self.ca_kpos_proj(pos)

        # For the first decoder layer, we concatenate the positional embedding predicted from 
        # the object query (the positional embedding) into the original query (key) in DETR.
        if is_first or self.keep_query_pos:
            q_pos = self.ca_qpos_proj(query_pos)
            q = q_content + q_pos
            k = k_content + k_pos
        else:
            q = q_content
            k = k_content

        q = q.view(num_queries, bs, self.nhead, n_model//self.nhead)
        query_sine_embed = self.ca_qpos_sine_proj(query_sine_embed)
        query_sine_embed = query_sine_embed.view(num_queries, bs, self.nhead, n_model//self.nhead)
        q = torch.cat([q, query_sine_embed], dim=3).view(num_queries, bs, n_model * 2)
        k = k.view(hw, bs, self.nhead, n_model//self.nhead)
        k_pos = k_pos.view(hw, bs, self.nhead, n_model//self.nhead)
        k = torch.cat([k, k_pos], dim=3).view(hw, bs, n_model * 2)

        tgt2 = self.cross_attn(query=q,
                                   key=k,
                                   value=v, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        # tgt2 = self.cross_attn(query=q,key=k,value=v, attn_mask=memory_mask,key_padding_mask=memory_key_padding_mask)[0]
        # ========== End of Cross-Attention =============
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt_temp = tgt
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt, tgt_temp


class AMStage(nn.Module):

    def __init__(self, query_dim=256,
                 feat_channels=256,
                 num_heads=8,
                 feedforward_channels=2048,
                 dropout=0.0,
                 num_ffn_fcs=2,
                 ffn_act='RelU',
                 spatial_points=32,
                 temporal_points=4,
                 out_multiplier=4,
                 n_groups=4,
                 num_cls_fcs=1,
                 num_reg_fcs=1,
                 num_action_fcs=1,
                 num_classes_object=1,
                 num_classes_action=80,):


        super(AMStage, self).__init__()

        # MHSA-S
        ffn_act_cfg = dict(type=ffn_act, inplace=True)
        self.attention_s = MultiheadAttention(query_dim, num_heads, dropout)
        self.attention_norm_s = nn.LayerNorm(query_dim, eps=1e-5)
        self.ffn_s = FFN(query_dim, feedforward_channels, num_ffn_fcs, act_cfg=ffn_act_cfg, dropout=dropout)
        self.ffn_norm_s = nn.LayerNorm(query_dim, eps=1e-5)
        self.iof_tau = nn.Parameter(torch.ones(self.attention_s.num_heads,))

        # MHSA-T
        self.attention_t = MultiheadAttention(query_dim, num_heads, dropout)
        self.attention_norm_t = nn.LayerNorm(query_dim, eps=1e-5)
        self.ffn_t = FFN(query_dim, feedforward_channels, num_ffn_fcs, act_cfg=ffn_act_cfg, dropout=dropout)
        self.ffn_norm_t = nn.LayerNorm(query_dim, eps=1e-5)

        self.samplingmixing = AdaptiveSTSamplingMixing(
            spatial_points=spatial_points,
            temporal_points=temporal_points,
            out_multiplier=out_multiplier,
            n_groups=n_groups,
            query_dim=query_dim,
            feat_channels=feat_channels
        )

        cls_feature_dim = query_dim
        reg_feature_dim = query_dim
        action_feat_dim = query_dim * 2

        # human classifier
        self.human_cls_fcs = nn.ModuleList()
        for _ in range(num_cls_fcs):
            self.human_cls_fcs.append(
                nn.Linear(cls_feature_dim, cls_feature_dim, bias=True))
            self.human_cls_fcs.append(
                nn.LayerNorm(cls_feature_dim, eps=1e-5))
            self.human_cls_fcs.append(
                _get_activation_layer(ffn_act_cfg))
        self.human_fc_cls = nn.Linear(cls_feature_dim, num_classes_object + 1)

        # human bbox regressor
        self.reg_fcs = nn.ModuleList()
        for _ in range(num_reg_fcs):
            self.reg_fcs.append(
                nn.Linear(reg_feature_dim, reg_feature_dim, bias=True))
            self.reg_fcs.append(
                nn.LayerNorm(reg_feature_dim, eps=1e-5))
            self.reg_fcs.append(
                _get_activation_layer(ffn_act_cfg))
        self.fc_reg = nn.Linear(reg_feature_dim, 4)

        # action classifier
        if dropout > 0:
            self.dropout = nn.Dropout(dropout)
        self.action_cls_fcs = nn.ModuleList()
        for _ in range(num_action_fcs):
            self.action_cls_fcs.append(
                nn.Linear(action_feat_dim, action_feat_dim, bias=True))
            self.action_cls_fcs.append(
                nn.LayerNorm(action_feat_dim, eps=1e-5))
            self.action_cls_fcs.append(
                _get_activation_layer(ffn_act_cfg))
        self.fc_action = nn.Linear(action_feat_dim, num_classes_action)

        self.init_weights()

    @torch.no_grad()
    def init_weights(self):
        for n, p in self.named_parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
            else:
                pass
        bias_init = bias_init_with_prob(0.01)
        nn.init.constant_(self.fc_action.bias, bias_init)
        nn.init.zeros_(self.fc_reg.weight)
        nn.init.zeros_(self.fc_reg.bias)
        nn.init.uniform_(self.iof_tau, 0.0, 4.0)
        self.samplingmixing.init_weights()

    def refine_xyzr(self, xyzr, xyzr_delta, return_bbox=True):
        z = xyzr[..., 2:3]
        new_xy = xyzr[..., 0:2] + xyzr_delta[..., 0:2] * (2 ** z)
        new_zr = xyzr[..., 2:4] + xyzr_delta[..., 2:4]
        xyzr = torch.cat([new_xy, new_zr], dim=-1)
        if return_bbox:
            return xyzr, decode_box(xyzr)
        else:
            return xyzr

    def forward(self, features, proposal_boxes, spatial_queries, temporal_queries, featmap_strides=[4, 8, 16, 32]):

        N, n_query = spatial_queries.shape[:2]

        with torch.no_grad():
            rois = decode_box(proposal_boxes)
            roi_box_batched = rois.view(N, n_query, 4)
            iof = bbox_overlaps(roi_box_batched, roi_box_batched, mode='iof')[:, None, :, :]
            iof = (iof + 1e-7).log()
            pe = position_embedding(proposal_boxes, spatial_queries.size(-1) // 4)

        # IoF
        attn_bias = (iof * self.iof_tau.view(1, -1, 1, 1)).flatten(0, 1)             # N*num_heads, n_query, n_query
        pe = pe.permute(1, 0, 2)                                                     # n_query, N, content_dim

        # sinusoidal positional embedding
        spatial_queries = spatial_queries.permute(1, 0, 2)  # n_query, N, content_dim
        spatial_queries_attn = spatial_queries + pe
        spatial_queries = self.attention_s(spatial_queries_attn, attn_mask=attn_bias,)
        spatial_queries = self.attention_norm_s(spatial_queries)
        spatial_queries = spatial_queries.permute(1, 0, 2)
        # N, n_query, content_dim

        temporal_queries = temporal_queries.permute(1, 0, 2)
        temporal_queries_attn = temporal_queries + pe
        temporal_queries = self.attention_t(temporal_queries_attn, attn_mask=attn_bias,)
        temporal_queries = self.attention_norm_t(temporal_queries)
        temporal_queries = temporal_queries.permute(1, 0, 2)
        # N, n_query, content_dim

        spatial_queries, temporal_queries = \
            self.samplingmixing(features, proposal_boxes, spatial_queries, temporal_queries, featmap_strides)

        spatial_queries = self.ffn_norm_s(self.ffn_s(spatial_queries))
        temporal_queries = self.ffn_norm_t(self.ffn_t(temporal_queries))

        ################################### heads ###################################
        # objectness head
        cls_feat = spatial_queries
        for cls_layer in self.human_cls_fcs:
            cls_feat = cls_layer(cls_feat)
        cls_score = self.human_fc_cls(cls_feat).view(N, n_query, -1)

        # regression head
        reg_feat = spatial_queries
        for reg_layer in self.reg_fcs:
            reg_feat = reg_layer(reg_feat)
        xyzr_delta = self.fc_reg(reg_feat).view(N, n_query, -1)

        # action head
        action_feat = torch.cat([spatial_queries, temporal_queries], dim=-1)
        for act_layer in self.action_cls_fcs:
            action_feat = act_layer(action_feat)
        action_score = self.fc_action(action_feat).view(N, n_query, -1)

        return cls_score, action_score, xyzr_delta, \
               spatial_queries.view(N, n_query, -1), temporal_queries.view(N, n_query, -1)


class DETR(nn.Module):
    """ This is the DETR module that performs object detection """
    def __init__(self, transformer, num_classes, num_queries, num_frames,
                 hidden_dim, temporal_length, aux_loss=False, generate_lfb=False, two_stage=False, random_refpoints_xy=False, query_dim=4,
                 backbone_name='CSN-152', ds_rate=1, last_stride=True, dataset_mode='ava', bbox_embed_diff_each_layer=True, training=True, iter_update=True,
                 gpu_world_rank=0, log_path=None, efficient=True):
        """ Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            num_classes: number of object classes
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         DETR can detect in a single image. For COCO, we recommend 100 queries.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
            random_refpoints_xy: random init the x,y of anchor boxes and freeze them. (It sometimes helps to improve the performance)
        """            
        super(DETR, self).__init__()
        self.temporal_length = temporal_length
        self.num_queries = num_queries
        self.num_frames = num_frames
        self.transformer = transformer
        self.dataset_mode = dataset_mode
        self.num_classes = num_classes
        self.bbox_embed_diff_each_layer = bbox_embed_diff_each_layer
        
        self.query_dim = query_dim
        assert query_dim in [2, 4]
        self.efficient = efficient
        if not efficient:
            self.refpoint_embed = nn.Embedding(num_queries*temporal_length, 4)
        else:
            assert dataset_mode == "ava", "efficient mode is only for AVA"
            self.refpoint_embed = nn.Embedding(num_queries, 4)
        self.transformer.eff = efficient
        self.random_refpoints_xy = random_refpoints_xy
        if random_refpoints_xy:
            # import ipdb; ipdb.set_trace()
            self.refpoint_embed.weight.data[:, :2].uniform_(0,1)
            self.refpoint_embed.weight.data[:, :2] = inverse_sigmoid(self.refpoint_embed.weight.data[:, :2])
            self.refpoint_embed.weight.data[:, :2].requires_grad = False          

        if "SWIN" in backbone_name:
            if gpu_world_rank == 0: print_log(log_path, "using swin")
            self.input_proj = nn.Conv3d(1024, hidden_dim, kernel_size=1)
            self.class_proj = nn.Conv3d(1024, hidden_dim, kernel_size=1)
        elif "SlowFast" in backbone_name:
            self.input_proj = nn.Conv3d(backbone.num_channels, hidden_dim, kernel_size=1)
            self.class_proj = nn.Conv3d(2048 + 512, hidden_dim, kernel_size=1)
        else:
            self.input_proj = nn.Conv3d(backbone.num_channels, hidden_dim, kernel_size=1)
            self.class_proj = nn.Conv3d(backbone.num_channels, hidden_dim, kernel_size=1)
        nn.init.xavier_uniform_(self.input_proj.weight, gain=1)
        nn.init.constant_(self.input_proj.bias, 0)    
        # self.class_proj = nn.Conv3d(backbone.num_channels[-1], hidden_dim, kernel_size=(4,1,1))

        # encoder_layer = TransformerEncoderLayer(hidden_dim, 8, 2048, 0.1, "relu", normalize_before=False)
        # self.encoder = TransformerEncoder(encoder_layer=encoder_layer, num_layers=2)
        # decoder_layer = TransformerDecoderLayer(hidden_dim, 8, 2048, 0.1, "relu", normalize_before=False)        
        # decoder_norm = nn.LayerNorm(hidden_dim)
        # self.decoder = TransformerDecoder(decoder_layer=decoder_layer, num_layers=3, norm=decoder_norm, return_intermediate=True, query_dim=4, modulate_hw_attn=True, bbox_embed_diff_each_layer=True)
        # self.num_patterns = 3
        # self.num_pattern_message:%3CTQB5fQ7CQ_2uZmev7pIQjA@geopod-ismtpd-5%3Elevel = 4
        # self.patterns = nn.Embedding(self.num_patterns*self.num_pattern_level, hidden_dim)
        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        if self.dataset_mode == 'ava':
            # self.class_embed = nn.Linear(hidden_dim, num_classes)
            self.class_embed = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1)
            )
            self.class_embed_b = nn.Linear(hidden_dim, 3)
            # self.class_embed.bias.data = torch.ones(num_classes) * bias_value
        else:
            self.class_embed = nn.Linear(2*hidden_dim, num_classes+1)
            self.class_embed_b = nn.Linear(hidden_dim, 3)
            self.class_embed.bias.data = torch.ones(num_classes+1) * bias_value
        

        if bbox_embed_diff_each_layer:
            self.bbox_embed = nn.ModuleList([MLP(hidden_dim, hidden_dim, 4, 3) for i in range(transformer.num_dec_layers)])
            for bbox_embed in self.bbox_embed:
                nn.init.constant_(bbox_embed.layers[-1].weight.data, 0)
                nn.init.constant_(bbox_embed.layers[-1].bias.data, 0)
        else:
            self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
            nn.init.constant_(self.bbox_embed.layers[-1].weight.data, 0)
            nn.init.constant_(self.bbox_embed.layers[-1].bias.data, 0)


        self.iter_update = iter_update
        if self.iter_update:
            self.transformer.decoder.bbox_embed = self.bbox_embed


        self.dropout = nn.Dropout(0.5)

        self.backbone = backbone
        self.aux_loss = aux_loss

        self.two_stage = two_stage
        self.hidden_dim = hidden_dim
        self.is_swin = "SWIN" in backbone_name
        self.generate_lfb = generate_lfb
        self.last_stride = last_stride
        self.training = training


    def freeze_params(self):
        for param in self.backbone.parameters():
            param.requires_grad = False
        for param in self.transformer.parameters():
            param.requires_grad = False
        for param in self.bbox_embed.parameters():
            param.requires_grad = False
        for param in self.input_proj.parameters():
            param.requires_grad = False
        for param in self.class_embed_b.parameters():
            param.requires_grad = False

    def forward(self, features, targets=None):
        """ The forward expects a NestedTensor, which consists of:
               - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
               - samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels

            It returns a dict with the following elements:
               - "pred_logits": the classification logits (including no-object) for all queries.
                                Shape= [batch_size x num_queries x (num_classes + 1)]
               - "pred_boxes": The normalized boxes coordinates for all queries, represented as
                               (center_x, center_y, height, width). These values are normalized in [0, 1],
                               relative to the size of each individual image (disregarding possible padding).
                               See PostProcess for information on how to retrieve the unnormalized bounding box.
               - "aux_outputs": Optional, only returned when auxilary losses are activated. It is a list of
                                dictionnaries containing the two above keys for each decoder layer.
        """
        features = make_interpolated_features(features)

        if not isinstance(features, NestedTensor):
            features = nested_tensor_from_tensor_list(features)
        
        srcs = []
        masks = []
        for l, feat in enumerate(features):
            src, mask = feat.decompose()
            srcs.append(self.input_proj[l](src))
            masks.append(mask)
            assert mask is not None
        # import ipdb; ipdb.set_trace()

        if self.num_feature_levels > len(srcs):
            _len_srcs = len(srcs)
            for l in range(_len_srcs, self.num_feature_levels):
                if l == _len_srcs:
                    src = self.input_proj[l](features[-1].tensors)
                else:
                    src = self.input_proj[l](srcs[-1])
                m = samples.mask
                mask = F.interpolate(m[None].float(), size=src.shape[-2:]).to(torch.bool)[0]
                pos_l = self.backbone[1](NestedTensor(src, mask)).to(src.dtype)
                srcs.append(src)
                masks.append(mask)
                pos.append(pos_l)        
        
        if not isinstance(samples, NestedTensor):
            samples = nested_tensor_from_tensor_list(samples)
        interpolated_features = make_interpolated_features(features)
        src, mask = features[-1].decompose()
        assert mask is not None
        # bs = samples.tensors.shape[0]
        if not self.efficient:
            embedweight = self.refpoint_embed.weight.view(self.num_queries, self.temporal_length, 4)      # nq, t, 4
        else:
            embedweight = self.refpoint_embed.weight.view(self.num_queries, 1, 4)  
        hs, cls_hs, reference  = self.transformer(self.input_proj(src), mask, embedweight, pos[-1])
        outputs_class_b = self.class_embed_b(hs)
        ######## localization head
        if not self.bbox_embed_diff_each_layer:
            reference_before_sigmoid = inverse_sigmoid(reference)
            tmp = self.bbox_embed(hs)
            tmp[..., :self.query_dim] += reference_before_sigmoid
            outputs_coord = tmp.sigmoid()
        else:
            reference_before_sigmoid = inverse_sigmoid(reference)
            outputs_coords = []
            for lvl in range(hs.shape[0]):
                # hs.shape: lay_n, bs, nq, dim
                tmp = self.bbox_embed[lvl](hs[lvl])
                tmp[..., :self.query_dim] += reference_before_sigmoid[lvl]
                outputs_coord = tmp.sigmoid()
                outputs_coords.append(outputs_coord)
            outputs_coord = torch.stack(outputs_coords)        

        ######## mix temporal features for classification
        # lay_n, bst, nq, dim = hs.shape
        # hw, bst, ch = memory.shape
        bs, _, t, h, w = src.shape
        # memory = self.encoder(memory, src.shape, mask, pos_embed)
        ##### prepare for the second decoder
        # tgt = self.patterns.weight[:, None, None, :].repeat(1, self.num_queries, bs*t, 1).flatten(0, 1) # n_q*n_pat, bs, d_model
        # embedweight = embedweight.repeat(self.num_patterns, bs, 1) # n_pat*n_q, bst, 4
        # hs_c, ref_c = self.decoder(tgt, memory, memory_key_padding_mask=mask, pos=pos_embed, refpoints_unsigmoid=embedweight)
        lay_n = self.transformer.decoder.num_layers
        # outputs_class = self.class_embed(self.dropout(hs)).reshape(lay_n, bs*t, self.num_patterns, self.num_queries, -1).max(dim = 2)[0]
        if not self.efficient:
            outputs_class = self.class_embed(self.dropout(cls_hs)).reshape(lay_n, bs*t, self.num_queries, -1)
        else:
            outputs_class = self.class_embed(self.dropout(cls_hs)).reshape(lay_n, bs, self.num_queries, -1)
        if self.dataset_mode == "ava":
            if not self.efficient:
                outputs_class = outputs_class.reshape(-1, bs, t, self.num_queries, self.num_classes)[:,:,self.temporal_length//2,:,:]
                outputs_coord = outputs_coord.reshape(-1, bs, t, self.num_queries, 4)[:,:,self.temporal_length//2,:,:]
                outputs_class_b = outputs_class_b.reshape(-1, bs, t, self.num_queries, 3)[:,:,self.temporal_length//2,:,:]
            else:
                outputs_class = outputs_class.reshape(-1, bs, self.num_queries, self.num_classes)
                outputs_coord = outputs_coord.reshape(-1, bs, self.num_queries, 4)
                outputs_class_b = outputs_class_b.reshape(-1, bs, self.num_queries, 3)
        else:
            outputs_class = outputs_class.reshape(-1, bs, t, self.num_queries, self.num_classes+1)
            outputs_coord = outputs_coord.reshape(-1, bs, t, self.num_queries, 4)
            outputs_class_b = outputs_class_b.reshape(-1, bs, t, self.num_queries, 3)
        out = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1], 'pred_logits_b': outputs_class_b[-1],}
        if self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss(outputs_class, outputs_coord, outputs_class_b)

        return out
    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord, outputs_class_b):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.

        return [{'pred_logits': a, 'pred_boxes': b, 'pred_logits_b': c}
                for a, b, c in zip(outputs_class[:-1], outputs_coord[:-1], outputs_class_b[:-1])]


class EncoderStage(nn.Module):
    def __init__(self,
                 feat_channels=256,
                 num_pts=4,
                 num_offsets=4,
                 num_levels=4,
                 feedforward_channels=2048,
                 dropout=0.1,
                 ffn_act='RelU',):

        super(EncoderStage, self).__init__()
        self.hidden_dim = feat_channels
        self.num_offsets = num_offsets # per level
        self.num_pts = num_pts # spatio-temporal
        self.feedforward_channels = feedforward_channels
        self.dropout = dropout
        self.ffn_act = ffn_act
        self.proj_layers = nn.Sequential(
            nn.Linear(3*feat_channels, feat_channels),
            nn.ReLU(),
            nn.Linear(feat_channels, feat_channels),
            nn.ReLU(),
        )
        self.offset_embed = nn.Sequential(
            nn.Linear(feat_channels//num_offsets, 3),
        )
        self.weight_o_embed = nn.Sequential(
            nn.Linear(feat_channels//num_offsets, 2),
            nn.Softmax(dim=-1),
        )
        self.weight_l_embed = nn.Sequential(
            nn.Linear(feat_channels, num_levels),
            nn.Softmax(dim=-1),
        )
        self.next_embed = nn.Sequential(
            nn.Linear(feat_channels, 3),
        )
        self.norm = nn.LayerNorm(feat_channels)

    def offset_adder(self, x, offset, indices):
        """
        offset: B, L, N, 3
        indices: a tensor of shape B, L, N
        
        return: flattened tensor of displaced indices
        """
        B, C, T, H, W, L = x.shape
        N = offset.size(2)
        w_indices = indices % W / (W-1) # B, L, N
        h_indices = indices // W % H / (H-1) # B, L, N
        t_indices = indices // W // H / (T-1) # B, L, N
        thw_indices = torch.stack([t_indices, h_indices, w_indices], dim=-1) # B, L, N, 3
        displaced_indices = (inverse_sigmoid(thw_indices) + offset).sigmoid() # B, L, N, 3
        w_ = torch.full((B,L,N,1), W-1, device=displaced_indices.device)
        h_ = torch.full((B,L,N,1), H-1, device=displaced_indices.device)
        t_ = torch.full((B,L,N,1), T-1, device=displaced_indices.device)
        displaced_indices = (displaced_indices*torch.cat([t_, h_, w_], dim=-1)).round().long() # B, L, N, 3
        flat_indices = (H*W*displaced_indices[...,0] + W*displaced_indices[...,1] + displaced_indices[...,2]) # B, L, N
        return flat_indices
        
        
    def build_flow_field(self, x, offset, indices):
        """
        x: B, C, T, H, W, L
        offset: B L N_o N c
        indices: a tensor of shape B, L, N

        output: flow field with updated indices of shape BLN_o, T, H, W, 3
        """
        B, C, T, H, W, L = x.shape
        N_o = self.num_offsets
        indices = repeat(indices, 'B L N -> B L N_o N', N_o=N_o)
        indices = rearrange(indices, 'B L N_o N -> (B L N_o) N')
        indices_ = repeat(indices, 'BLN_o N -> BLN_o N c', c=3)
        
        w_indices = indices % W / (W-1) # BLN_o, N
        h_indices = indices // W % H / (H-1) # BLN_o, N
        t_indices = indices // W // H / (T-1) # BLN_o, N
        thw_indices = torch.stack([t_indices, h_indices, w_indices], dim=-1) # BLN_o, N, 3

        sampled_indices = (inverse_sigmoid(thw_indices) + offset).sigmoid()
        src = sampled_indices*2 - 1 # BLN_o, N, 3
        
        # Build a grid
        dt = torch.linspace(-1, 1, T, device=x.device)
        dh = torch.linspace(-1, 1, H, device=x.device)
        dw = torch.linspace(-1, 1, W, device=x.device)
        mesht, meshh, meshw = torch.meshgrid((dt, dh, dw))
        grid = torch.stack((mesht, meshh, meshw), -1) # T, H, W, 3
        grid = repeat(grid, 'T H W c -> B L N_o T H W c', B=B, L=L, N_o=N_o)
        grid = rearrange(grid, 'B L N_o T H W c-> (B L N_o) (T H W) c')
        
        # Update the grid
        flow_field = grid.clone().scatter_(1, indices_, src) # (B L N_o) THW 3
        flow_field = rearrange(flow_field, 'BLN_o (T H W) c -> BLN_o T H W c', T=T, H=H, W=W)

        return flow_field
    
    def select_at(self, x, num_pts, indices=None):
        """
        selects num_pts of selected indices for each dimension.
        if no indices are given, uniformly sample.
        indices: B, L, N (flattened)
        input: B, C, T, H, W, L
        output: B, C, num_pts, num_pts, num_pts, L
        """
        B, C, T, H, W, L = x.shape
        if not indices is None:
            flat_indices = indices
        else: # uniformly sample
            hop_t = T/num_pts
            hop_h = H/num_pts
            hop_w = W/num_pts 
            t_indices = torch.linspace(0+hop_t/2, T-1-hop_t/2, num_pts, device=x.device).round().long()
            h_indices = torch.linspace(0+hop_h/2, H-1-hop_h/2, num_pts, device=x.device).round().long()
            w_indices = torch.linspace(0+hop_w/2, W-1-hop_w/2, num_pts, device=x.device).round().long()

            # make it to 1d coordinate tensor
            t_indices = repeat(t_indices, 'n_t -> B L n_t n_h n_w', B=B, L=L, n_h = num_pts, n_w = num_pts).flatten(2)
            h_indices = repeat(h_indices, 'n_h -> B L n_t n_h n_w', B=B, L=L, n_t = num_pts, n_w = num_pts).flatten(2)
            w_indices = repeat(w_indices, 'n_w -> B L n_t n_h n_w', B=B, L=L, n_t = num_pts, n_h = num_pts).flatten(2)

            indices = torch.stack([t_indices, h_indices, w_indices], dim=3) # B, L, N, 3

            flat_indices = (H*W*indices[:,:,:,0] + W*indices[:,:,:,1] + indices[:,:,:,2]) # B, L, N

        flat_indices_ = repeat(flat_indices, 'B L N -> B L N C', C=C)
        
        x_ = rearrange(x, 'B C T H W L -> B L (T H W) C')
        sampled_pts = x_.gather(dim=2, index=flat_indices_) # B, L, N, C

        return sampled_pts, flat_indices
        
    
    def forward(self, features, pos, ind=None):
        B, C, T, H, W, L = features.shape
        THW = T*H*W
        N_o = self.num_offsets
        N_p = self.num_pts
        sampled_pts, ind_pts = self.select_at(features, N_p, ind)
        sampled_pos, ind_pos = self.select_at(pos, N_p, ind)

        _, _, N, _ = sampled_pts.shape
        glob_context = features.mean(dim=(2,3,4)) # B C T H W L
        glob_context_ = repeat(glob_context, 'B C L -> B L N C', N=N)
        offset_src = torch.cat([sampled_pts, sampled_pos, glob_context_], dim=-1)
        offset_src = self.proj_layers(offset_src) # B L N 3C -> B L N C
        next = self.next_embed(offset_src) # B L N 3
        weight_l = self.weight_l_embed(offset_src) # B L N L
        
        offset_src = rearrange(offset_src, 'B L N (N_o d) -> (B L N_o) N d', N_o=N_o, d=C//N_o)
        offset = self.offset_embed(offset_src) # BLN_o N 3
        weight_o = self.weight_o_embed(offset_src)[..., 1:] # BLN_o N 1
        

        flow_field = self.build_flow_field(features, offset, ind_pts) # BLN_o T H W 3
        
        features_ = repeat(features, 'B C T H W L -> B N_o C T H W L', N_o=N_o)
        features_ = rearrange(features_, 'B N_o C T H W L -> (B L N_o) C T H W')
        # rearrange(repeat(features, 'B C T H W L -> B N_o C T H W L', N_o=N_o), 'B N_o C T H W L -> (B L N_o) C T H W')
        ## here the memory increases by 1.2GB

        sampled_features = F.grid_sample(
            features_, flow_field,
            mode='bilinear', padding_mode='zeros', align_corners=False,
        ) # (B L N_o) C THW 
        ## 1.5GB
        
        sampled_features = rearrange(sampled_features, 'BLN_o C T H W -> BLN_o T H W C')
        weight_o_ = torch.zeros((B*L*N_o, THW, 1), device=weight_o.device)
        ind_pts_o = repeat(ind_pts, 'B L N -> B L N_o N c', N_o=N_o, c=1)
        ind_pts_o = rearrange(ind_pts_o, 'B L N_o N c -> (B L N_o) N c')
        weight_o = weight_o_.scatter_(1, ind_pts_o, weight_o)
        weight_o = rearrange(weight_o, 'BLN_o (T H W) c -> BLN_o T H W c', T=T, H=H, W=W)

        weighted_sampled_features = rearrange(
            sampled_features*weight_o, '(B L N_o) T H W C -> B N_o C T H W L', B=B, L=L, N_o=N_o).sum(dim=1)
        # this operation adds 2GB
        
        features = features+weighted_sampled_features
        
        weight_l_ = torch.zeros((B, L, THW, L), device=weight_l.device)
        ind_pts_l = repeat(ind_pts, 'B L N -> B L N C', C=4)
        weight_l = weight_l_.scatter_(2, ind_pts_l, weight_l) # B L THW L
        weight_l = rearrange(weight_l, 'B L (T H W) l -> B T H W L l', T=T, H=H, W=W)
        weight_l = repeat(weight_l, 'B T H W L l -> B c T H W L l', c=1)
    
        features = (features[..., None]*weight_l).sum(dim=-1) # B C T H W L

        features = self.norm(features.transpose(1,-1).contiguous()).transpose(1,-1).contiguous()

        next_ind = self.offset_adder(features, next, ind_pts)
        return features, next_ind


class STMDecoder(nn.Module):

    def __init__(self, cfg):

        super(STMDecoder, self).__init__()
        self.device = torch.device('cuda')

        self._generate_queries(cfg)
        self.num_enc_stages = cfg.MODEL.STM.NUM_ENC_STAGES
        self.num_dec_stages = cfg.MODEL.STM.NUM_STAGES
        self.encoder_stages = nn.ModuleList()
        self.decoder_stages = nn.ModuleList()
        self.query_dim = 4
        self.eff = True
        self.d_model = cfg.MODEL.STM.HIDDEN_DIM
        assert self.query_dim in [2, 4]
        self.query_scale_type = 'cond_elewise'
        assert self.query_scale_type in ['cond_elewise', 'cond_scalar', 'fix_elewise']        
        
        for i in range(self.num_enc_stages):
            encoder_stage = EncoderStage(
                feat_channels=cfg.MODEL.STM.HIDDEN_DIM,
                num_pts=cfg.MODEL.STM.SAMPLING_POINTS,
                feedforward_channels=cfg.MODEL.STM.DIM_FEEDFORWARD,
                dropout=cfg.MODEL.STM.DROPOUT,
                ffn_act=cfg.MODEL.STM.ACTIVATION,
            )
            self.encoder_stages.append(encoder_stage)
        # for i in range(self.num_stages):
        #     decoder_stage = DecoderStage(
        #         d_model=cfg.MODEL.STM.HIDDEN_DIM,
        #         num_heads=cfg.MODEL.STM.NUM_HEADS,
        #         dim_feedforward=cfg.MODEL.STM.DIM_FEEDFORWARD,
        #         dropout=cfg.MODEL.STM.DROPOUT,
        #         activation=cfg.MODEL.STM.ACTIVATION,
        #         normalize_before=False,
        #         keep_query_pos=False,
        #         )
        #     self.decoder_stages.append(decoder_stage)
        
        decoder_layer = DecoderStage(
                d_model=cfg.MODEL.STM.HIDDEN_DIM,
                nhead=cfg.MODEL.STM.NUM_HEADS,
                dim_feedforward=cfg.MODEL.STM.DIM_FEEDFORWARD,
                dropout=cfg.MODEL.STM.DROPOUT,
                activation=cfg.MODEL.STM.ACTIVATION,
                )
        
        decoder_norm = nn.LayerNorm(cfg.MODEL.STM.HIDDEN_DIM)
        self.decoder = TransformerDecoder(decoder_layer,
                                          num_layers=self.num_dec_stages,
                                          norm=decoder_norm,
                                          return_intermediate=True,
                                          d_model=cfg.MODEL.STM.HIDDEN_DIM,
                                          query_dim=self.query_dim,
                                          keep_query_pos=False,
                                          modulate_hw_attn=True,
                                          bbox_embed_diff_each_layer=False
        )
        
        object_weight = cfg.MODEL.STM.OBJECT_WEIGHT
        giou_weight = cfg.MODEL.STM.GIOU_WEIGHT
        l1_weight = cfg.MODEL.STM.L1_WEIGHT
        action_weight = cfg.MODEL.STM.ACTION_WEIGHT
        background_weight = cfg.MODEL.STM.BACKGROUND_WEIGHT
        weight_dict = {"loss_ce": object_weight,
                       "loss_bbox": l1_weight,
                       "loss_giou": giou_weight,
                       "loss_bce": action_weight}
        self.person_threshold = cfg.MODEL.STM.PERSON_THRESHOLD


        matcher = HungarianMatcher(cfg=cfg,
                                   cost_class=object_weight,
                                   cost_bbox=l1_weight,
                                   cost_giou=giou_weight,
                                   use_focal=False)

        self.intermediate_supervision = cfg.MODEL.STM.INTERMEDIATE_SUPERVISION
        if self.intermediate_supervision:
            for i in range(self.num_dec_stages - 1):
                inter_weight_dict = {k + f"_{i}": v for k, v in weight_dict.items()}
                weight_dict.update(inter_weight_dict)


        losses = ["labels", "boxes"]
        self.criterion = SetCriterion(cfg=cfg,
                                      num_classes=cfg.MODEL.STM.OBJECT_CLASSES,
                                      matcher=matcher,
                                      weight_dict=weight_dict,
                                      eos_coef=background_weight,
                                      losses=losses,
                                      use_focal=False)
        
        self.num_frames = cfg.DATA.NUM_FRAMES

    def _generate_queries(self, cfg):
        self.num_queries = cfg.MODEL.STM.NUM_QUERIES
        self.hidden_dim = cfg.MODEL.STM.HIDDEN_DIM
        self.num_classes = cfg.MODEL.STM.ACTION_CLASSES
        
        # Build queries
        self.refpoint_embed = nn.Embedding(self.num_queries, 4)
        self.cls_queries = nn.Embedding(self.num_classes, self.hidden_dim)

    def _decode_init_queries(self, whwh):
        
        batch_size = len(whwh)
        init_spatial_queries = self.refpoint_embed.weight.clone()
        init_spatial_queries.data[:, :2].uniform_(0,1)
        init_spatial_queries.data[:, :2] = inverse_sigmoid(init_spatial_queries.data[:, :2])
        init_spatial_queries = init_spatial_queries[None].expand(batch_size, *init_spatial_queries.size())
        init_spatial_queries.data[:, :2].requires_grad = False
        
        init_cls_queries = self.cls_queries.weight.clone()
        init_cls_queries = torch.layer_norm(init_cls_queries,
                                                 normalized_shape=[init_cls_queries.size(-1)])        

        return init_spatial_queries, init_cls_queries

    def person_detector_loss(self, outputs_class, outputs_coord, criterion, targets, outputs_actions):
        if self.intermediate_supervision:
            output = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1], 'pred_actions':outputs_actions[-1],
                      'aux_outputs': [{'pred_logits': a, 'pred_boxes': b, 'pred_actions':c}
                                      for a, b, c in zip(outputs_class[:-1], outputs_coord[:-1], outputs_actions[:-1])]}
        else:
            raise NotImplementedError

        loss_dict = criterion(output, targets)
        return loss_dict


    def make_targets(self, gt_boxes, whwh, labels):
        targets = []
        for box_in_clip, frame_size, label in zip(gt_boxes, whwh, labels):
            target = {}
            target['action_labels'] = torch.tensor(label, dtype=torch.float32, device=self.device)
            target['boxes_xyxy'] = torch.tensor(box_in_clip, dtype=torch.float32, device=self.device)
            # num_box, 4 (x1,y1,x2,y2) w.r.t augmented images unnormed
            target['labels'] = torch.zeros(len(target['boxes_xyxy']), dtype=torch.int64, device=self.device)
            target["image_size_xyxy"] = frame_size.to(self.device)
            # (4,) whwh
            image_size_xyxy_tgt = frame_size.unsqueeze(0).repeat(len(target['boxes_xyxy']), 1)
            target["image_size_xyxy_tgt"] = image_size_xyxy_tgt.to(self.device)
            # (num_box, 4) whwh
            targets.append(target)

        return targets

    def forward(self, features, pos, whwh, gt_boxes, labels, extras={}, part_forward=-1):
        """
        features: list of tensors
        each element is of shape B, 
        """
        features, pos = make_interpolated_features(features, pos, num_frames=self.num_frames, level=2)
        srcs = torch.stack([feature.tensors for feature in features], dim=-1) # B, C, T, H, W, L
        masks = torch.stack([feature.mask for feature in features], dim=-1) # B, T, H, W, L        
        pos = torch.stack(pos, dim=-1) # B, C, T, H, W, L

        ind = None

        for l, encoder_stage in enumerate(self.encoder_stages):
            srcs, ind = encoder_stage(srcs, pos, ind)
            print("stage ", l, " passed")

        spatial_queries, class_queries = self._decode_init_queries(whwh)
        # B, N_q, query_dim // N_c, D
        # inter_class_logits = []
        # inter_pred_bboxes = []
        # inter_action_logits = []
        
        B, C, T, H, W, L = srcs.shape
        
        mask = features[-1].mask # B, T, H, W
        pos = pos[..., -1] # B, C, T, H, W
        
        N_q = self.num_queries
        refpoint_embed = spatial_queries.transpose(0,1)
        memory = rearrange(srcs.mean(dim=-1), 'B C T H W -> (H W) (B T) C')
        mask = rearrange(mask, 'B T H W -> B T (H W)')
        pos_embed = rearrange(pos, 'B C T H W -> (H W) (B T) C')
        tgt = torch.zeros(N_q, B*T, self.d_model, device=refpoint_embed.device)
        if self.eff:
            memory = memory.reshape(-1, B, T, C)[:,:,T//2:T//2+1,:].flatten(1,2)
            pos_embed = pos_embed.reshape(-1, B, T, C)[:,:,T//2:T//2+1,:].flatten(1,2)
            mask = mask.reshape(B, T, -1)[:,T//2:T//2+1,:].flatten(0,1)
            tgt = torch.zeros(N_q, B, self.d_model, device=refpoint_embed.device)
        else:
            tgt = torch.zeros(N_q, B*T, self.d_model, device=refpoint_embed.device)        
        
        hs, cls_hs, references = self.decoder(tgt,
                                              memory,
                                              memory_key_padding_mask=mask, 
                                              pos=pos_embed,
                                              refpoints_unsigmoid=refpoint_embed,
                                              class_queries=class_queries,
                                              orig_res=(H,W))            
        
        import pdb; pdb.set_trace()
            # objectness_score, action_score, delta_xyzr, spatial_queries, temporal_queries = \
                # decoder_stage(features, proposal_boxes, spatial_queries, temporal_queries)
            # proposal_boxes, pred_boxes = decoder_stage.refine_xyzr(proposal_boxes, delta_xyzr)

        if not self.training:
            action_scores = torch.sigmoid(inter_action_logits[-1])
            scores = F.softmax(inter_class_logits[-1], dim=-1)[:, :, 0]
            # scores: B*100
            action_score_list = []
            box_list = []
            for i in range(B):
                selected_idx = scores[i] >= self.person_threshold
                if not any(selected_idx):
                    _,selected_idx = torch.topk(scores[i],k=3,dim=-1)

                action_score = action_scores[i][selected_idx]
                box = inter_pred_bboxes[-1][i][selected_idx]
                cur_whwh = whwh[i]
                box = clip_boxes_tensor(box, cur_whwh[1], cur_whwh[0])
                box[:, 0::2] /= cur_whwh[0]
                box[:, 1::2] /= cur_whwh[1]
                action_score_list.append(action_score)
                box_list.append(box)
            return action_score_list, box_list


        targets = self.make_targets(gt_boxes, whwh, labels)
        losses = self.person_detector_loss(inter_class_logits, inter_pred_bboxes, self.criterion, targets, inter_action_logits)
        weight_dict = self.criterion.weight_dict
        for k in losses.keys():
            if k in weight_dict:
                losses[k] *= weight_dict[k]
        return losses

def build_stm_decoder(cfg):
    return STMDecoder(cfg)
