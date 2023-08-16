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
from .util.loss import SetCriterion, HungarianMatcher
# from .transformer.dab_transformer import build_transformer
from .util.misc import (NestedTensor, nested_tensor_from_tensor_list, inverse_sigmoid)
from einops import rearrange, repeat


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

        import pdb; pdb.set_trace()
        
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
                 num_pts=8,
                 num_offsets=8,
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
            nn.Conv3d(3*feat_channels, feat_channels, (1,1,1)),
            nn.ReLU(),
            # nn.Conv3d(feat_channels, feat_channels, (1,1,1)),
            # nn.ReLU(),
        )
        self.offset_embed = nn.Sequential(
            nn.Linear(feat_channels//num_offsets, 3),
            nn.Sigmoid(),
        )
        self.weight_embed = nn.Sequential(
            nn.Linear(feat_channels//num_offsets, 2),
            nn.Softmax(dim=-1),
        )

    def build_flow_field(self, x: torch.Tensor, H, W, L):
        BT, HWL, L, N_p, _ = x.shape
        dh = torch.linspace(-1, 1, H, device=x.device)
        dw = torch.linspace(-1, 1, W, device=x.device)
        dl = torch.linspace(-1, 1, L, device=x.device)
        meshh, meshw, meshl = torch.meshgrid((dh, dw, dl))
        grid = torch.stack((meshh, meshw, meshl), -1) # H, W, L, 3
        grid = repeat(grid, 'H W L c -> BT H W L LN_p c', BT=BT, LN_p=L*N_p)
        grid = rearrange(grid, 'BT H W L LN_p c -> BT (H W L) LN_p c')
        return grid
    
    def uniform_select(self, x, num_pts):
        """
        uniformly selects num_pts for each dimension:
        input: B, C, T, H, W, L
        output: B, C, num_pts, num_pts, num_pts, L
        """
        B, C, T, H, W, L = x.shape
        hop_h = H//num_pts
        hop_w = W//num_pts
        hop_t = T//num_pts
        t_indices = torch.linspace(0+hop_t/2,T-hop_t/2,num_pts, device=x.device).round().long()
        h_indices = torch.linspace(0+hop_h/2,H-hop_h/2,num_pts, device=x.device).round().long()
        w_indices = torch.linspace(0+hop_w/2,W-hop_w/2,num_pts, device=x.device).round().long()        
        sampled_pts = x.index_select(
            dim=2, index=t_indices
            ).index_select(
                dim=3, index=h_indices
                ).index_select(
                    dim=4, index=w_indices
                    ) # B, C, num_pts, num_pts, num_pts, L
        
        return sampled_pts
        
    
    def forward(self, features, pos, layer):
        B, C, T, H, W, L = features.shape
        BT = B*T
        HWL = H*W*L
        N_o = self.num_offsets
        N_p = self.num_pts
        # features = rearrange(features, 'B C T H W L -> (B T) C H W L')
        sampled_pts = self.uniform_select(features, N_p)
        sampled_pos = self.uniform_select(pos, N_p)
        _, _, t, h, w, _ = sampled_pts.shape
        glob_context = features.mean(dim=(2,3,4), keepdim=True) # B, C, T, H, W, L
        glob_context_ = glob_context.expand(-1, -1, t, h, w, -1)
        offset_src = rearrange(torch.cat([sampled_pts, sampled_pos, glob_context_], dim=1), 'B C T H W L -> (B L) C T H W')
        import pdb; pdb.set_trace()
        offset_src = self.proj_layers(offset_src) # BL, 3C, T, H, W -> BL, C, T, H, W
        offset_src = rearrange(offset_src, 'B T H W L (N_o d) -> B T H W L N_o d', N_o=N_o, d=C//N_o)
        import pdb; pdb.set_trace()
        offset = self.offset_embed(offset_src) # B T H W L N_o 3
        weight = self.weight_embed(offset_src)[..., 1:] # B T H W L N_o 1
        import pdb; pdb.set_trace()
        dl = torch.linspace(0, 1, L, device=offset.device)[None, None, ..., None, None].expand(BT, HWL, -1, N_o, -1)
        flow_field = self.build_flow_field(offset, H, W, L) # BT (H W L) LN_o c
        offset = rearrange(torch.cat([offset, dl], dim=-1), 'bt hwl l n_o d -> bt hwl (l n_o) d')
        sample_points = (inverse_sigmoid(flow_field) + offset).sigmoid()

        sample_points = rearrange(sample_points, 'bt (h w l) ln_o c -> (bt ln_o) h w l c', h=H, w=W, l=L)
        features_ = repeat(features, 'bt c h w l -> bt ln_o c h w l', ln_o=L*N_o)
        features_ = rearrange(features_, 'bt ln_o c h w l -> (bt ln_o) c h w l')
        sampled_features = F.grid_sample(
            features_, sample_points,
            mode='bilinear', padding_mode='zeros', align_corners=False,
        )
        
        sampled_features = rearrange(sampled_features, '(bt ln_o) c h w l -> bt ln_o c h w l', bt=BT, ln_o=L*N_o)
        weight = rearrange(weight, 'bt (h w l2) l n_o c -> bt (l n_o) c h w l2', h=H, w=W, l2=L)
        features = features + (sampled_features*weight).sum(dim=1)
        
        features = rearrange(features, '(b t) c h w l -> b c t h w l', b=B, t=T)

        return features


class STMDecoder(nn.Module):

    def __init__(self, cfg):

        super(STMDecoder, self).__init__()
        self.device = torch.device('cuda')

        self._generate_queries(cfg)
        # transformer = build_transformer(cfg)
        # self.model = DETR(transformer=transformer,
        #                   num_classes=cfg.MODEL.STM.ACTION_CLASSES,
        #                   num_queries=cfg.MODEL.STM.NUM_QUERIES,
        #                   num_frames=cfg.DATA.NUM_FRAMES,
        #                   hidden_dim=cfg.MODEL.STM.HIDDEN_DIM,
        #                   temporal_length=cfg.DATA.NUM_FRAMES)
        self.num_stages = cfg.MODEL.STM.NUM_STAGES
        self.num_stages = cfg.MODEL.STM.NUM_STAGES
        self.decoder_stages = nn.ModuleList()
        for i in range(self.num_stages):
            decoder_stage = AMStage(
                query_dim=cfg.MODEL.STM.HIDDEN_DIM,
                feat_channels=cfg.MODEL.STM.HIDDEN_DIM,
                num_heads=cfg.MODEL.STM.NUM_HEADS,
                feedforward_channels=cfg.MODEL.STM.DIM_FEEDFORWARD,
                dropout=cfg.MODEL.STM.DROPOUT,
                num_ffn_fcs=cfg.MODEL.STM.NUM_FCS,
                ffn_act=cfg.MODEL.STM.ACTIVATION,
                spatial_points=cfg.MODEL.STM.SPATIAL_POINTS,
                temporal_points=cfg.MODEL.STM.TEMPORAL_POINTS,
                out_multiplier=cfg.MODEL.STM.OUT_MULTIPLIER,
                n_groups=cfg.MODEL.STM.N_GROUPS,
                num_cls_fcs=cfg.MODEL.STM.NUM_CLS,
                num_reg_fcs=cfg.MODEL.STM.NUM_REG,
                num_action_fcs=cfg.MODEL.STM.NUM_ACT,
                num_classes_object=cfg.MODEL.STM.OBJECT_CLASSES,
                num_classes_action=cfg.MODEL.STM.ACTION_CLASSES
                )
            self.decoder_stages.append(decoder_stage)
        self.encoder_stages = nn.ModuleList()
        for i in range(self.num_stages):
            encoder_stage = EncoderStage(
                feat_channels=cfg.MODEL.STM.HIDDEN_DIM,
                num_pts=cfg.MODEL.STM.SAMPLING_POINTS,
                feedforward_channels=cfg.MODEL.STM.DIM_FEEDFORWARD,
                dropout=cfg.MODEL.STM.DROPOUT,
                ffn_act=cfg.MODEL.STM.ACTIVATION,
            )
            self.encoder_stages.append(encoder_stage)
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
            for i in range(self.num_stages - 1):
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
        srcs = torch.stack([feature.tensors for feature in features], dim=-1)
        masks = torch.stack([feature.mask for feature in features], dim=-1)
        pos = torch.stack(pos, dim=-1)
        for l, encoder_stage in enumerate(self.encoder_stages):
            srcs = encoder_stage(srcs, pos, l)
            print(l)

        spatial_queries, class_queries = self._decode_init_queries(whwh)

        inter_class_logits = []
        inter_pred_bboxes = []
        inter_action_logits = []
        B, N, _ = spatial_queries.size()

        for decoder_stage in self.decoder_stages:
            objectness_score, action_score, delta_xyzr, spatial_queries, temporal_queries = \
                decoder_stage(features, proposal_boxes, spatial_queries, temporal_queries)
            proposal_boxes, pred_boxes = decoder_stage.refine_xyzr(proposal_boxes, delta_xyzr)

            inter_class_logits.append(objectness_score)
            inter_pred_bboxes.append(pred_boxes)
            inter_action_logits.append(action_score)


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
