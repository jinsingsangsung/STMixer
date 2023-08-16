import math
import torch
from torch import Tensor
from typing import Optional, List
from ..modeling.stm_decoder.util.misc import NestedTensor

# def _max_by_axis(the_list):
#     # type: (List[List[int]]) -> List[int]
#     maxes = the_list[0]
#     for sublist in the_list[1:]:
#         for index, item in enumerate(sublist):
#             maxes[index] = max(maxes[index], item)
#     return maxes


# class NestedTensor(object):
#     def __init__(self, tensors, mask: Optional[Tensor]):
#         self.tensors = tensors
#         self.mask = mask

#     def to(self, device):
#         # type: (Device) -> NestedTensor # noqa
#         cast_tensor = self.tensors.to(device)
#         mask = self.mask
#         if mask is not None:
#             assert mask is not None
#             cast_mask = mask.to(device)
#         else:
#             cast_mask = None
#         return NestedTensor(cast_tensor, cast_mask)

#     def decompose(self):
#         return self.tensors, self.mask

#     def __repr__(self):
#         return str(self.tensors)


# def nested_tensor_from_tensor_list(tensor_list: List[Tensor]):

#     # docs make this more general
#     if tensor_list[0].ndim == 3:
#         # docs make it support different-sized images
#         max_size = _max_by_axis([list(img.shape) for img in tensor_list])

#         # min_size = tuple(min(s) for s in zip(*[img.shape for img in tensor_list]))
#         batch_shape = [len(tensor_list)] + max_size

#         b, c, h, w = batch_shape
#         dtype = tensor_list[0].dtype
#         device = tensor_list[0].device
#         tensor = torch.zeros(batch_shape, dtype=dtype, device=device)
#         mask = torch.ones((b, h, w), dtype=torch.bool, device=device)
#         for img, pad_img, m in zip(tensor_list, tensor, mask):
#             pad_img[: img.shape[0], : img.shape[1], : img.shape[2]].copy_(img)
#             m[: img.shape[1], :img.shape[2]] = False

#     elif tensor_list[0].ndim == 4:
#         max_size = _max_by_axis([list(clip.shape) for clip in tensor_list])

#         # min_size = tuple(min(s) for s in zip(*[img.shape for img in tensor_list]))
#         batch_shape = [len(tensor_list)] + max_size
#         b, c, t, h, w = batch_shape
#         dtype = tensor_list[0].dtype
#         device = tensor_list[0].device
#         tensor = torch.zeros(batch_shape, dtype=dtype, device=device)
#         mask = torch.ones((b, h, w), dtype=torch.bool, device=device)
#         for img, pad_img, m in zip(tensor_list, tensor, mask):
#             pad_img[: img.shape[0], : img.shape[1], : img.shape[2], : img.shape[3]].copy_(img)
#             m[: img.shape[2], :img.shape[3]] = False
#     else:
#         raise ValueError('not supported')
#     return NestedTensor(tensor, mask)



def batch_different_videos(videos, size_divisible=0):
    '''
    :param videos: a list of video tensors
    :param size_divisible: output_size(width and height) should be divisble by this param
    :return: batched videos as a single tensor
    '''
    assert isinstance(videos, (tuple, list))
    max_size = tuple(max(s) for s in zip(*[clip.shape for clip in videos]))

    if size_divisible > 0:
        stride = size_divisible
        max_size = list(max_size)
        max_size[2] = int(math.ceil(max_size[2] / stride) * stride)
        max_size[3] = int(math.ceil(max_size[3] / stride) * stride)
        max_size = tuple(max_size)

    batch_shape = (len(videos),) + max_size
    b, c, t, h, w = batch_shape
    dtype = videos[0].dtype
    device = videos[0].device
    tensor = torch.zeros(batch_shape, dtype=dtype, device=device)
    mask = torch.ones((b, h, w), dtype=torch.bool, device=device)
    for img, pad_img, m in zip(videos, tensor, mask):
        pad_img[: img.shape[0], : img.shape[1], : img.shape[2], : img.shape[3]].copy_(img)
        m[: img.shape[2], :img.shape[3]] = False
    # batched_clips = videos[0].new(*batch_shape).zero_()
    # for clip, pad_clip in zip(videos, batched_clips):
    #     pad_clip[:clip.shape[0], :clip.shape[1], :clip.shape[2], :clip.shape[3]].copy_(clip)

    return NestedTensor(tensor, mask)


class BatchCollator(object):
    """
    From a list of samples from the dataset,
    returns the batched objectimages and targets.
    This should be passed to the DataLoader
    """

    def __init__(self, size_divisible=0):
        self.divisible = size_divisible
        self.size_divisible = self.divisible

    def __call__(self, batch):
        transposed_batch = list(zip(*batch))
        slow_clips = batch_different_videos(transposed_batch[0], self.size_divisible)
        if transposed_batch[1][0] is not None:
            fast_clips = batch_different_videos(transposed_batch[1], self.size_divisible)
        else:
            fast_clips = None
        whwh = torch.stack(transposed_batch[2])
        boxes = transposed_batch[3]
        label_arrs = transposed_batch[4]
        metadata = transposed_batch[5]
        clip_ids = transposed_batch[6]
        return slow_clips, fast_clips, whwh, boxes, label_arrs, metadata, clip_ids