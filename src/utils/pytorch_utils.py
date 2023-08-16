from typing import Dict, Optional, Any

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms.functional as F
from torch import Tensor
from torch.utils.data.dataloader import default_collate


# source: https://discuss.pytorch.org/t/first-nonzero-index/24769/3


class LambdaLayer(torch.nn.Module):
    def __init__(self, func):
        super().__init__()
        self.func = func

    def forward(self, x):
        return self.func(x)


def find_first_zero_value_on_column_or_return_length(matrix: Tensor):
    matrix_size = matrix.size()
    # zero values mask
    zero_mask = matrix == 0
    # operations on the mask to find first zero values in the columns
    mask_max_values, mask_max_indices = torch.max(zero_mask, dim=0)
    # if the max-mask is non zero, there is no zero value in the column
    mask_max_indices[mask_max_values == False] = matrix_size[0]
    return mask_max_indices.cpu().detach().numpy()


def collate_without_none_element(batch):
    batch = list(filter(lambda x: x is not None, batch))
    return default_collate(batch)


def visualize_tensor(imgs_tensor: Tensor,
                     unorm: Optional[Any] = None,
                     save_path: Optional[str] = None):
    if not isinstance(imgs_tensor, list) and len(imgs_tensor.shape) < 4:
        imgs_tensor = [imgs_tensor]
    fig, axs = plt.subplots(ncols=len(imgs_tensor), squeeze=False)
    for i, img in enumerate(imgs_tensor):
        img = img.detach()
        img = unorm(img) if unorm is not None else img
        img = F.to_pil_image(img)
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

    # Save the image if save_path is provided
    if save_path is not None:
        plt.savefig(save_path)

    plt.show()


def visualize_vision_batch(batch: Dict):
    vision_input = batch.get('vision_input')
    vision_output = batch.get('vision_output')

    unorm = UnNormalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])

    for vision_i in vision_input:
        for i in range(len(vision_i)):
            visualize_tensor(vision_i[i], unorm)

    for vision_o in vision_output:
        for i in range(len(vision_o)):
            visualize_tensor(vision_o[i], unorm)


class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
            # The normalize code -> t.sub_(m).div_(s)
        return tensor
