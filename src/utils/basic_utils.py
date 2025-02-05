import glob
import itertools
import json
import os
import time
from enum import Enum
from typing import Callable, List, Optional, Tuple, Dict

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from shapely import affinity
from shapely.geometry import box
from tqdm import tqdm


def search_files(extension='.ttf',
                 folder='H:\\',
                 filename_condition: Optional[Callable[[str], bool]] = None,
                 limit: Optional[int] = None,
                 enable_tqdm: bool = False):
    if limit:
        files = []
        for r, d, f in (tqdm(os.walk(folder), desc='walking in folders...') if enable_tqdm else os.walk(folder)):
            if limit is not None and len(files) >= limit:
                break
            for file in f:
                if file.endswith(extension):
                    filename = r + "/" + file
                    if filename_condition is not None:
                        if filename_condition(filename):
                            files.append(filename)
                    else:
                        files.append(filename)
        return files
    else:
        return alternative_search_files_2(extension, folder)


def scandir_walk(top):
    for entry in os.scandir(top):
        if entry.is_dir(follow_symlinks=False):
            yield from scandir_walk(entry.path)
        else:
            yield entry.path


# this is way faster...
def alternative_search_files_2(extension=".ttf", folder="H:\\") -> List[str]:
    return [
        file for file in tqdm(scandir_walk(folder), desc="walking in folder..")
        if file.endswith(extension)
    ]


def read_or_get_image(img,
                      read_rgb: bool = False):
    img_str = ""
    if not isinstance(img, (np.ndarray, str)):
        raise AssertionError('Images must be strings or numpy arrays')

    if isinstance(img, str):
        img_str = img
        img = cv2.imread(img)

    if img is None:
        raise AssertionError('Image could not be read: ' + img_str)

    if read_rgb:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    return img


def read_or_get_image_masked(img,
                             read_rgb: bool = True,
                             masks: List = [],
                             mask_fill_value=0,
                             return_pil_image: bool = True):
    read_image = read_or_get_image(img, read_rgb=read_rgb)
    for bb in masks:
        read_image[int(bb[1]): int(bb[3]), int(bb[0]):int(bb[2])] = mask_fill_value
    return Image.fromarray(read_image) if return_pil_image else read_image


# source: https://stackoverflow.com/a/42728126/8265079 | https://stackoverflow.com/questions/42727586/nest-level-of-a-list
def nest_level(obj):
    # Not a list? So the nest level will always be 0:
    if type(obj) != list:
        return 0
    # Now we're dealing only with list objects:
    max_level = 0
    for item in obj:
        # Getting recursively the level for each item in the list,
        # then updating the max found level:
        max_level = max(max_level, nest_level(item))
    # Adding 1, because 'obj' is a list (here is the recursion magic):
    return max_level + 1


def read_ad_pages(ad_page_path: str = '../../data/ad_pages_original.txt'):
    with open(ad_page_path) as f:
        lines = f.readlines()
    ad_pages = []
    for line in lines:
        comic_no, page_no = line.strip().split('---')
        ad_pages.append((int(comic_no), int(page_no)))
    return ad_pages


def flatten_list(l: List):
    return list(itertools.chain(*l))


def all_equal(lst):
    return all(element == lst[0] for element in lst)


def map_values(obj: Dict, fn):
    return dict((k, fn(v)) for k, v in obj.items())


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)


def inverse_normalize(tensor,
                      mean=(0.485, 0.456, 0.406),
                      std=(0.229, 0.224, 0.225),
                      in_place: bool = True):
    mean = torch.as_tensor(mean, dtype=tensor.dtype, device=tensor.device)
    std = torch.as_tensor(std, dtype=tensor.dtype, device=tensor.device)
    if mean.ndim == 1:
        mean = mean.view(-1, 1, 1)
    if std.ndim == 1:
        std = std.view(-1, 1, 1)
    if not in_place:
        tensor = tensor.clone()
    tensor.mul_(std).add_(mean)
    return tensor


class OCRFileKey(str, Enum):
    COMIC_NO = 'comic_no'
    PAGE_NO = 'page_no'
    PANEL_NO = 'panel_no'
    TEXTBOX_NO = 'textbox_no'
    DIALOG_OR_NARRATION = 'dialog_or_narration'
    TEXT = 'text'
    x1 = 'x1'
    y1 = 'y1'
    x2 = 'x2'
    y2 = 'y2'


def merge_pt_boxes(box1: torch.Tensor, box2) -> torch.Tensor:
    xmin = torch.min(box1[0], box2[0])
    ymin = torch.min(box1[1], box2[1])
    xmax = torch.max(box1[2], box2[2])
    ymax = torch.max(box1[3], box2[3])
    return torch.tensor([xmin, ymin, xmax, ymax], dtype=box1.dtype, device=box1.device)


def box_intersection_rate(source_bb: List, target_bb: List) -> float:
    box_1, box_2 = box(*source_bb), box(*target_bb)
    if box_1.area == 0:
        return 0
    return box_1.intersection(box_2).area / box_1.area


def box_to_box_center_distance(box1: torch.Tensor,
                               box2: torch.Tensor):
    # Calculate the center coordinates of each box
    center1_x = (box1[0] + box1[2]) / 2  # Scalar
    center1_y = (box1[1] + box1[3]) / 2  # Scalar
    center2_x = (box2[0] + box2[2]) / 2  # Scalar
    center2_y = (box2[1] + box2[3]) / 2  # Scalar

    # Calculate the absolute center-to-center distance between the boxes
    center_distance = torch.sqrt(torch.pow(center1_x - center2_x, 2) + torch.pow(center1_y - center2_y, 2))  # Scalar
    return center_distance


def sort_elements_by_z_order(box_list: List[torch.Tensor],
                             grid_count: float = 4.0,
                             normalization_box: Optional[torch.Tensor] = None):
    if len(box_list) == 0:
        return []

    boxes = torch.stack(box_list, dim=0)

    # Calculate the center coordinates of each bounding box
    x_centers = (boxes[:, 0] + boxes[:, 2]) / 2
    y_centers = (boxes[:, 1] + boxes[:, 3]) / 2

    x_centers_min = x_centers.min()
    x_centers_max = x_centers.max()
    y_centers_min = y_centers.min()
    y_centers_max = y_centers.max()

    if normalization_box is not None:
        x_centers_min = normalization_box[[0, 2]].min()
        x_centers_max = normalization_box[[0, 2]].max()
        y_centers_min = y_centers.min()  # normalization_box[[1, 3]].min() - this makes top bubbles to snap together
        y_centers_max = normalization_box[[1, 3]].max()

    # Normalize the coordinates to the range [0, 1]
    x_centers_normalized = (x_centers - x_centers_min) / (x_centers_max - x_centers_min)
    y_centers_normalized = (y_centers - y_centers_min) / (y_centers_max - y_centers_min)

    x_centers_snapped = torch.round(x_centers_normalized * grid_count) / grid_count
    y_centers_snapped = torch.round(y_centers_normalized * grid_count) / grid_count

    ind = np.lexsort((x_centers_snapped.cpu().numpy(), y_centers_snapped.cpu().numpy()))

    return ind


def extract_bounding_box_from_mask(mask,
                                   alternative_mask: Optional[np.ndarray] = None) -> Optional[List]:
    # Find rows and columns with non-zero values
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)

    if (not any(rows) or not any(cols)) and alternative_mask is not None:
        # then there must be alternative mask
        rows = np.any(alternative_mask, axis=1)
        cols = np.any(alternative_mask, axis=0)

    # if all false then return None
    if not any(rows) or not any(cols):
        return None

    # Find the first and last row and column indices with non-zero values
    ymin, ymax = np.where(rows)[0][[0, -1]]
    xmin, xmax = np.where(cols)[0][[0, -1]]

    # Return the bounding box coordinates as [xmin, ymin, xmax, ymax]
    return [xmin, ymin, xmax, ymax]


def smooth_edges(mask):
    # Convert the mask to binary image format (0 and 255)
    mask = mask.astype(np.uint8) * 255

    # Create a structuring element for the morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))  # Adjust the kernel size as needed

    # Apply a closing operation to close small gaps in the mask
    closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # Apply an opening operation to smooth out the edges
    smoothed = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel)

    return smoothed / 255


def bbox_area(bbox: torch.Tensor):
    width = bbox[2].item() - bbox[0].item()
    height = bbox[3].item() - bbox[1].item()
    area = width * height
    return area


def cat_df(dfs: List[Optional[pd.DataFrame]]) -> Optional[pd.DataFrame]:
    dfs = [elem for elem in dfs if elem is not None]
    if len(dfs) == 0:
        return None
    elif len(dfs) == 1:
        return dfs[0]

    return pd.concat(dfs, ignore_index=True)


def timeit(f):
    def timed(*args, **kw):
        ts = time.time()
        result = f(*args, **kw)
        te = time.time()
        print('func:%r args:[%r, %r] took: %2.4f sec' % \
              (f.__name__, args, kw, te - ts))
        return result

    return timed


@timeit
def alternative_search_files(ext: str, search_dir: str) -> List[str]:
    # Find all the files with the current extension in the search directory and its subdirectories
    files = glob.glob(os.path.join(search_dir, "**", ext), recursive=True)
    return files


def convert_raw_samples(raw_samples: str) -> List[Tuple[str, float]]:
    """
    This is primarily used for visualizing embedding projectors neighbor outputs..
    raw samples are in the following format:
    3334_12_3_face_0
0.0
1593_20_4_face_0
0.043
    """
    lines = raw_samples.splitlines()
    ids = lines[::2]
    scores = lines[1::2]
    items = []
    for item in zip(ids, scores):
        items.append((item[0], float(item[1])))
    return items


def calculate_bb_area(bb) -> float:
    return box(*bb).area


def get_iou(a, b, epsilon=1e-5):
    """Given two boxes `a` and `b` defined as a list of four numbers:
            [x1,y1,x2,y2]
        where:
            x1,y1 represent the upper left corner
            x2,y2 represent the lower right corner
        It returns the Intersect of Union score for these two boxes.
        source: http://ronny.rest/tutorials/module/localization_001/iou/
    Args:
        a:          (list of 4 numbers) [x1,y1,x2,y2]
        b:          (list of 4 numbers) [x1,y1,x2,y2]
        epsilon:    (float) Small value to prevent division by zero
    Returns:
        (float) The Intersect of Union score.
    """
    # COORDINATES OF THE INTERSECTION BOX
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])

    # AREA OF OVERLAP - Area where the boxes intersect
    width = x2 - x1
    height = y2 - y1
    # handle case where there is NO overlap
    if (width < 0) or (height < 0):
        return 0.0
    area_overlap = width * height

    # COMBINED AREA
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    area_combined = area_a + area_b - area_overlap

    # RATIO OF AREA OF OVERLAP OVER COMBINED AREA
    iou = area_overlap / (area_combined + epsilon)
    return iou


def make_bounding_box_square(x1,
                             y1,
                             x2,
                             y2,
                             scale: Optional[float] = None) -> List[int]:
    if scale is not None:
        bbox = box(x1, y1, x2, y2)
        envelope = bbox.envelope
        scaled_envelope = affinity.scale(envelope, xfact=scale, yfact=scale)
        x1, y1, x2, y2 = scaled_envelope.bounds
    # Calculate the width and height of the bounding box
    width = x2 - x1
    height = y2 - y1

    # Scale the bounding box to make it square
    if width > height:
        # The bounding box is wider than it is tall, so we need to increase the height
        y1 -= (width - height) / 2
        y2 += (width - height) / 2
    else:
        # The bounding box is taller than it is wide, so we need to increase the width
        x1 -= (height - width) / 2
        x2 += (height - width) / 2

    return int(x1), int(y1), int(x2), int(y2)
