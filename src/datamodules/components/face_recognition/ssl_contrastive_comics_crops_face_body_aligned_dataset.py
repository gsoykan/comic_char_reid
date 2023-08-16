import os
from typing import Optional, Any, List, Dict

import numpy as np
import pandas as pd
from PIL import Image
from pandas.core.groupby import SeriesGroupBy
from src.utils.pytorch_utils import visualize_tensor, UnNormalize
from torch.utils.data import Dataset
from tqdm import tqdm

from src.datamodules.components.vision_transform_setting import VisionTransformSetting, ContrastiveTransformations


class SSLContrastiveComicsCropsFaceBodyAlignedDataset(Dataset):
    def __init__(self,
                 data_dir: str,
                 face_prefiltered_csv_folder_dir: str,
                 face_contrastive_transform: Any,
                 face_weak_transform: Any,
                 body_prefiltered_csv_folder_dir: str,
                 body_contrastive_transform: Any,
                 body_weak_transform: Any,
                 face_body_csv_path: str,
                 face_folder_dir: Optional[str] = None,
                 body_folder_dir: Optional[str] = None,
                 limit_search_files: Optional[int] = None,
                 is_torch_transform: bool = False,
                 is_torch_transform_weak: bool = False,
                 disable_alignment: bool = False,):
        self.disable_alignment = disable_alignment
        self.data_dir = data_dir
        self.face_prefiltered_csv_folder_dir = face_prefiltered_csv_folder_dir
        self.face_contrastive_transform = face_contrastive_transform
        self.face_folder_dir = face_folder_dir
        self.body_prefiltered_csv_folder_dir = body_prefiltered_csv_folder_dir
        self.body_contrastive_transform = body_contrastive_transform
        self.body_folder_dir = body_folder_dir
        self.face_body_csv_path = os.path.join(data_dir, face_body_csv_path)
        self.face_weak_transform = face_weak_transform
        self.body_weak_transform = body_weak_transform
        self.limit_search_files = limit_search_files
        self.is_torch_transform = is_torch_transform
        self.is_torch_transform_weak = is_torch_transform_weak
        self.dataset = self.load_dataset()

    def read_face_body_csv_to_get_pair(self, by_series_and_page, series_id, page_panel_id, item_idx, item_type):
        item_idx = int(item_idx.split('.')[0])

        panel_df_records = self.get_from_group(by_series_and_page, series_id, page_panel_id)

        item_row = None
        char_idx = None
        for row in panel_df_records:
            if row['index'] == item_idx and row['type'] == item_type:
                item_row = row
                char_idx = row['char_index']
                break

        if item_row is None:
            return None

        pair_item_row = None
        pair_item_type = 'face' if item_type == 'body' else 'body'
        for row in panel_df_records:
            if row['char_index'] == char_idx and row['type'] == pair_item_type:
                pair_item_row = row
                break

        return pair_item_row

    def get_from_group(self,
                       by_page_and_panel: SeriesGroupBy,
                       seq_id: str,
                       page_panel_id: str) -> List[Dict]:
        try:
            rows = by_page_and_panel.get_group((int(seq_id), page_panel_id))
            return rows.to_dict('records')
        except KeyError:
            return []

    def load_dataset(self):
        face_body_index_df = pd.read_csv(self.face_body_csv_path)
        by_series_and_page = face_body_index_df.groupby(['series_id', 'page_id'])

        def construct_img_path(item_csv_path: str):
            series_id, page_panel_id, img_type, img_idx_path = item_csv_path.split('/')[-4:]
            pair_row = self.read_face_body_csv_to_get_pair(by_series_and_page,
                                                           series_id, page_panel_id, img_idx_path,
                                                           'face' if img_type in 'faces' else 'body')

            def construct(img_type, img_idx_path):
                if img_type in ['body', 'bodies']:
                    img_path = os.path.join(self.body_folder_dir, str(series_id), f'{page_panel_id}',
                                            'bodies',
                                            f'{img_idx_path}')
                else:
                    img_path = os.path.join(self.face_folder_dir, str(series_id), f'{page_panel_id}',
                                            f'{img_idx_path}')
                return img_path

            img_path = construct(img_type, img_idx_path)
            if pair_row is not None:
                pair_img_path = construct(pair_row['type'], str(pair_row['index']) + '.jpg')
            else:
                # for now let's just use samples with pairs...
                return None

            # returns face, body paths
            if img_type in ['face', 'faces']:
                return img_path, pair_img_path
            else:
                return pair_img_path, img_path

        print('loading face dataset')
        face_dataset = pd.read_csv(self.face_prefiltered_csv_folder_dir)['img_path'].tolist()
        if self.limit_search_files is not None:
            face_dataset = face_dataset[:self.limit_search_files]
        face_dataset = list(
            tqdm(
                filter(None, map(lambda x: construct_img_path(x), face_dataset)),
                total=len(face_dataset)
            )
        )

        print('loading body dataset')
        body_dataset = pd.read_csv(self.body_prefiltered_csv_folder_dir)['img_path'].tolist()
        if self.limit_search_files is not None:
            body_dataset = body_dataset[:self.limit_search_files]
        body_dataset = list(
            tqdm(
                filter(None, map(lambda x: construct_img_path(x), body_dataset)),
                total=len(body_dataset)
            )
        )

        dataset = list({*face_dataset, *body_dataset})

        return dataset

    def __getitem__(self, index):
        face_raw_item, body_raw_item = self.dataset[index]

        def get_face():
            raw_image = Image.open(face_raw_item)

            if self.is_torch_transform:
                transformeds = self.face_contrastive_transform(raw_image)
            else:
                transformeds = self.face_contrastive_transform(image=np.array(raw_image))['image']

            if self.disable_alignment:
                return transformeds, None

            if self.is_torch_transform_weak:
                weak_transformed_face = self.face_weak_transform(raw_image)
            else:
                weak_transformed_face = self.face_weak_transform(image=np.array(raw_image))['image']
            return transformeds, weak_transformed_face

        def get_body():
            raw_image = Image.open(body_raw_item)

            if self.is_torch_transform:
                transformeds = self.body_contrastive_transform(raw_image)
            else:
                transformeds = self.body_contrastive_transform(image=np.array(raw_image))['image']

            if self.disable_alignment:
                return transformeds, None

            if self.is_torch_transform_weak:
                weak_transformed_body = self.body_weak_transform(raw_image)
            else:
                weak_transformed_body = self.body_weak_transform(image=np.array(raw_image))['image']
            return transformeds, weak_transformed_body

        face_transformeds, weak_face = get_face()
        body_transformeds, weak_body = get_body()

        if self.disable_alignment:
            return face_transformeds, body_transformeds
        else:
            return face_transformeds, body_transformeds, [weak_face, weak_body]

    def __len__(self):
        return len(self.dataset)


if __name__ == '__main__':
    data_dir = '/scratch/users/gsoykan20/projects/amazing-mysteries-of-gutter-demystified/data'
    # data_dir = '/home/gsoykan20/Desktop/self_development/amazing-mysteries-of-gutter-demystified/data/'
    face_prefiltered_csv_folder_dir = f'{data_dir}/ssl/filtered_all_face_100k.csv'
    body_prefiltered_csv_folder_dir = f'{data_dir}/ssl/filtered_all_body_100k.csv'
    transform = ContrastiveTransformations(VisionTransformSetting.SIMCLR_PRETRAIN_TORCH.get_transformation())
    dataset = SSLContrastiveComicsCropsFaceBodyAlignedDataset(data_dir=data_dir,
                                                              face_prefiltered_csv_folder_dir=face_prefiltered_csv_folder_dir,
                                                              face_contrastive_transform=transform,
                                                              face_weak_transform=transform,
                                                              body_prefiltered_csv_folder_dir=body_prefiltered_csv_folder_dir,
                                                              body_contrastive_transform=transform,
                                                              body_weak_transform=transform,
                                                              face_folder_dir='/scratch/users/gsoykan20/projects/DASS_Det_Inference/comics_crops_large_faces/',
                                                              body_folder_dir='/datasets/COMICS/comics_crops',
                                                              face_body_csv_path='ssl/merged_face_body.csv',
                                                              limit_search_files=None,
                                                              is_torch_transform=True)
    breakpoint()
    unorm = UnNormalize(mean=[0.5, 0.5, 0.5, ],
                        std=[0.5, 0.5, 0.5, ], )
    for (e, e_prime) in dataset:
        visualize_tensor(e, unorm)
        visualize_tensor(e_prime, unorm)
        print(e)
        print(e, e_prime)
