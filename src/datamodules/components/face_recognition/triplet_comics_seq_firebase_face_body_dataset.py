import itertools
import json
import os
import random
from typing import Optional, Dict, Tuple

import albumentations as A
import pandas as pd
from torch.utils.data import Dataset
from PIL import Image
import torch

from src.datamodules.components.vision_transform_setting import VisionTransformSetting
from src.utils.basic_utils import read_or_get_image


class TripletComicsSeqFirebaseFaceBodyDataset(Dataset):
    def __init__(self,
                 transform_face: A.Compose,
                 positive_transform_face: A.Compose,
                 negative_transform_face: A.Compose,
                 transform_body: A.Compose,
                 positive_transform_body: A.Compose,
                 negative_transform_body: A.Compose,
                 data_dir: str,
                 # path from datadir...
                 sequences_json_path: str = 'comics_seq/train_sequences.json',
                 triplet_index_csv_path_face: str = 'comics_seq/bounded_training_face_triplets.csv',
                 img_folder_root_dir_face: str = '/scratch/users/gsoykan20/projects/DASS_Det_Inference/comics_crops_large_faces/',
                 triplet_index_csv_path_body: str = 'comics_seq/bounded_training_body_triplets.csv',
                 img_folder_root_dir_body: str = '/datasets/COMICS/comics_crops/',
                 lower_idx_bound: Optional[int] = None,
                 higher_idx_bound: Optional[int] = None,
                 is_torch_transform: bool = False,
                 is_train: bool = True,
                 face_img_dim: int = 96,
                 body_img_dim: int = 128
                 ):
        self.face_img_dim = face_img_dim
        self.body_img_dim = body_img_dim
        self.lower_idx_bound = lower_idx_bound
        self.higher_idx_bound = higher_idx_bound
        self.sequences_json_path = sequences_json_path
        self.triplet_index_csv_path_face = triplet_index_csv_path_face
        self.img_folder_root_dir_face = img_folder_root_dir_face
        self.triplet_index_csv_path_body = triplet_index_csv_path_body
        self.img_folder_root_dir_body = img_folder_root_dir_body
        self.data_dir = data_dir
        self.is_torch_transform = is_torch_transform
        self.transform_face = transform_face
        self.positive_transform_face = positive_transform_face
        self.negative_transform_face = negative_transform_face
        self.transform_body = transform_body
        self.positive_transform_body = positive_transform_body
        self.negative_transform_body = negative_transform_body
        self.dataset, self.dataset_dict = self.load_dataset()
        self.sorted_dataset_keys = list(sorted(self.dataset_dict.keys()))
        self.is_train = is_train

    def _create_item_id_to_char_id_dict(self, sequences_json: Dict) -> Tuple[Dict, Dict, Dict]:
        # given face returns body id
        d_f = {}
        # given body returns face id
        d_b = {}
        d = {}
        for element in sequences_json:
            annotations = element['annotations']
            for annotation in annotations:
                char_id = annotation['charId']
                d_f[char_id] = {}
                d_b[char_id] = {}
                for char_instance in annotation['charInstances']:
                    face_instance = char_instance.get('face')
                    body_instance = char_instance.get('body')
                    if face_instance is not None:
                        d[face_instance] = char_id
                        d_f[char_id][face_instance] = body_instance
                    if body_instance is not None:
                        d[body_instance] = char_id
                        d_b[char_id][body_instance] = face_instance
        return d, d_f, d_b

    def load_dataset(self):
        with open(os.path.join(self.data_dir, self.sequences_json_path), 'r') as file:
            sequences_json = json.load(file)

        instance_to_char_id, face_to_body, body_to_face = self._create_item_id_to_char_id_dict(sequences_json)

        def construct_img_path(item_id: str,
                               is_comics_crops_with_body_face: bool,
                               img_folder_root_dir: str,
                               ):
            series_id, page_id, panel_id, img_type, image_idx = item_id.split('_')
            if is_comics_crops_with_body_face:
                return os.path.join(img_folder_root_dir, str(series_id), f'{str(page_id)}_{str(panel_id)}',
                                    'bodies' if img_type == 'body' or img_type == 'bodies' else 'faces',
                                    f'{str(image_idx)}.jpg')
            else:
                return os.path.join(img_folder_root_dir, str(series_id), f'{str(page_id)}_{str(panel_id)}',
                                    f'{str(image_idx)}.jpg')

        csv_path_face = self.triplet_index_csv_path_face
        csv_path_body = self.triplet_index_csv_path_body

        dataset = []
        triplet_records_face = pd.read_csv(os.path.join(self.data_dir, csv_path_face)).to_dict('records')
        for item_row in triplet_records_face:
            anchor_id = item_row['anchor']
            anchor_char_id = instance_to_char_id[anchor_id]
            anchor_pair_id = face_to_body[anchor_char_id].get(anchor_id)

            positive_id = item_row['positive']
            positive_char_id = instance_to_char_id[positive_id]
            positive_pair_id = face_to_body[positive_char_id].get(positive_id)

            negative_id = item_row['negative']
            negative_char_id = instance_to_char_id[negative_id]
            negative_pair_id = face_to_body[negative_char_id].get(negative_id)

            anchor_path = construct_img_path(anchor_id, False, self.img_folder_root_dir_face)
            positive_path = construct_img_path(positive_id, False, self.img_folder_root_dir_face)
            negative_path = construct_img_path(negative_id, False, self.img_folder_root_dir_face)

            anchor_pair_path = construct_img_path(anchor_pair_id, True,
                                                  self.img_folder_root_dir_body) if anchor_pair_id is not None else None
            positive_pair_path = construct_img_path(positive_pair_id, True,
                                                    self.img_folder_root_dir_body) if positive_pair_id is not None else None
            negative_pair_path = construct_img_path(negative_pair_id, True,
                                                    self.img_folder_root_dir_body) if negative_pair_id is not None else None

            anchor_pair_id = anchor_pair_id if anchor_pair_id is not None else ''
            # format: face_id _ body_id, (face_path, body_path) x N
            triplet = (
                anchor_id + '_' + anchor_pair_id, (anchor_path, anchor_pair_path), (positive_path, positive_pair_path),
                (negative_path, negative_pair_path))
            dataset.append(triplet)

        triplet_records_body = pd.read_csv(os.path.join(self.data_dir, csv_path_body)).to_dict('records')
        for item_row in triplet_records_body:
            anchor_id = item_row['anchor']
            anchor_char_id = instance_to_char_id[anchor_id]
            anchor_pair_id = body_to_face[anchor_char_id].get(anchor_id)

            positive_id = item_row['positive']
            positive_char_id = instance_to_char_id[positive_id]
            positive_pair_id = body_to_face[positive_char_id].get(positive_id)

            negative_id = item_row['negative']
            negative_char_id = instance_to_char_id[negative_id]
            negative_pair_id = body_to_face[negative_char_id].get(negative_id)

            anchor_path = construct_img_path(item_row['anchor'], True, self.img_folder_root_dir_body)
            positive_path = construct_img_path(item_row['positive'], True, self.img_folder_root_dir_body)
            negative_path = construct_img_path(item_row['negative'], True, self.img_folder_root_dir_body)

            anchor_pair_path = construct_img_path(anchor_pair_id, False,
                                                  self.img_folder_root_dir_face) if anchor_pair_id is not None else None
            positive_pair_path = construct_img_path(positive_pair_id, False,
                                                    self.img_folder_root_dir_face) if positive_pair_id is not None else None
            negative_pair_path = construct_img_path(negative_pair_id, False,
                                                    self.img_folder_root_dir_face) if negative_pair_id is not None else None

            # format: face_id _ body_id, (face_path, body_path) x N
            anchor_pair_id = anchor_pair_id if anchor_pair_id is not None else ''
            triplet = (
                anchor_pair_id + '_' + anchor_id, (anchor_pair_path, anchor_path), (positive_pair_path, positive_path),
                (negative_pair_path, negative_path))
            dataset.append(triplet)

        dataset = list({*dataset})
        if self.lower_idx_bound is not None and self.higher_idx_bound is None:
            dataset = dataset[self.lower_idx_bound:]
        elif self.higher_idx_bound is not None and self.lower_idx_bound is None:
            dataset = dataset[:self.higher_idx_bound]
        elif self.higher_idx_bound is not None and self.lower_idx_bound is not None:
            dataset = dataset[self.lower_idx_bound:self.higher_idx_bound]

        dataset_group = itertools.groupby(sorted(dataset, key=lambda x: x[0]), lambda x: x[0])
        dataset_dict = {}
        for key, group in dataset_group:
            dataset_dict[key] = list(group)
        return dataset, dataset_dict

    def __getitem__(self, index):
        if self.is_train:
            anchor_id_to_take = self.sorted_dataset_keys[index]
            triplets = self.dataset_dict[anchor_id_to_take]
            triplet = random.choice(triplets)
            _, (anchor_path_face, anchor_path_body), \
            (positive_path_face, positive_path_body), \
            (negative_path_face, negative_path_body) = triplet
        else:
            _, (anchor_path_face, anchor_path_body), \
            (positive_path_face, positive_path_body), \
            (negative_path_face, negative_path_body) = self.dataset[index]
        if self.is_torch_transform:
            if anchor_path_face is not None:
                anchor_img_face = Image.open(anchor_path_face)
                anchor_img_face = self.transform_face(anchor_img_face)
            else:
                anchor_img_face = torch.zeros(3, self.face_img_dim, self.face_img_dim)

            if positive_path_face is not None:
                positive_img_face = Image.open(positive_path_face)
                positive_img_face = self.positive_transform_face(positive_img_face)
            else:
                positive_img_face = torch.zeros(3, self.face_img_dim, self.face_img_dim)

            if negative_path_face is not None:
                negative_img_face = Image.open(negative_path_face)
                negative_img_face = self.negative_transform_face(negative_img_face)
            else:
                negative_img_face = torch.zeros(3, self.face_img_dim, self.face_img_dim)

            if anchor_path_body is not None:
                anchor_img_body = Image.open(anchor_path_body)
                anchor_img_body = self.transform_body(anchor_img_body)
            else:
                anchor_img_body = torch.zeros(3, self.body_img_dim, self.body_img_dim)

            if positive_path_body is not None:
                positive_img_body = Image.open(positive_path_body)
                positive_img_body = self.positive_transform_body(positive_img_body)
            else:
                positive_img_body = torch.zeros(3, self.body_img_dim, self.body_img_dim)

            if negative_path_body is not None:
                negative_img_body = Image.open(negative_path_body)
                negative_img_body = self.negative_transform_body(negative_img_body)
            else:
                negative_img_body = torch.zeros(3, self.body_img_dim, self.body_img_dim)
        else:
            if anchor_path_face is not None:
                anchor_img_face = read_or_get_image(anchor_path_face, read_rgb=True)
                anchor_img_face = self.transform_face(image=anchor_img_face)['image']
            else:
                anchor_img_face = torch.zeros(3, self.face_img_dim, self.face_img_dim)

            if positive_path_face is not None:
                positive_img_face = read_or_get_image(positive_path_face, read_rgb=True)
                positive_img_face = self.positive_transform_face(image=positive_img_face)['image']
            else:
                positive_img_face = torch.zeros(3, self.face_img_dim, self.face_img_dim)

            if negative_path_face is not None:
                negative_img_face = read_or_get_image(negative_path_face, read_rgb=True)
                negative_img_face = self.negative_transform_face(image=negative_img_face)['image']
            else:
                negative_img_face = torch.zeros(3, self.face_img_dim, self.face_img_dim)

            if anchor_path_body is not None:
                anchor_img_body = read_or_get_image(anchor_path_body, read_rgb=True)
                anchor_img_body = self.transform_body(image=anchor_img_body)['image']
            else:
                anchor_img_body = torch.zeros(3, self.body_img_dim, self.body_img_dim)

            if positive_path_body is not None:
                positive_img_body = read_or_get_image(positive_path_body, read_rgb=True)
                positive_img_body = self.positive_transform_body(image=positive_img_body)['image']
            else:
                positive_img_body = torch.zeros(3, self.body_img_dim, self.body_img_dim)

            if negative_path_body is not None:
                negative_img_body = read_or_get_image(negative_path_body, read_rgb=True)
                negative_img_body = self.negative_transform_body(image=negative_img_body)['image']
            else:
                negative_img_body = torch.zeros(3, self.body_img_dim, self.body_img_dim)

        return {
            'anchor_face': (anchor_img_face, anchor_path_face is not None),
            'anchor_body': (anchor_img_body, anchor_path_body is not None),
            'positive_face': (positive_img_face, positive_path_face is not None),
            'positive_body': (positive_img_body, positive_path_body is not None),
            'negative_face': (negative_img_face, negative_path_face is not None),
            'negative_body': (negative_img_body, negative_path_body is not None)
        }

    def __len__(self):
        return len(self.sorted_dataset_keys) if self.is_train else len(self.dataset)


if __name__ == '__main__':
    data_dir = '/home/gsoykan20/Desktop/self_development/amazing-mysteries-of-gutter-demystified/data'
    dataset = TripletComicsSeqFirebaseFaceBodyDataset(data_dir=data_dir,
                                                      transform=VisionTransformSetting.VANILLA_VAE_FACE.get_transformation(),
                                                      positive_transform=VisionTransformSetting.VAE_FACE_POSITIVE.get_transformation(),
                                                      negative_transform=VisionTransformSetting.VAE_FACE_NEGATIVE.get_transformation(),
                                                      sequences_json_path='comics_seq/train_sequences.json',
                                                      triplet_index_csv_path_face='comics_seq/bounded_training_face_triplets.csv',
                                                      img_folder_root_dir_face='/scratch/users/gsoykan20/projects/DASS_Det_Inference/comics_crops_large_faces/',
                                                      triplet_index_csv_path_body='comics_seq/bounded_training_body_triplets.csv',
                                                      img_folder_root_dir_body='/datasets/COMICS/comics_crops/',
                                                      is_torch_transform=False,
                                                      is_train=False)
    print(len(dataset))
