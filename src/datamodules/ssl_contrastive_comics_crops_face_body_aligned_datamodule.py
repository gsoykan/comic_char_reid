from typing import Optional, Tuple

import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset

from src.datamodules.components.face_recognition.ssl_contrastive_comics_crops_face_body_aligned_dataset import \
    SSLContrastiveComicsCropsFaceBodyAlignedDataset
from src.datamodules.components.vision_transform_setting import VisionTransformSetting, ContrastiveTransformations
from src.utils.pytorch_utils import visualize_tensor, UnNormalize


class SSLContrastiveComicsCropsFaceBodyAlignedDataModule(LightningDataModule):
    def __init__(
            self,
            # dataset args
            face_prefiltered_csv_folder_dir: str,
            body_prefiltered_csv_folder_dir: str,
            face_body_csv_path: str,
            data_dir: str = "data/",
            face_folder_dir: Optional[str] = None,
            body_folder_dir: Optional[str] = None,
            face_img_dim: int = 96,
            body_img_dim: int = 128,
            limit_search_files: Optional[int] = None,
            disable_alignment: bool = False,
            # val - test dataset args
            train_val_test_split: Tuple[float, float, float] = (0.92, 0.04, 0.04),
            # dataloading args
            batch_size: int = 64,
            num_workers: int = 0,
            pin_memory: bool = False,
    ):
        super().__init__()
        self.save_hyperparameters(logger=False)
        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

    @property
    def num_classes(self) -> int:
        return 2

    def prepare_data(self):
        pass

    def setup(self, stage: Optional[str] = None):
        if not self.data_train and not self.data_val and not self.data_test:
            face_transform_args = {'N': self.hparams.face_img_dim, 'min_scale': 0.08, 'use_padding': False}
            face_weak_transform_args = {'N': self.hparams.face_img_dim, 'min_scale': 0.2, 'use_padding': False}
            body_transform_args = {'N': self.hparams.body_img_dim, 'min_scale': 0.08}
            body_weak_transform_args = {'N': self.hparams.body_img_dim, 'min_scale': 0.5}
            face_train_transform = ContrastiveTransformations(
                VisionTransformSetting.SIMCLR_PRETRAIN_TORCH.get_transformation(**face_transform_args))
            body_train_transform = ContrastiveTransformations(
                VisionTransformSetting.SIMCLR_PRETRAIN_TORCH.get_transformation(**body_transform_args))
            face_weak_transform = VisionTransformSetting.SIMCLR_FINE_TUNING.get_transformation(
                **face_weak_transform_args)
            body_weak_transform = VisionTransformSetting.SIMCLR_FINE_TUNING.get_transformation(
                **body_weak_transform_args)

            dataset = SSLContrastiveComicsCropsFaceBodyAlignedDataset(
                data_dir=self.hparams.data_dir,
                face_prefiltered_csv_folder_dir=self.hparams.face_prefiltered_csv_folder_dir,
                face_contrastive_transform=face_train_transform,
                face_weak_transform=face_weak_transform,
                body_prefiltered_csv_folder_dir=self.hparams.body_prefiltered_csv_folder_dir,
                body_contrastive_transform=body_train_transform,
                body_weak_transform=body_weak_transform,
                face_folder_dir=self.hparams.face_folder_dir,
                body_folder_dir=self.hparams.body_folder_dir,
                face_body_csv_path=self.hparams.face_body_csv_path,
                limit_search_files=self.hparams.limit_search_files,
                disable_alignment=self.hparams.disable_alignment,
                is_torch_transform=True,
                is_torch_transform_weak=False
            )
            train_size = int(len(dataset) * self.hparams.train_val_test_split[0])
            val_size = int(len(dataset) * self.hparams.train_val_test_split[1])
            test_size = len(dataset) - train_size - val_size
            self.data_train, self.data_val, self.data_test = torch.utils.data.random_split(dataset,
                                                                                           [train_size,
                                                                                            val_size,
                                                                                            test_size])

    def train_dataloader(self):
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )


if __name__ == '__main__':
    data_dir = '/scratch/users/gsoykan20/projects/amazing-mysteries-of-gutter-demystified/data'
    # data_dir = '/home/gsoykan20/Desktop/self_development/amazing-mysteries-of-gutter-demystified/data/'
    face_prefiltered_csv_folder_dir = f'{data_dir}/ssl/filtered_all_face_10k.csv'
    body_prefiltered_csv_folder_dir = f'{data_dir}/ssl/filtered_all_body_10k.csv'
    datamodule = SSLContrastiveComicsCropsFaceBodyAlignedDataModule(
        data_dir=data_dir,
        face_prefiltered_csv_folder_dir=face_prefiltered_csv_folder_dir,
        body_prefiltered_csv_folder_dir=body_prefiltered_csv_folder_dir,
        face_folder_dir='/scratch/users/gsoykan20/projects/DASS_Det_Inference/comics_crops_large_faces/',
        body_folder_dir='/datasets/COMICS/comics_crops',
        face_body_csv_path='ssl/merged_face_body.csv',
        batch_size=4,
        limit_search_files=100,
    )
    datamodule.prepare_data()
    datamodule.setup()
    train_dataloader = iter(datamodule.train_dataloader())
    test_dataloader = iter(datamodule.test_dataloader())
    batch = next(train_dataloader)
    batch_test = next(test_dataloader)
    breakpoint()
    unorm = UnNormalize(mean=[0.5, 0.5, 0.5],
                        std=[0.5, 0.5, 0.5], )
    visualize_tensor(batch[0][0], unorm)
    visualize_tensor(batch_test[0][0], unorm)
    print(len(batch))
