from functools import partial
from typing import Optional

import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset

from src.datamodules.components.face_recognition.comics_seq_firebase_face_body_dataset import \
    ComicsSeqFirebaseFaceBodyDataset, comics_seq_firebase_face_body_collate_fn
from src.datamodules.components.face_recognition.triplet_comics_seq_firebase_face_body_dataset import \
    TripletComicsSeqFirebaseFaceBodyDataset
from src.datamodules.components.vision_transform_setting import VisionTransformSetting
from src.models.components.ssl_module.ssl_backbone import SSLBackbone
from src.utils.pickle_helper import PickleHelper
from src.utils.pml_accuracy_wrapper import PMLAccuracyWrapper
from src.utils.pml_seq_miner_wrapper import PMLSeqMinerWrapper


class ComicsSeqFirebaseFaceBodyDataModule(LightningDataModule):
    def __init__(
            self,
            data_dir: str = "data/",
            train_char_faces_json_path: str = 'comics_seq/train_char_faces.json',
            train_char_bodies_json_path: str = 'comics_seq/train_char_bodies.json',
            train_sequences_json_path: str = 'comics_seq/train_sequences.json',
            train_triplet_index_csv_path_face: str = 'comics_seq/bounded_training_face_triplets.csv',
            train_triplet_index_csv_path_body: str = 'comics_seq/bounded_training_body_triplets.csv',
            val_char_faces_json_path: str = 'comics_seq/val_char_faces.json',
            val_char_bodies_json_path: str = 'comics_seq/val_char_bodies.json',
            val_sequences_json_path: str = 'comics_seq/validation_sequences.json',
            val_triplet_index_csv_path_face: str = 'comics_seq/validation_face_triplets.csv',
            val_triplet_index_csv_path_body: str = 'comics_seq/validation_body_triplets.csv',
            test_char_faces_json_path: str = 'comics_seq/test_char_faces.json',
            test_char_bodies_json_path: str = 'comics_seq/test_char_bodies.json',
            test_sequences_json_path: str = 'comics_seq/test_sequences.json',
            test_triplet_index_csv_path_face: str = 'comics_seq/testing_face_triplets.csv',
            test_triplet_index_csv_path_body: str = 'comics_seq/testing_body_triplets.csv',
            img_folder_root_dir_face: str = '/scratch/users/gsoykan20/projects/DASS_Det_Inference/comics_crops_large_faces/',
            img_folder_root_dir_body: str = '/datasets/COMICS/comics_crops/',
            randomly_mask_face_or_body: Optional[float] = 0,
            intense_transform: bool = False,
            no_transform: bool = False,
            use_singular_chars: bool = False,
            img_dim: int = 96,
            batch_size: int = 64,
            num_workers: int = 0,
            pin_memory: bool = False,
            ssl_backbone: Optional[SSLBackbone] = None,
            use_triplets_dataset_for_val_test: bool = True,
    ):
        super().__init__()
        if isinstance(ssl_backbone, str):
            self.ssl_backbone = SSLBackbone(ssl_backbone)
        else:
            self.ssl_backbone = ssl_backbone
        self.save_hyperparameters(logger=False)
        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

    @property
    def num_classes(self) -> int:
        return 3

    def prepare_data(self):
        pass

    def setup(self, stage: Optional[str] = None):
        if not self.data_train and not self.data_val and not self.data_test:
            transform_args_face = {'N': 96, 'min_scale': 0.08, 'use_padding': False}
            transform_args_body = {'N': self.hparams.img_dim, 'min_scale': 0.5}
            if self.ssl_backbone is not None:
                if self.ssl_backbone is SSLBackbone.CORINFOMAX:
                    # TODO: @gsoykan - update this...
                    train_transform_face = VisionTransformSetting.CORINFOMAX_FINE_TUNING.get_transformation(
                        **transform_args_face)
                    transform_face = VisionTransformSetting.CORINFOMAX_EVAL_TEST.get_transformation(
                        **transform_args_face)
                    train_transform_body = VisionTransformSetting.CORINFOMAX_FINE_TUNING.get_transformation(
                        **transform_args_body)
                    transform_body = VisionTransformSetting.CORINFOMAX_EVAL_TEST.get_transformation(
                        **transform_args_body)
                elif self.ssl_backbone is SSLBackbone.SIM_CLR:
                    if not self.hparams.no_transform:
                        train_transform_face = VisionTransformSetting.SIMCLR_FINE_TUNING_INTENSE.get_transformation(
                            **transform_args_face) if self.hparams.intense_transform else VisionTransformSetting.SIMCLR_FINE_TUNING.get_transformation(
                            **transform_args_face)
                    else:
                        train_transform_face = VisionTransformSetting.SIMCLR_TEST.get_transformation(
                            **transform_args_face)
                    transform_face = VisionTransformSetting.SIMCLR_TEST.get_transformation(**transform_args_face)

                    if not self.hparams.no_transform:
                        train_transform_body = VisionTransformSetting.SIMCLR_FINE_TUNING_INTENSE.get_transformation(
                            **transform_args_body) if self.hparams.intense_transform else VisionTransformSetting.SIMCLR_FINE_TUNING.get_transformation(
                            **transform_args_body)
                    else:
                        train_transform_body = VisionTransformSetting.SIMCLR_TEST.get_transformation(
                            **transform_args_body)
                    transform_body = VisionTransformSetting.SIMCLR_TEST.get_transformation(**transform_args_body)
                else:
                    raise Exception(f'unhandled backbone for triplet id net fine-tuning: {self.ssl_backbone}')

                self.data_train = ComicsSeqFirebaseFaceBodyDataset(
                    char_faces_json_path=self.hparams.train_char_faces_json_path,
                    char_bodies_json_path=self.hparams.train_char_bodies_json_path,
                    sequences_json_path=self.hparams.train_sequences_json_path,
                    transform_face=train_transform_face,
                    transform_body=train_transform_body,
                    is_train=True,
                    img_folder_root_dir_face=self.hparams.img_folder_root_dir_face,
                    img_folder_root_dir_body=self.hparams.img_folder_root_dir_body,
                    data_dir=self.hparams.data_dir,
                    is_torch_transform=False,
                    randomly_mask_face_or_body=self.hparams.randomly_mask_face_or_body,
                    use_singular_chars=self.hparams.use_singular_chars
                )
                print('train dataset size => ', len(self.data_train))
                if self.hparams.use_triplets_dataset_for_val_test:
                    dataset = partial(TripletComicsSeqFirebaseFaceBodyDataset,
                                      img_folder_root_dir_face=self.hparams.img_folder_root_dir_face,
                                      img_folder_root_dir_body=self.hparams.img_folder_root_dir_body,
                                      data_dir=self.hparams.data_dir,
                                      is_torch_transform=False,
                                      )
                    self.data_val = dataset(
                        sequences_json_path=self.hparams.val_sequences_json_path,
                        triplet_index_csv_path_face=self.hparams.val_triplet_index_csv_path_face,
                        triplet_index_csv_path_body=self.hparams.val_triplet_index_csv_path_body,
                        transform_face=transform_face,
                        positive_transform_face=transform_face,
                        negative_transform_face=transform_face,
                        transform_body=transform_body,
                        positive_transform_body=transform_body,
                        negative_transform_body=transform_body,
                        is_train=False,
                    )
                    self.data_test = dataset(
                        sequences_json_path=self.hparams.test_sequences_json_path,
                        triplet_index_csv_path_face=self.hparams.test_triplet_index_csv_path_face,
                        triplet_index_csv_path_body=self.hparams.test_triplet_index_csv_path_body,
                        transform_face=transform_face,
                        positive_transform_face=transform_face,
                        negative_transform_face=transform_face,
                        transform_body=transform_body,
                        positive_transform_body=transform_body,
                        negative_transform_body=transform_body,
                        is_train=False,
                    )
                else:
                    self.data_val = ComicsSeqFirebaseFaceBodyDataset(
                        char_faces_json_path=self.hparams.val_char_faces_json_path,
                        char_bodies_json_path=self.hparams.val_char_bodies_json_path,
                        sequences_json_path=self.hparams.val_sequences_json_path,
                        transform_face=transform_face,
                        transform_body=transform_body,
                        is_train=False,
                        img_folder_root_dir_face=self.hparams.img_folder_root_dir_face,
                        img_folder_root_dir_body=self.hparams.img_folder_root_dir_body,
                        data_dir=self.hparams.data_dir,
                        is_torch_transform=False,
                        randomly_mask_face_or_body=0,
                        use_singular_chars=False
                    )
                    self.data_test = ComicsSeqFirebaseFaceBodyDataset(
                        char_faces_json_path=self.hparams.test_char_faces_json_path,
                        char_bodies_json_path=self.hparams.test_char_bodies_json_path,
                        sequences_json_path=self.hparams.test_sequences_json_path,
                        transform_face=transform_face,
                        transform_body=transform_body,
                        is_train=False,
                        img_folder_root_dir_face=self.hparams.img_folder_root_dir_face,
                        img_folder_root_dir_body=self.hparams.img_folder_root_dir_body,
                        data_dir=self.hparams.data_dir,
                        is_torch_transform=False,
                        randomly_mask_face_or_body=0,
                        use_singular_chars=False
                    )
            else:
                raise NotImplementedError()

    def train_dataloader(self):
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
            collate_fn=lambda batch: comics_seq_firebase_face_body_collate_fn(batch,
                                                                              self.data_train.char_id_to_certain_other_char_ids)
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.data_val,
            batch_size=256 if self.hparams.use_triplets_dataset_for_val_test else self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
            collate_fn=None if self.hparams.use_triplets_dataset_for_val_test else lambda
                batch: comics_seq_firebase_face_body_collate_fn(batch,
                                                                self.data_val.char_id_to_certain_other_char_ids)
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.data_test,
            batch_size=256 if self.hparams.use_triplets_dataset_for_val_test else self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
            collate_fn=None if self.hparams.use_triplets_dataset_for_val_test else lambda
                batch: comics_seq_firebase_face_body_collate_fn(batch,
                                                                self.data_test.char_id_to_certain_other_char_ids)
        )


from pytorch_metric_learning import miners as pml_miners

if __name__ == '__main__':
    # data_dir = '/home/gsoykan20/Desktop/self_development/amazing-mysteries-of-gutter-demystified/data/'
    data_dir = "/scratch/users/gsoykan20/projects/amazing-mysteries-of-gutter-demystified/data/"
    datamodule = ComicsSeqFirebaseFaceBodyDataModule(data_dir,
                                                     batch_size=4,
                                                     ssl_backbone=SSLBackbone.SIM_CLR,
                                                     use_triplets_dataset_for_val_test=True,
                                                     randomly_mask_face_or_body=0.66,
                                                     use_singular_chars=True,
                                                     img_dim=128)
    datamodule.prepare_data()
    datamodule.setup()
    dataloader = iter(datamodule.train_dataloader())
    batch = next(dataloader)
    breakpoint()
    accuracy_wrapper = PMLAccuracyWrapper()
    accuracy_wrapper(embeddings=torch.randn(len(batch['char_ids']), 64),
                     char_ids=batch['char_ids'],
                     seq_ids=batch['seq_ids'])
    acc_res = accuracy_wrapper.compute()
    miner_wrapper = PMLSeqMinerWrapper(pml_miners.MultiSimilarityMiner(epsilon=0.1))
    PickleHelper.save_object(PickleHelper.comics_seq_firebase_face_body_batch, batch)
    res = miner_wrapper(embeddings=torch.randn(len(batch['char_ids']), 64),
                        char_ids=batch['char_ids'],
                        seq_ids=batch['seq_ids'])
    loaded_batch = PickleHelper.load_object(PickleHelper.comics_seq_firebase_face_body_batch)
    print(dataloader)
