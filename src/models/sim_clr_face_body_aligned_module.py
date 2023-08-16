from typing import Any, List, Optional

import pandas as pd
import torch
import torch.nn.functional as F
from pytorch_lightning import LightningModule
from torch import Tensor

from src.datamodules.components.vision_transform_setting import VisionTransformSetting
from src.models.components.ssl_module.info_nce_loss import InfoNCELoss
from src.models.components.ssl_module.nt_xent_loss import NTXentLoss
from src.models.components.ssl_module.sim_clr import SimCLR
from src.utils.basic_utils import read_or_get_image


class SimCLRFaceBodyAlignedLitModule(LightningModule):
    def __init__(
            self,
            # model
            model_name: str = 'resnet18',
            encoder_dim: Optional[int] = None,
            use_deeper_proj_head: Optional[bool] = False,
            normalize_projections: bool = False,
            hidden_dim: int = 128,
            # loss
            temperature: float = 0.07,  # 0.5 for nt-xent loss
            loss_fn='info_nce',  # 'info_nce' or 'nt_xent'
            disable_alignment: bool = False,
            # training
            max_epochs: int = 500,
            lr: float = 5e-4,
            weight_decay: float = 0.01,
    ):
        super().__init__()
        self.save_hyperparameters(logger=False)
        assert self.hparams.temperature > 0.0, "The temperature must be a positive float!"
        self.model = SimCLR(model_name=model_name,
                            hidden_dim=hidden_dim,
                            encoder_dim=encoder_dim,
                            use_deeper_proj_head=use_deeper_proj_head,
                            normalize=normalize_projections)
        if loss_fn == 'info_nce':
            self.criterion = InfoNCELoss(temperature)
        elif loss_fn == 'nt_xent':
            self.criterion = NTXentLoss(temperature)

    def forward(self, x_all: Tensor) -> Tensor:
        return self.model(x_all)

    def step(self, batch: Any):
        # face_transformeds, body_transformeds, [weak_face, weak_body]
        if self.hparams.disable_alignment:
            x_face_all = torch.cat([batch[0][0], batch[0][1]])
            x_body_all = torch.cat([batch[1][0], batch[1][1]])
        else:
            x_face_all = torch.cat([batch[0][0], batch[0][1], batch[2][0]])
            x_body_all = torch.cat([batch[1][0], batch[1][1], batch[2][1]])

        representation_face_all_ = self.forward(x_face_all)
        representation_body_all_ = self.forward(x_body_all)

        if self.hparams.disable_alignment:
            face_embedding_anchor, face_embedding_prime = torch.tensor_split(
                representation_face_all_, 2)
            body_embedding_anchor, body_embedding_prime = torch.tensor_split(
                representation_body_all_, 2)
        else:
            face_embedding_anchor, face_embedding_prime, face_embedding_identity = torch.tensor_split(
                representation_face_all_, 3)
            body_embedding_anchor, body_embedding_prime, body_embedding_identity = torch.tensor_split(
                representation_body_all_, 3)

        face_loss, face_logging_metrics = self.criterion(face_embedding_anchor, face_embedding_prime)
        body_loss, body_logging_metrics = self.criterion(body_embedding_anchor, body_embedding_prime)

        if self.hparams.disable_alignment:
            identity_loss, identity_logging_metrics = 0, {}
        else:
            identity_loss, identity_logging_metrics = self.criterion(face_embedding_identity, body_embedding_identity)

        loss = face_loss + body_loss + identity_loss
        return loss, \
            face_loss, body_loss, identity_loss, \
            face_logging_metrics, body_logging_metrics, identity_logging_metrics

    def training_step(self, batch: Any, batch_idx: int):
        loss, \
            face_loss, body_loss, identity_loss, \
            face_logging_metrics, body_logging_metrics, identity_logging_metrics = self.step(
            batch)

        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/face_loss", face_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/body_loss", body_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/id_loss", identity_loss, on_step=True, on_epoch=True, prog_bar=True)

        for k, v in face_logging_metrics.items():
            self.log(f"train/face_{k}", v, on_step=True, on_epoch=True, prog_bar=True)
        for k, v in body_logging_metrics.items():
            self.log(f"train/body_{k}", v, on_step=True, on_epoch=True, prog_bar=True)
        for k, v in identity_logging_metrics.items():
            self.log(f"train/id_{k}", v, on_step=True, on_epoch=True, prog_bar=True)

        return {"loss": loss}

    def training_epoch_end(self, outputs: List[Any]):
        pass

    def validation_step(self, batch: Any, batch_idx: int):
        loss, \
            face_loss, body_loss, identity_loss, \
            face_logging_metrics, body_logging_metrics, identity_logging_metrics = self.step(
            batch)

        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log("val/face_loss", face_loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log("val/body_loss", body_loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log("val/id_loss", identity_loss, on_step=False, on_epoch=True, prog_bar=False)

        for k, v in face_logging_metrics.items():
            self.log(f"val/face_{k}", v, on_step=False, on_epoch=True, prog_bar=False)
        for k, v in body_logging_metrics.items():
            self.log(f"val/body_{k}", v, on_step=False, on_epoch=True, prog_bar=False)
        for k, v in identity_logging_metrics.items():
            self.log(f"val/id_{k}", v, on_step=False, on_epoch=True, prog_bar=False)

        return {"loss": loss}

    def validation_epoch_end(self, outputs: List[Any]):
        pass

    def test_step(self, batch: Any, batch_idx: int):
        loss, \
            face_loss, body_loss, identity_loss, \
            face_logging_metrics, body_logging_metrics, identity_logging_metrics = self.step(
            batch)

        self.log("test/loss", loss, on_step=False, on_epoch=True, )
        self.log("test/face_loss", face_loss, on_step=False, on_epoch=True, )
        self.log("test/body_loss", body_loss, on_step=False, on_epoch=True, )
        self.log("test/id_loss", identity_loss, on_step=False, on_epoch=True, )

        for k, v in face_logging_metrics.items():
            self.log(f"test/face_{k}", v, on_step=False, on_epoch=True, )
        for k, v in body_logging_metrics.items():
            self.log(f"test/body_{k}", v, on_step=False, on_epoch=True, )
        for k, v in identity_logging_metrics.items():
            self.log(f"test/id_{k}", v, on_step=False, on_epoch=True, )

        return {"loss": loss}

    def test_epoch_end(self, outputs: List[Any]):
        pass

    def on_epoch_end(self):
        # make a system that trains a logreg model
        pass

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.hparams.max_epochs, eta_min=self.hparams.lr / 50
        )
        return [optimizer], [lr_scheduler]

    @classmethod
    def checkpoint_to_eval(cls,
                           checkpoint_path: str,
                           **kwargs, ):
        trained_model = cls.load_from_checkpoint(checkpoint_path, strict=False, **kwargs).to('cuda')
        trained_model.eval()
        trained_model.freeze()
        return trained_model

    @staticmethod
    def get_embeddings_from_imgs(trained_model: LightningModule, img_paths: List[str]) -> torch.Tensor:
        transform = VisionTransformSetting.CORINFOMAX_EVAL_TEST.get_transformation()
        batch = []
        for img_path in img_paths:
            source_img = read_or_get_image(img_path, read_rgb=True)
            source_img = transform(image=source_img)['image'].to(trained_model.device)
            batch.append(source_img)
        batch = torch.stack(batch)
        with torch.no_grad():
            embeddings = trained_model.model(batch)
        return embeddings


######################################

def check_embedding_generation():
    ckpt_path = '/home/gsoykan20/Desktop/self_development/char_reid_repo/data/checkpoints/face_body_aligned_ssl.ckpt'
    trained_model: SimCLRFaceBodyAlignedLitModule = SimCLRFaceBodyAlignedLitModule.checkpoint_to_eval(ckpt_path)
    img_paths = [
        '/scratch/users/gsoykan20/projects/DASS_Det_Inference/comics_crops/1008/10_5/bodies/1.jpg',
        '/scratch/users/gsoykan20/projects/DASS_Det_Inference/comics_crops/100/12_3/bodies/1.jpg',
        '/scratch/users/gsoykan20/projects/DASS_Det_Inference/comics_crops/1008/10_4/bodies/0.jpg']
    embeddings = SimCLRFaceBodyAlignedLitModule.get_embeddings_from_imgs(trained_model, img_paths).detach().cpu()
    return trained_model, embeddings


if __name__ == '__main__':
    loaded_module, embeddings = check_embedding_generation()
    eigs = loaded_module.cov_criterion.save_eigs()
    dists = [[torch.linalg.norm(e1 - e2).item() for e2 in embeddings] for e1 in embeddings]
    print(pd.DataFrame(dists, columns=['same 1', 'diff', 'same 2', ], index=['same 1', 'diff', 'same 2', ]))
    cos_sims = [[F.cosine_similarity(e1.unsqueeze(0), e2.unsqueeze(0))[0].item() for e2 in embeddings] for e1 in
                embeddings]
    print(pd.DataFrame(cos_sims, columns=['same 1', 'diff', 'same 2', ], index=['same 1', 'diff', 'same 2', ]))
