<div align="center">

# Identity-Aware Semi-Supervised Learning for Comic Character Re-Identification

<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
<a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning-792ee5?logo=pytorchlightning&logoColor=white"></a>
<a href="https://hydra.cc/"><img alt="Config: Hydra" src="https://img.shields.io/badge/Config-Hydra-89b8cd"></a>
<a href="https://github.com/ashleve/lightning-hydra-template"><img alt="Template" src="https://img.shields.io/badge/-Lightning--Hydra--Template-017F2F?style=flat&logo=github&labelColor=gray"></a><br>
[![Paper](http://img.shields.io/badge/paper-arxiv.1001.2234-B31B1B.svg)](https://www.nature.com/articles/nature14539)
[![Conference](http://img.shields.io/badge/AnyConference-year-4b44ce.svg)](https://papers.nips.cc/paper/2020)

</div>

## Description

This study aims to produce "Comic Character Embeddings" that are unified and identity-aligned.
"Unified" in the sense that it uses face features and body features in tandem.
"Identity-Aligned" because during self-supervision stage we use "Identity-Awareness Loss" which
is a contrastive loss term (NT-Xent) forces model to maximize similarity face features of a comic character and
body features.

## Datasets
- **Comic Character Instance Dataset**: It contains character instances from the golden age of comics.
As character instances we mean face and body bounding boxes of the character. We share 5 files for this dataset.
Find them under 'data/ssl'
  - filtered_all_body.csv
  - filtered_all_body_100k.csv
  - filtered_all_face.csv
  - filtered_all_face_100k.csv
  - merged_face_body.csv -> by using this you can see pairs of face and body for character instances.
- **Comic Sequence Identity Dataset**: Contains the relative identity annotations from comic sequences. 
Specifically, from 4 consecutive panels. You can find the related annotations under 'data/comics_seq'. Mainly, you'd 
need to use followings: 
  - <train-validation-test>_char_faces.json: has type Dict[str, List[str]], character id's are the keys, values are the face instances 
    they are formatted as follows => "<series_id>_<page_id>_<panel_id>_<face_or_body>_<id>" e.g. "1443_58_6_face_0". You can match this instances with the 
  Comic Character Instance Dataset.
  - <train-validation-test>_char_bodies.json: has the same format as faces.json 
  - <train-validation-test>_sequences.json: in sequences files, you can find annotations for each character identity consisting of their character instances, each
  instance may contain, a face, a body, and multiple speech bubble associations. Although, using a character-speech bubble associations is not part of this study,
  they are still valuable resource. Additionally, this is human annotated data, it is collected via a web-interface called, "Comic Sequence Identity Annotator".
  During the data collection phase some characters are skipped if their bounding boxes are not aligned well with the characters. 
Apart from these files, you can find some additional csv or json files, such as random tuples, triplets from the data.
## Data Dependencies

- you need to download the "COMICS" dataset. https://github.com/miyyer/comics to crop faces 
and bodies of character. Currently, we only share bounding boxes of those with "Comic Character Instances Dataset".
- you should crop panels of the comics by bounding boxes presented in Comic Character Instances Dataset.
  - after cropping you should save bodies it like this => "./data/comics_crops/<series>/<page_id>_<panel_id>>/bodies/<id>.jpg"
  - for faces =>  "./data/comics_crops_large_faces/<series>/<page_id>_<panel_id>>/<id>.jpg"
    - remember face bounding boxes undergo post-processing before they are, they are scaled (x1.2) and should be squared.
    
## Inference 

- You should use 'PMLIdNetFineTunedSSLBackboneFaceBodyLitModule' for main reidentification model
  - check the arguments of the 'forward' method
- For Identity Assignment demo, use 'CharacterIdentityAssigner' in 'character_identity_assigner.py'
  - you should update model_checkpoint, data_dir, face_root, body_root variables
  - identity assigner currently uses "ComicsSeqFirebaseFaceBodySequenceIteratorDataset" which 
  iterates on <train-validation-test>_sequences.json's.
  - You can find necessary transformations in that.
  - Eventually, it gives you identity cluster for a given sequence of panels 
    - you can also visualize the results, check Appendix A for how it is shown.

## Checkpoints

- Identity-Aware Self-Supervised Model
- Fine-Tuned Re-Identification Network

## How to run

Install dependencies

```bash
# clone project
git clone https://github.com/gsoykan/comic_char_reid
cd comic_char_reid

# [OPTIONAL] create conda environment
conda create -n myenv python=3.8
conda activate myenv

# install pytorch according to instructions
# https://pytorch.org/get-started/

# install requirements
pip install -r requirements.txt
```

You can train models with the following configurations:
<ul>
<li>for Identity-Aware Self-Supervision stage, you should use "sim_clr_comics_crops_face_body_aligned_infonce_multi"</li>
<li>for semi-supervision (fine-tuning with identity labels on sequences of 4 panels), you should use "pml_id_net_fine_tuned_sim_clr_backbone_face_body"</li>
</ul>

Note that, this configurations are the ones, that is used with the final results of the paper.
If you'd like to change hyper-parameters please feel free to do so.

Train model with chosen experiment configuration from [configs/experiment/](configs/experiment/)

```bash
python train.py experiment=experiment_name.yaml
```

You can override any parameter from command line like this

```bash
python train.py trainer.max_epochs=20 datamodule.batch_size=64

# train on CPU
python train.py trainer.gpus=0

# train on GPU
python train.py trainer.gpus=1
```
