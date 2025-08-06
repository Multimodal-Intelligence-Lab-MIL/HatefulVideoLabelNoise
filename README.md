# Revealing Temporal Label Noise in Multimodal Hateful Video Classification

> Official repository for the paper:  
> **"Revealing Temporal Label Noise in Multimodal Hateful Video Classification"**  
> Accepted at *The 4th International Workshop on Multimodal Human Understanding for the Web and Social Media (MMUW '25)*, co-located with ACM Multimedia 2025.

---

## Abstract

The rapid proliferation of online multimedia content has intensified the spread of hate speech, presenting critical societal and regulatory challenges. While recent work has advanced multimodal hateful video detection, most approaches rely on coarse, video-level annotations that overlook the temporal granularity of hateful content. This introduces substantial label noise, as videos annotated as hateful often contain long non-hateful segments. In this paper, we investigate the impact of such label ambiguity through a fine-grained approach. Specifically, we trim hateful videos from the HateMM and MultiHateClip English datasets using annotated timestamps to isolate explicitly hateful segments. We then conduct an exploratory analysis of these trimmed segments to examine the distribution and characteristics of both hateful and non-hateful content. This analysis highlights the degree of semantic overlap and the confusion introduced by coarse, video-level annotations. Finally, controlled experiments demonstrate that time-stamp noise fundamentally alters model decision boundaries and weakens classification confidence, highlighting the inherent context dependency and temporal continuity of hate speech expression. Our findings emphasize the need for temporally aware models and benchmarks for improved robustness and interpretability.


## Overview
<img width="1480" height="666" alt="88381ca1ab7d947c6220479e476f33ce" src="https://github.com/user-attachments/assets/3cd4a01f-7a1e-4f17-9428-07347f17fb2b" />

---

## Dataset
Due to copyright restrictions, the raw datasets are not included. 
You can obtain the datasets from their respective original project sites:

### HateMM

Access the full dataset from [hate-alert/HateMM](https://github.com/hate-alert/HateMM).

### MultiHateClip

Access the full dataset from [Social-AI-Studio/MultiHateClip](https://github.com/Social-AI-Studio/MultiHateClip):  

## Data Preprocess
1. For hateful video trimming, can refer to the annotations provided by HateMM and MultiHateClip, and trim the hateful videos according to their time annotations to obtain trimmed hate segments.
2. For full video feature extraction methods, refer to the feature extraction methods published by hatemm.
3. We have provided examples of feature extraction for trimmed videos in the code folder (Feature extraction).

## Code Structure

The code is organized in a modular and readable structure for multimodal hateful video classification with temporal label noise analysis.

Code_Label_Noise/
├── configs/ # Configuration files
│ └── multimodal_config.py # Main config
│
├── Feature_extraction/ # Feature extraction from raw videos
│ ├── bert_hatexplain_pure_hate_seg.py # BERT-based textual features
│ ├── MFCC_pure_hate_seg.py # Audio features using MFCCs
│ └── vit_pure_hate_seg.py # Visual features using ViT
│
├── scripts/
│ └── train_multimodal.py # Training entry point
│
├── src/
│ ├── data/
│ │ └── dataset_multimodal.py # Dataset loading and preprocessing
│ │
│ ├── evaluation/
│ │ └── metrics_multimodal.py # Evaluation metrics (e.g., F1, accuracy)
│ │
│ ├── modules/
│ │ └── multimodal_models.py # Model definitions (e.g., fusion models)
│ │
│ ├── training/
│ │ └── trainer_multimodal.py # Training loop and optimization
│ │
│ └── utils/
│ └── seed.py # Seeding for reproducibility
