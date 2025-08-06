# Revealing Temporal Label Noise in Multimodal Hateful Video Classification

> Official repository for the paper:  
> **"Revealing Temporal Label Noise in Multimodal Hateful Video Classification"**  
> Accepted at *The 4th International Workshop on Multimodal Human Understanding for the Web and Social Media (MMUW '25)*, co-located with ACM Multimedia 2025.

---

## Abstract

The rapid proliferation of online multimedia content has intensified the spread of hate speech, presenting critical societal and regulatory challenges. While recent work has advanced multimodal hateful video detection, most approaches rely on coarse, video-level annotations that overlook the temporal granularity of hateful content. This introduces substantial label noise, as videos annotated as hateful often contain long non-hateful segments. In this paper, we investigate the impact of such label ambiguity through a fine-grained approach. Specifically, we trim hateful videos from the HateMM and MultiHateClip English datasets using annotated timestamps to isolate explicitly hateful segments. We then conduct an exploratory analysis of these trimmed segments to examine the distribution and characteristics of both hateful and non-hateful content. This analysis highlights the degree of semantic overlap and the confusion introduced by coarse, video-level annotations. Finally, controlled experiments demonstrate that time-stamp noise fundamentally alters model decision boundaries and weakens classification confidence, highlighting the inherent context dependency and temporal continuity of hate speech expression. Our findings emphasize the need for temporally aware models and benchmarks for improved robustness and interpretability.


## 🖼️ Overview

*Figure: Overview of the temporal trimming and analysis pipeline.*

---
## 🗂️ Code Structure

```bash
├── data_splits/              # Temporal and 5-fold splits (video IDs)
├── analysis/                 # Distributional analysis of segments
├── models/                   # Baseline models & training scripts
├── utils/                    # Helper functions
├── figures/                  # Paper figures and visuals
└── README.md
