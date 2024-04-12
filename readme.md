# Interpretable Large Language Models in a Clinical Context

## Project Summary
Our project focuses on assessing and improving local interpretability method of LLMs in healthcare, specifically using Meta AI's Open Pre-trained Transformers (OPT) model as a baseline classifer, we fine-tune the OPT model on the MIMIC-IV medical dataset, focusing on free-text clinical notes from the MIMIC-IV-Note dataset. The data is cleaned and split based on ICD-9 and ICD-10 codes, creating two separate datasets. The model is then fine-tuned for each dataset to predict corresponding ICD codes and the interpretability methods are applied on the better performing model. Post-training, the HotFlip, Integrated Gradients, SHAP, and LIME interpretability methods are applied on the fine-tuned model. We assess these methods' efficacy based on faithfulness, using a modified version of DeYoung et al.'s evaluation procedures.

## Requirements
- [MIMIC-IV](https://physionet.org/content/mimiciv/2.2/) and [MIMIC-IV Note](https://physionet.org/content/mimic-iv-note/2.2/) datasets
- see requirements.txt


## TODOs
<ul>
  <li> Trained Classifier
  <li> XAI Implementations
  <li> Report
</ul>