# Evaluating Interpretable Large Language Models in a Clinical Context

## Project Summary

Our project focuses on assessing and improving local interpretability method of LLMs in healthcare, specifically using Meta AI's Open Pre-trained Transformers (OPT) model as a baseline classifier, we fine-tune the OPT model on the MIMIC-IV medical dataset, focusing on free-text clinical notes from the MIMIC-IV-Note dataset. The data is cleaned and split based on ICD-9 and ICD-10 codes, creating two separate datasets. The model is then fine-tuned for each dataset to predict corresponding ICD codes and the interpretability methods are applied on the better performing model. Post-training, the, SHAP, and LIME interpretability methods are applied on the fine-tuned model. We assess these methods' efficacy based on faithfulness, using a modified version of DeYoung et al.'s evaluation procedures.

## Requirements

- [MIMIC-IV](https://physionet.org/content/mimiciv/2.2/) and [MIMIC-IV Note](https://physionet.org/content/mimic-iv-note/2.2/) datasets
- see requirements.txt

## Faithfulness data processing

- For faithfulness, there are several data processing steps that need to be taken:
  - First, using your XAI method, split up your input text strings in to the list of words used by the XAI method. The final list structure should be a list of lists, where each sub list contains all of the words in order for one sample.
  - Next, get the indices for the words used by the XAI methods explanation. These indices should be formatted in to an array as follows:
    - [
      [ Indices of corresponding text input sample ],
      [ Indices of words in text sample ]
      ]
  - Next, pass the formatted text instances and the indices to the remove_rationalle_words and remove_other_words functions. These will return the strings with related rationalle words (or all non rationalle words) removed.
  - Finally, the instances along with the returned arrays from the previous step can be passed to the faithfulness function. Note that remove_rationalle_words and remove_other_words arrays are expected to be in a larger array containing the explanations from all XAI functions

## Running explain_bert.ipynb

1. Download model weights [best_model_state.bin](https://drive.google.com/drive/folders/1a7MW1GxHa8tzSiYX_4hgAZ5jvma3Ix-L?usp=drive_link)
2. Ensure that the preprocessed outputs are correctly named. If preprocessing step unavailable, see here for preprocessed [datasets](https://drive.google.com/drive/folders/1a7MW1GxHa8tzSiYX_4hgAZ5jvma3Ix-L?usp=drive_link) necessary to run the notebook.
3. Run the notebook.
