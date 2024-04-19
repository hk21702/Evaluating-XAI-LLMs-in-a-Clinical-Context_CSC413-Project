import pandas as pd
import pickle
from transformers import AutoTokenizer, AutoModel, AutoConfig
import torch
from datasets import load_dataset
from lime.lime_text import LimeTextExplainer
import torch.nn.functional as F
import shap
from transformers import Pipeline
import os 
import numpy

torch.cuda.empty_cache()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = "cpu"
data_dir = "output/"
destination_dir = "./"
print(device)


test_short_path = "data/test_10_top50_short.csv"
labels_10_top50 = pd.read_csv('data/icd10_codes_top50.csv')
code_labels_10 = pd.read_csv("data/icd10_codes.csv")

# Create class dictionaries
classes = [class_ for class_ in code_labels_10["icd_code"] if class_]
class2id = {class_: id for id, class_ in enumerate(classes)}
id2class = {id: class_ for class_, id in class2id.items()}

# Model Parameters
MAX_LEN = 512
MODEL = "emilyalsentzer/Bio_ClinicalBERT"
CKPT = os.path.join(data_dir,"best_model_state.bin")

config, unused_kwargs = AutoConfig.from_pretrained(
    MODEL,
    num_labels=len(classes),
    id2label=id2class,
    label2id=class2id,
    problem_type="multi_label_classification",
    return_unused_kwargs=True,
)

tokenizer_bert = AutoTokenizer.from_pretrained(MODEL)
model_bert = AutoModel.from_pretrained(MODEL, config=config, cache_dir='./model_ckpt/')

class TokenizerWrapper:
    def __init__(self, tokenizer, length, classes):
        self.tokenizer = tokenizer
        self.max_length = length
        self.classes = classes
        self.class2id = {class_: id for id, class_ in enumerate(self.classes)}
        self.id2class = {id: class_ for class_, id in self.class2id.items()}
        
    def multi_labels_to_ids(self, labels: list[str]) -> list[float]:
        ids = [0.0] * len(self.class2id)  # BCELoss requires float as target type
        for label in labels:
            ids[self.class2id[label]] = 1.0
        return ids
    
    def tokenize_function(self, example):
        result = self.tokenizer(
            example["text"],
            max_length = self.max_length,
            padding = 'max_length',
            truncation = True,
            return_tensors='pt'
        )
        result["label"] = torch.tensor([self.multi_labels_to_ids(eval(label)) for label in example["label"]])
        return result
        
data_files = {
        "test": test_short_path,
    }

tokenizer_wrapper = TokenizerWrapper(tokenizer_bert, MAX_LEN, classes)
dataset = load_dataset("csv", data_files=data_files)
dataset = dataset.map(tokenizer_wrapper.tokenize_function, batched=True, num_proc=1)
dataset = dataset.with_format("torch")

class BERTClass(torch.nn.Module):
    def __init__(self):
        super(BERTClass, self).__init__()
        self.config = config
        self.device = device
        self.bert_model = model_bert
        self.can_generate = model_bert.can_generate
        self.base_model_prefix = model_bert.base_model_prefix
        self.get_input_embeddings = model_bert.get_input_embeddings
        self.dropout = torch.nn.Dropout(0.2)
        self.linear = torch.nn.Linear(self.bert_model.config.hidden_size, 50)
    
    def forward(self, input_ids, attn_mask, token_type_ids):
        output = self.bert_model(
            input_ids, 
            attention_mask=attn_mask, 
            token_type_ids=token_type_ids
        )
        output_dropout = self.dropout(output.pooler_output)
        output = self.linear(output_dropout)
        return output
    
model_bert = BERTClass()
model_bert.load_state_dict(torch.load(CKPT))
model_bert = model_bert.to(device)


class BERT_ICD10_Pipeline(Pipeline):
    def _sanitize_parameters(self, **kwargs):
        preprocess_kwargs = {}
        if "maybe_arg" in kwargs:
            preprocess_kwargs["maybe_arg"] = kwargs["maybe_arg"]
        return preprocess_kwargs, {}, {}

    def preprocess(self, text):
        return self.tokenizer(
            text,
            max_length = MAX_LEN,
            padding = 'max_length',
            truncation = True,
            return_tensors='pt'
        ).to(self.device)

    def _forward(self, model_inputs):
        ids = model_inputs['input_ids'].to(self.device, dtype = torch.long)
        mask = model_inputs['attention_mask'].to(self.device, dtype = torch.long)
        token_type_ids = model_inputs['token_type_ids'].to(self.device, dtype = torch.long)
        outputs = self.model(ids, mask, token_type_ids).to(self.device)
        return outputs

    def postprocess(self, model_outputs):
        probs = F.sigmoid(model_outputs).detach().cpu().numpy() # if there's more than one possible diagnosis

        output = []
        for i, prob in enumerate(probs[0]):
            label = self.model.config.id2label[i]
            score = prob
            output.append({"label": label, "score": score})
        # print(output)
        return output
    
pipeline = BERT_ICD10_Pipeline(model=model_bert, tokenizer=tokenizer_bert, device = device)
masker = shap.maskers.Text(pipeline.tokenizer)
explainer = shap.Explainer(pipeline, masker)
shap_input = dataset['test']['text'][:1]
shap_values = explainer(shap_input)
# filename = os.path.join(destination_dir, "shap_explainer_batch_5")
# with open(filename, "wb") as f:
#     pickle.dump(shap_values, f)

shap_df = pd.DataFrame(shap_values)
shap_df.to_parquet('shap_explainer_batch_1.parquet', compression='gzip')

# Test pickle file
# file = open("./shap_explainer_batch_5", "rb")
# data = pickle.load(file)
# print(data)

loaded_shap_df = pd.read_parquet('shap_explainer_batch_1.parquet')
print(loaded_shap_df)
