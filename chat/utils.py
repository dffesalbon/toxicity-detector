from nltk.corpus import stopwords
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
from transformers import AutoModel, AutoTokenizer
import torch
import numpy as np
import torch.nn.functional as F
from transformers import BertForSequenceClassification

import wget
import pickle
from textblob import TextBlob
from textblob.classifiers import NaiveBayesClassifier
from textblob.classifiers import DecisionTreeClassifier
import nltk
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

pretrained = "dffesalbon/dota-toxic-bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(pretrained)
bert = BertForSequenceClassification.from_pretrained(pretrained)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


nb_download_link = 'https://huggingface.co/dffesalbon/dota-toxic-nb-lexicon/resolve/main/model.pkl'
nb_download = wget.download(nb_download_link)
nb = pickle.load(open(nb_download, 'rb'))

dt_download_link = 'https://huggingface.co/dffesalbon/dota-toxic-dt-lexicon/resolve/main/model.pkl'
dt_download = wget.download(dt_download_link)
dt = pickle.load(open(dt_download, 'rb'))

id2label = {0: 'not toxic', 1: 'somehow toxic', 2: 'totally toxic'}


def predict_text_bert(text, model=bert):
    inputs = tokenizer(text, return_tensors="pt", padding=True,
                       truncation=True, max_length=64)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    out = F.softmax(outputs.logits, dim=1)
    y_out = np.argmax(out.cpu(), axis=1)

    return id2label[y_out.item()]
    # return out, y_out


def predict_text_nb(text):
    return id2label[nb.classify(text)]


def predict_text_dt(text):
    return id2label[dt.classify(text)]
