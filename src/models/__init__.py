from typing import Union, Optional
from pathlib import Path

import torch
import transformers
from torch import nn

from utils import get_default_device
from utils.paths import MODELS_DIR


class DenseClassifier(nn.Module):
    def __init__(
        self,
        input_dim: int = 768,
        hidden_dim: int = 256, 
        n_classes: int = 5,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.clf = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Dropout(p=dropout),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_classes),
        )

    def forward(self, x):
        return self.clf(x)


class LstmClassifier(nn.Module):
    def __init__(
        self,
        input_dim: int = 768,
        hidden_dim: int = 512,
        n_classes: int = 5,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.rnn = nn.Sequential(
            nn.LSTM(input_dim, hidden_dim, batch_first=True, bidirectional=True),
            nn.Dropout(p=dropout),
            nn.ReLU(),
        )
        self.clf = nn.Linear(2 * hidden_dim, n_classes)

    def forward(self, x):
        x, _ = self.rnn(x)
        return self.clf(x)


class BertClassifier(nn.Module):
    def __init__(
        self,
        bert_model: nn.Module,
        classifier_model: nn.Module,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.bert_model = bert_model
        self.dropout = nn.Dropout(p=dropout)
        self.classifier_model = classifier_model

    def forward(self, x):
        x = self.bert_model(input_ids=x).last_hidden_state
        x = self.dropout(x)
        x = self.classifier_model(x)
        return x


def load_bert(model_name: str) -> nn.Module:
    model = transformers.BertModel.from_pretrained(model_name)
    return model


def load_bert_classifier(
    bert_model_name: str = 'bert-base-uncased',
    classifier_name: str = 'dense-256',
    bert_emb_dropout: float = 0.2,  
    clf_dropout: float = 0.2,
    n_classes: int = 5,
) -> torch.nn.Module:
    bert = load_bert(bert_model_name)
    
    if classifier_name == 'dense-256':
        clf = DenseClassifier(768, 256, n_classes=n_classes, dropout=clf_dropout)
    elif classifier_name == 'dense-1568':
        clf = DenseClassifier(768, 1568, n_classes=n_classes, dropout=clf_dropout)
    elif classifier_name == 'lstm-512':
        clf = LstmClassifier(768, 512, n_classes=n_classes, dropout=clf_dropout)
    else:
        raise ValueError(f'Unexpected classifier name: {classifier_name}')

    return BertClassifier(bert, clf, dropout=bert_emb_dropout)


def get_model_by_name(name: str) -> torch.nn.Module:
    if name == 'base-256':
        return load_bert_classifier('bert-base-uncased', 'dense-256')
    elif name == 'base-1568':
        return load_bert_classifier('bert-base-uncased', 'dense-1568')
    elif name == 'base-lstm':
        return load_bert_classifier('bert-base-uncased', 'lstm-512')
    else:
        raise ValueError('Unknown model name')


def load_model_checkpoint(model_name: str, model_path: Optional[Union[Path, str]] = None) -> torch.nn.Module:
    if model_path is None:
        model_path = MODELS_DIR / f'{model_name}.pth'
    model_path = str(model_path)

    device = get_default_device()

    model = get_model_by_name(model_name)
    model.load_state_dict(torch.load(model_path, map_location=device))
    return model
