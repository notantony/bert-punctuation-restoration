import torch
import transformers
from torch import nn


class DenseClassifier(nn.Module):
    def __init__(
        self,
        input_dim: int = 768,
        n_classes: int = 5,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.clf = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.Dropout(p=dropout),
            nn.ReLU(),
            nn.Linear(256, n_classes),
            nn.ReLU(),
        )

    def forward(self, x):
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


def load_bert(model_name) -> torch.nn.Module:
    model = transformers.BertModel.from_pretrained(model_name)
    return model


def load_bert_classifier(bert_model_name: str = 'bert-base-uncased') -> torch.nn.Module:
    bert = load_bert(bert_model_name)
    clf = DenseClassifier(768, 5, dropout=0.0)
    return BertClassifier(bert, clf)
