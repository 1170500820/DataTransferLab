import sys
sys.path.append('..')

import bmtrain as bmt
bmt.init_distributed(seed=0)

import torch
import torch.nn as nn
import torch.nn.functional as F
from model_center.model import Bert, BertConfig
from model_center.layer import Linear


class BertModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.bert = Bert.from_pretrained('bert-base-uncased')
        self.dense = Linear(config.dim_model, 2)
        bmt.init_parameters(self.dense)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        pooler_output = self.bert(input_ids=input_ids, attention_mask=attention_mask).pooler_output
        logits = self.dense(pooler_output)
        return logits

