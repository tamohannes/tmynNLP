from typing import Any
from cores import Model
from common import Device
import torch.nn as nn
from transformers import AutoModelForSequenceClassification


@Model.register("huggingface_sequence_classifier")
class HuggingFaceSequenceClassifier(Model, nn.Module):
    def __init__(self, arch: str, num_labels: int) -> None:
        super().__init__()

        self.model = AutoModelForSequenceClassification.from_pretrained(
            arch, num_labels=num_labels, ignore_mismatched_sizes=True).to(Device.device)

        for name, param in self.model.named_parameters():
            if not name.startswith("classifier"):
                param.requires_grad = False

    def forward(self, input_ids=None, attention_mask=None, labels=None, token_type_ids=None) -> Any:
        return self.model(input_ids=input_ids, attention_mask=attention_mask)
