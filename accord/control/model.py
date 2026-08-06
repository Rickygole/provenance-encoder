"""
Defect-free control: model.

Deliberately boring, on purpose -- this is a CONTROL, not a competitor to
ACCORD. A standard cross-encoder pair classifier:

    [CLS] nursing_sentence [SEP] discharge_sentence [SEP]  -> BiomedBERT
    -> raw [CLS] hidden state (no pooler tanh layer, no extra dense layer)
    -> ONE linear layer -> 1 logit -> BCEWithLogitsLoss

No rule feature, no authority parameter, no ATCG / PPC / GGHE / harm head,
no auxiliary losses, no weak_confidence reweighting. Any of those would
reintroduce the defects this control exists to rule out.
"""
import torch
import torch.nn as nn

MODEL_NAME = "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext"


class PairClassifier(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        self.classifier = nn.Linear(backbone.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask, token_type_ids, labels=None):
        out = self.backbone(
            input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids
        )
        cls = out.last_hidden_state[:, 0, :]
        logit = self.classifier(cls).squeeze(-1)
        loss = None
        if labels is not None:
            loss = nn.functional.binary_cross_entropy_with_logits(logit, labels)
        return loss, logit
