import logging

import torch
from torch.nn import CrossEntropyLoss, Linear
from torch.nn.functional import softmax
from torch.nn.utils.rnn import pack_sequence, pad_packed_sequence
from transformers import BertModel, BertPreTrainedModel

logger = logging.getLogger(__name__)


class BertForQuotationAttribution(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = 1

        self.bert = BertModel(config)
        self.qa_outputs = Linear(config.hidden_size, self.num_labels)

        self.loss_fn = CrossEntropyLoss()
        self.init_weights()

    def forward(
        self,
        input_ids,
        mask_ids,
        *,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        targets=None,
    ):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )

        sequence_output = outputs[0]
        logits = [
            self.qa_outputs(output[mask[mask >= 0]]).squeeze(-1)
            for output, mask in zip(sequence_output, mask_ids)
        ]

        proba, _ = pad_packed_sequence(
            pack_sequence(
                [softmax(logit, dim=0) for logit in logits], enforce_sorted=False
            ),
            batch_first=True,
            total_length=100,
        )

        # logger.info(f"logits: {logits},\ntargets: {targets}\nmask_ids: {mask_ids}")
        outputs = (proba,) + outputs[2:]
        if targets is not None:
            loss = torch.stack(
                [
                    self.loss_fn(logit[None, :], target[None])
                    for logit, target in zip(logits, targets)
                ]
            ).mean()
            correct = torch.tensor(
                [
                    1 if logit.argmax().item() == target.item() else 0
                    for logit, target in zip(logits, targets)
                ],
                device=loss.get_device(),
            ).sum()
            outputs = (loss, correct,) + outputs
        return outputs  # (loss, correct), logits, (hidden_states), (attentions)
