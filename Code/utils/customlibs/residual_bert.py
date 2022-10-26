from transformers import BertForMaskedLM, BertForQuestionAnswering

import torch.nn as nn
import torch.nn.functional as F

##############################################
# BERTForMaskedLM with Residual Connections  #
##############################################
class ResidualBertForMaskedLM(BertForMaskedLM):
    def __init__(self, config, out_embed=768, res_layer=3, dropout=0.2) -> None:
        super(ResidualBertForMaskedLM, self).__init__(config)
        self.out_embed = out_embed
        self.res_layer = res_layer
        self.dropout = dropout
        self.ln = nn.LayerNorm((config.hidden_size,))
    
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        masked_lm_labels=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        lm_labels=None,
    ):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
        )

        skip_conn = outputs[2][self.res_layer]
        final_out = outputs[2][-1]
        resid_out = F.dropout(skip_conn, p=self.dropout) + final_out
        prediction_scores = self.cls(resid_out)

        outputs = (prediction_scores,) + outputs[2:]

        if masked_lm_labels is not None:
            loss_fct = nn.CrossEntropyLoss()  # -100 index = padding token
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), masked_lm_labels.view(-1))
            outputs = (masked_lm_loss,) + outputs

        if lm_labels is not None:
            # we are doing next-token prediction; shift prediction scores and input ids by one
            prediction_scores = prediction_scores[:, :-1, :].contiguous()
            lm_labels = lm_labels[:, 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss()
            ltr_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), lm_labels.view(-1))
            outputs = (ltr_lm_loss,) + outputs

        return outputs



#######################################################
# BERTForQuestionAnswering with Residual Connections  #
#######################################################
class ResidualBertForQuestionAnswering(BertForQuestionAnswering):
    def __init__(self, config, out_embed=768, res_layer=3, dropout=0.2) -> None:
        super(ResidualBertForQuestionAnswering, self).__init__(config)
        self.out_embed = out_embed
        self.res_layer = res_layer
        self.dropout = dropout
        self.ln = nn.LayerNorm((config.hidden_size,))
    
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        start_positions=None,
        end_positions=None,
    ):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )

        skip_conn = outputs[2][self.res_layer]
        final_out = outputs[2][-1]
        resid_out = F.dropout(skip_conn, p=self.dropout) + final_out

        logits = self.qa_outputs(resid_out)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        outputs = (start_logits, end_logits,) + outputs[2:]

        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions.clamp_(0, ignored_index)
            end_positions.clamp_(0, ignored_index)

            loss_fct = nn.CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2
            outputs = (total_loss,) + outputs

        return outputs
