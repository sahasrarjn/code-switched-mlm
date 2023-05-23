from transformers import BertModel, BertForMaskedLM, BertForQuestionAnswering, BertForSequenceClassification

import torch
import torch.nn as nn
import torch.nn.functional as F

import logging
logger = logging.getLogger(__name__)


########################################
# BERTModel with Residual Connections  #
########################################
class ResidualBertModel(BertModel):
    def __init__(self, config, out_embed=768, res_layer=2, res_dropout=0.2) -> None:
        super(ResidualBertModel, self).__init__(config)
        self.out_embed = out_embed
        self.res_layer = res_layer
        self.res_dropout = res_dropout
        self.ln = nn.LayerNorm((config.hidden_size,))
        logging.info("ResidualBertModel")
        logging.info("RES_LAYER: {}".format(self.res_layer))
        logging.info("DROPOUT: {}".format(self.res_dropout))
    

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
    ):
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device)
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, input_shape, device)

        # If a 2D ou 3D attention mask is provided for the cross-attention
        # we need to make broadcastabe to [batch_size, num_heads, seq_length, seq_length]
        if self.config.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        embedding_output = self.embeddings(
            input_ids=input_ids, position_ids=position_ids, token_type_ids=token_type_ids, inputs_embeds=inputs_embeds
        )
        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
        )

        skip_conn = encoder_outputs[1][self.res_layer]
        final_out = encoder_outputs[1][-1]
        resid_out = F.dropout(skip_conn, p=self.res_dropout) + final_out

        pooled_output = self.pooler(resid_out)

        outputs = (resid_out, pooled_output,) + encoder_outputs[
            1:
        ]  # add hidden_states and attentions if they are here
        return outputs  # sequence_output, pooled_output, (hidden_states), (attentions)


##############################################
# BERTForMaskedLM with Residual Connections  #
##############################################
class ResidualBertForMaskedLM(BertForMaskedLM):
    def __init__(self, config, out_embed=768, res_layer=2, dropout=0.2) -> None:
        super(ResidualBertForMaskedLM, self).__init__(config)
        self.out_embed = out_embed
        self.res_layer = res_layer
        self.dropout = dropout
        self.ln = nn.LayerNorm((config.hidden_size,))
        logging.info("ResidualBertForMaskedLM")
        logging.info("RES_LAYER: {}".format(self.res_layer))
        logging.info("DROPOUT: {}".format(self.dropout))
    
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
    def __init__(self, config, out_embed=768, res_layer=2, dropout=0.2) -> None:
        super(ResidualBertForQuestionAnswering, self).__init__(config)
        self.out_embed = out_embed
        self.res_layer = res_layer
        self.dropout = dropout
        self.ln = nn.LayerNorm((config.hidden_size,))
        logging.info("ResidualBertForQuestionAnswering")
        logging.info("RES_LAYER: {}".format(self.res_layer))
        logging.info("DROPOUT: {}".format(self.dropout))
    
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


############################################################
# BERTForSequenceClassification with Residual Connections  #
############################################################
class ResidualBertForSequenceClassification(BertForSequenceClassification):
    def __init__(self, config, out_embed=768, res_layer=2, res_dropout=0.2) -> None:
        super(ResidualBertForSequenceClassification, self).__init__(config)
        self.out_embed = out_embed
        self.res_layer = res_layer
        self.res_dropout = res_dropout
        self.ln = nn.LayerNorm((config.hidden_size,))
        logging.info("ResidualBertForSequenceClassification")
        logging.info("RES_LAYER: {}".format(self.res_layer))
        logging.info("DROPOUT: {}".format(self.res_dropout))

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
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
        resid_out = F.dropout(skip_conn, p=self.res_dropout) + final_out
        pooled_output = self.bert.pooler(resid_out)

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here

        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = nn.MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions)


################################################################
# BERTForMaskedLM with Residual Connections and Auxilary Loss  #
################################################################
class ResAuxBertForMaskedLM(BertForMaskedLM):
    def __init__(self, config, out_embed=768, res_layer=2, dropout=0.2) -> None:
        super(ResAuxBertForMaskedLM, self).__init__(config)
        self.out_embed = out_embed
        self.res_layer = res_layer
        self.dropout = dropout
        self.aux_head = nn.Linear(out_embed, 256)
        self.aux_activation = nn.ReLU()
        self.aux_classifier = nn.Linear(256, 2)
        self.ln = nn.LayerNorm((config.hidden_size,))
        logging.info("ResAuxBertForMaskedLM")
        logging.info("RES_LAYER: {}".format(self.res_layer))
        logging.info("DROPOUT: {}".format(self.dropout))
    
    def compute_loss(self, outputs, logits, auxilary_logits, aux_labels, 
                     masked_lm_labels, lm_labels):
        factor = 5e-2
        aux_loss_fct = nn.NLLLoss()
        aux_loss = aux_loss_fct(auxilary_logits.view(-1, 2), aux_labels.view(-1))
        
        if masked_lm_labels is not None:
            loss_fct = nn.CrossEntropyLoss()  # -100 index = padding token
            masked_lm_loss = loss_fct(logits.view(-1, self.config.vocab_size), masked_lm_labels.view(-1))
            loss = masked_lm_loss

        if lm_labels is not None:
            # we are doing next-token prediction; shift prediction scores and input ids by one
            prediction_scores = prediction_scores[:, :-1, :].contiguous()
            lm_labels = lm_labels[:, 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss()
            ltr_lm_loss = loss_fct(logits.view(-1, self.config.vocab_size), lm_labels.view(-1))
            loss = ltr_lm_loss
        
        # loss += factor*aux_loss + 0.001*factor*reg_loss
        loss += factor*aux_loss
        outputs = (loss,) + outputs
        return outputs
    
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
        aux_labels=None
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

        aux_out = self.aux_head(outputs[2][self.res_layer])
        aux_out = self.aux_activation(aux_out)
        aux_out = self.aux_classifier(aux_out)
        auxilary_logits = F.log_softmax(aux_out, dim=-1) # have to use NLLLoss

        outputs = (prediction_scores,) + outputs[2:]

        outputs = self.compute_loss(outputs, prediction_scores, auxilary_logits, aux_labels,
                                    masked_lm_labels, lm_labels)

        return outputs
