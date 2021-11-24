import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers import BertPreTrainedModel
from transformers.modeling_bert import (
    BertLayerNorm,
    BertEncoder,
    BertPooler,
    BertPredictionHeadTransform,
)

from losses import SmoothCrossEntropyLoss
from config import HuggingFaceConfig


class LeadSheetEmbeddings(nn.Module):
    def __init__(self, config: HuggingFaceConfig):
        super().__init__()
        self.pattern_embeddings = nn.Embedding(
            config.n_patterns, config.hidden_size, padding_idx=config.pad_token_id
        )
        self.chord_embeddings = nn.Embedding(
            config.n_chords, config.hidden_size, padding_idx=config.pad_token_id
        )
        self.bar_number_embeddings = nn.Embedding(
            config.n_bar_numbers, config.hidden_size, padding_idx=config.pad_token_id
        )
        self.beat_number_embeddings = nn.Embedding(
            config.n_beat_numbers, config.hidden_size, padding_idx=config.pad_token_id
        )

        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, pattern_ids, chord_ids, bar_numbers, beat_numbers):
        input_shape = pattern_ids.size()
        seq_length = input_shape[1]
        device = pattern_ids.device

        pos_ids = torch.arange(seq_length, dtype=torch.long, device=device)
        pos_ids = pos_ids.unsqueeze(0).expand(input_shape)

        pattern_embeds = self.pattern_embeddings(pattern_ids)
        chord_embeds = self.chord_embeddings(chord_ids)
        bar_number_embeds = self.bar_number_embeddings(bar_numbers)
        beat_number_embeds = self.beat_number_embeddings(beat_numbers)

        embeddings = (
            pattern_embeds + chord_embeds + bar_number_embeds + beat_number_embeds
        )
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class LeadSheetModel(BertPreTrainedModel):
    config_class = HuggingFaceConfig

    def __init__(self, config):
        super().__init__(config)
        self.config = config

        self.embeddings = LeadSheetEmbeddings(config)
        self.encoder = BertEncoder(config)
        self.pooler = BertPooler(config)

        self.init_weights()

    def get_input_embeddings(self):
        return (self.embeddings.pattern_embeddings, self.embeddings.chord_embeddings)

    def set_input_embeddings(self, value):
        # self.embeddings.word_embeddings = value
        raise Exception("not implemented")

    def _prune_heads(self, heads_to_prune):
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    def forward(
        self,
        pattern_ids,
        chord_ids,
        bar_numbers,
        beat_numbers,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
    ):
        input_shape = pattern_ids.size()

        device = pattern_ids.device

        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(
            attention_mask, input_shape, device
        )

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        embedding_output = self.embeddings(
            pattern_ids=pattern_ids,
            chord_ids=chord_ids,
            bar_numbers=bar_numbers,
            beat_numbers=beat_numbers,
        )
        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
        )
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output)

        outputs = (sequence_output, pooled_output,) + encoder_outputs[
            1:
        ]  # add hidden_states and attentions if they are here
        return outputs  # sequence_output, pooled_output, (hidden_states), (attentions)


class LeadSheetPredictionHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.transform = BertPredictionHeadTransform(config)

        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        self.pattern_decoder = nn.Linear(
            config.hidden_size, config.n_patterns, bias=False
        )

        self.pattern_bias = nn.Parameter(torch.zeros(config.n_patterns))

        # Need a link between the two variables so that the bias is correctly resized with `resize_token_embeddings`
        self.pattern_decoder.bias = self.pattern_bias

        self.chord_decoder = nn.Linear(config.hidden_size, config.n_chords, bias=False)
        self.chord_bias = nn.Parameter(torch.zeros(config.n_chords))
        self.chord_decoder.bias = self.chord_bias

    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        pattern_pred = self.pattern_decoder(hidden_states)
        chord_pred = self.chord_decoder(hidden_states)
        return pattern_pred, chord_pred


class LeadSheetForMaskedLM(BertPreTrainedModel):
    config_class = HuggingFaceConfig

    def __init__(self, config):
        super().__init__(config)

        self.bert = LeadSheetModel(config)
        self.cls = LeadSheetPredictionHead(config)

        self.init_weights()

    def tie_weights(self):
        (
            pattern_output_embeddings,
            chord_output_embeddings,
        ) = self.get_output_embeddings()
        pattern_input_embeddings, chord_input_embeddings = self.get_input_embeddings()
        self._tie_or_clone_weights(pattern_output_embeddings, pattern_input_embeddings)
        self._tie_or_clone_weights(chord_output_embeddings, chord_input_embeddings)

    def get_output_embeddings(self):
        return self.cls.pattern_decoder, self.cls.chord_decoder

    def forward(
        self,
        pattern_ids,
        chord_ids,
        bar_numbers,
        beat_numbers,
        attention_mask=None,
        head_mask=None,
        masked_pattern_labels=None,
        masked_chord_labels=None,
        encoder_hidden_states=None,
    ):
        outputs = self.bert(
            pattern_ids=pattern_ids,
            chord_ids=chord_ids,
            bar_numbers=bar_numbers,
            beat_numbers=beat_numbers,
            attention_mask=attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
        )

        sequence_output = outputs[0]
        pattern_pred, chord_pred = self.cls(sequence_output)

        outputs = (pattern_pred, chord_pred) + outputs[
            2:
        ]  # Add hidden states and attention if they are here

        if masked_pattern_labels is not None:
            assert masked_chord_labels is not None

            pattern_loss_fct = SmoothCrossEntropyLoss(
                label_smoothing=0.1, vocab_size=self.config.n_patterns,
            )  # -100 index = padding token
            chord_loss_fct = SmoothCrossEntropyLoss(
                label_smoothing=0.1, vocab_size=self.config.n_chords,
            )  # -100 index = padding token
            masked_pattern_loss = pattern_loss_fct(
                pattern_pred.view(-1, self.config.n_patterns),
                masked_pattern_labels.view(-1),
            )
            masked_chord_loss = chord_loss_fct(
                chord_pred.view(-1, self.config.n_chords), masked_chord_labels.view(-1),
            )
            total_loss = masked_pattern_loss + masked_chord_loss
            outputs = (total_loss, masked_pattern_loss, masked_chord_loss) + outputs

        return outputs  # (ltr_lm_loss), (masked_lm_loss), prediction_scores, (hidden_states), (attentions)

    def prepare_inputs_for_generation(
        self,
        pattern_ids,
        chord_ids,
        bar_numbers,
        beat_numbers,
        attention_mask=None,
        **model_kwargs
    ):
        input_shape = pattern_ids.shape
        effective_batch_size = input_shape[0]

        # if model is used as a decoder in encoder-decoder model, the decoder attention mask is created on the fly
        if attention_mask is None:
            attention_mask = pattern_ids.new_ones(input_shape)

        # if model is does not use a causal mask then add a dummy token
        if self.config.is_decoder is False:
            assert (
                self.config.pad_token_id is not None
            ), "The PAD token should be defined for generation"
            attention_mask = torch.cat(
                [
                    attention_mask,
                    attention_mask.new_zeros((attention_mask.shape[0], 1)),
                ],
                dim=-1,
            )

            dummy_token = torch.full(
                (effective_batch_size, 1),
                self.config.pad_token_id,
                dtype=torch.long,
                device=pattern_ids.device,
            )
            pattern_ids = torch.cat([pattern_ids, dummy_token], dim=1)
            chord_ids = torch.cat([chord_ids, dummy_token], dim=1)
            bar_numbers = torch.cat([bar_numbers, dummy_token], dim=1)
            beat_numbers = torch.cat([beat_numbers, dummy_token], dim=1)

        return {
            "pattern_ids": pattern_ids,
            "chord_ids": chord_ids,
            "bar_numbers": bar_numbers,
            "beat_numbers": beat_numbers,
            "attention_mask": attention_mask,
        }


def enable_output_attention(model: LeadSheetForMaskedLM, output_attention: bool):
    model.output_attention = output_attention
    for layer in model.bert.encoder.layer:
        layer.attention.self.output_attentions = output_attention


def format_attention(attention):
    squeezed = []
    for layer_attention in attention:
        # 1 x num_heads x seq_len x seq_len
        if len(layer_attention.shape) != 4:
            raise ValueError(
                "The attention tensor does not have the correct number of dimensions. Make sure you set "
                "output_attentions=True when initializing your model."
            )
        squeezed.append(layer_attention.squeeze(0))
    # num_layers x num_heads x seq_len x seq_len
    return torch.stack(squeezed)
