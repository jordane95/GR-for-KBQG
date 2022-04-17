from typing import Optional, Tuple

from torch import nn

from transformers.configuration_bart import BartConfig
from modeling_bart import PretrainedBartModel, BartStructureEncoder, _filter_out_falsey_values


class GraphEncoder(PretrainedBartModel):
    def __init__(self, config: BartConfig):
        super().__init__(config)

        padding_idx, vocab_size = config.pad_token_id, config.vocab_size

        config.is_encoder_decoder = False
        self.shared = nn.Embedding(vocab_size, config.d_model, padding_idx)

        self.encoder = BartStructureEncoder(config, self.shared)

        self.init_weights()

    def forward(
        self,
        input_ids,
        attention_mask=None,
        encoder_outputs: Optional[Tuple] = None,
        input_node_ids=None,
        input_edge_ids=None,
        node_length=None,
        edge_length=None,
        adj_matrix=None,
        output_attentions=None,
        output_hidden_states=None,
    ):
        """
        Args:
            input_ids:

        Returns:
            torch.Tensor: [batch_size, seq_len, emb_dim]
            attention_mask: [batch_size, seq_len]
        """

        if encoder_outputs is None:
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                input_node_ids=input_node_ids,
                input_edge_ids=input_edge_ids,
                node_length=node_length,
                edge_length=edge_length,
                adj_matrix=adj_matrix,
            )
        assert isinstance(encoder_outputs, tuple)

        # Attention and hidden_states will be [] or None if they aren't needed
        encoder_outputs: Tuple = _filter_out_falsey_values(encoder_outputs)
        return encoder_outputs[0][:, 0] # [graph] embedding for each input graph, [batch_size, emb_dim]

    def get_input_embeddings(self):
        return self.shared

    def set_input_embeddings(self, value):
        self.shared = value
        self.encoder.embed_tokens = self.shared
