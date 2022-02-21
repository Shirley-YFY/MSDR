import numpy as np
import torch
import torch.nn as nn

from model.pytorch.gmsdr_cell import GMSDRCell

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class Seq2SeqAttrs:
    def __init__(self, adj_mx, **model_kwargs):
        self.adj_mx = adj_mx
        self.max_diffusion_step = int(model_kwargs.get('max_diffusion_step', 2))
        self.cl_decay_steps = int(model_kwargs.get('cl_decay_steps', 1000))
        self.filter_type = model_kwargs.get('filter_type', 'laplacian')
        self.num_nodes = int(model_kwargs.get('num_nodes', 1))
        self.num_rnn_layers = int(model_kwargs.get('num_rnn_layers', 1))
        self.rnn_units = int(model_kwargs.get('rnn_units'))
        self.hidden_state_size = self.num_nodes * self.rnn_units
        self.pre_k = int(model_kwargs.get('pre_k', 1))
        self.pre_v = int(model_kwargs.get('pre_v', 1))
        self.input_dim = int(model_kwargs.get('input_dim', 1))
        self.output_dim = int(model_kwargs.get('output_dim', 1))

class EncoderModel(nn.Module, Seq2SeqAttrs):
    def __init__(self, adj_mx, **model_kwargs):
        nn.Module.__init__(self)
        Seq2SeqAttrs.__init__(self, adj_mx, **model_kwargs)
        self.input_dim = int(model_kwargs.get('input_dim', 1))
        self.seq_len = int(model_kwargs.get('seq_len'))  # for the encoder
        self.msdr_layers = nn.ModuleList(
            [GMSDRCell(self.rnn_units, adj_mx, self.max_diffusion_step, self.num_nodes, self.pre_k, self.pre_v,
                       filter_type=self.filter_type) for _ in range(self.num_rnn_layers)])

    def forward(self, inputs, hx_k):
        """
        Encoder forward pass.

        :param inputs: shape (batch_size, self.num_nodes * self.input_dim)
        :param hx_k: (num_layers, batch_size, pre_k, self.num_nodes, self.rnn_units)
               optional, zeros if not provided
        :return: output: # shape (batch_size, self.hidden_state_size)
                 hx_k # shape (num_layers, batch_size, pre_k, self.num_nodes, self.rnn_units)
                 (lower indices mean lower layers)
        """
        hx_ks = []
        output = inputs
        for layer_num, msdr_layer in enumerate(self.msdr_layers):
            next_hidden_state, new_hx_k = msdr_layer(output, hx_k[layer_num])
            hx_ks.append(new_hx_k)
            output = next_hidden_state
        return output, torch.stack(hx_ks)

class DecoderModel(nn.Module, Seq2SeqAttrs):
    def __init__(self, adj_mx, **model_kwargs):
        nn.Module.__init__(self)
        Seq2SeqAttrs.__init__(self, adj_mx, **model_kwargs)
        self.output_dim = int(model_kwargs.get('output_dim', 1))
        self.horizon = int(model_kwargs.get('horizon', 1))  # for the decoder
        self.projection_layer = nn.Linear(self.rnn_units, self.output_dim)
        self.msdr_layers = nn.ModuleList(
            [GMSDRCell(self.rnn_units, adj_mx, self.max_diffusion_step, self.num_nodes, self.pre_k, self.pre_v,
                       filter_type=self.filter_type) for _ in range(self.num_rnn_layers)])

    def forward(self, inputs, hx_k):
        """
        Decoder forward pass.

        :param inputs: shape (batch_size, self.num_nodes * self.output_dim)
        :param hx_k: (num_layers, batch_size, pre_k, num_nodes, rnn_units)
               optional, zeros if not provided
        :return: output: # shape (batch_size, self.num_nodes * self.output_dim)
                 hidden_state # shape (num_layers, batch_size, self.hidden_state_size)
                 (lower indices mean lower layers)
        """
        hx_ks = []
        output = inputs
        for layer_num, msdr_layer in enumerate(self.msdr_layers):
            next_hidden_state, new_hx_k = msdr_layer(output, hx_k[layer_num])
            hx_ks.append(new_hx_k)
            output = next_hidden_state

        projected = self.projection_layer(output.view(-1, self.rnn_units))
        output = projected.view(-1, self.num_nodes * self.output_dim)

        return output, torch.stack(hx_ks)


class GMSDRModel(nn.Module, Seq2SeqAttrs):
    def __init__(self, adj_mx, logger, **model_kwargs):
        super().__init__()
        Seq2SeqAttrs.__init__(self, adj_mx, **model_kwargs)
        self.encoder_model = EncoderModel(adj_mx, **model_kwargs)
        self.decoder_model = DecoderModel(adj_mx, **model_kwargs)
        self.cl_decay_steps = int(model_kwargs.get('cl_decay_steps', 1000))
        self.use_curriculum_learning = bool(model_kwargs.get('use_curriculum_learning', False))
        self._logger = logger
        self.out = nn.Linear(self.rnn_units, self.decoder_model.output_dim)

    def _compute_sampling_threshold(self, batches_seen):
        return self.cl_decay_steps / (
                self.cl_decay_steps + np.exp(batches_seen / self.cl_decay_steps))

    def encoder(self, inputs):
        """
        encoder forward pass on t time steps
        :param inputs: shape (seq_len, batch_size, num_sensor * input_dim)
        :return: hx_k: (num_layers, batch_size, pre_k, num_sensor, rnn_units)
        """
        hx_k = torch.zeros(self.num_rnn_layers, inputs.shape[1], self.pre_k, self.num_nodes, self.rnn_units,
                           device=device)
        outputs = []
        for t in range(self.encoder_model.seq_len):
            output, hx_k = self.encoder_model(inputs[t], hx_k)
            outputs.append(output)
        return torch.stack(outputs), hx_k

    def decoder(self, inputs, hx_k, labels=None, batches_seen=None):
        """
        Decoder forward pass
        :param inputs: (seq_len, batch_size, num_sensor * rnn_units)
        :param hx_k: (num_layers, batch_size, pre_k, num_sensor, rnn_units)
        :param labels: (self.horizon, batch_size, self.num_nodes * self.output_dim) [optional, not exist for inference]
        :param batches_seen: global step [optional, not exist for inference]
        :return: output: (self.horizon, batch_size, self.num_nodes * self.output_dim)
        """
        go_symbol = inputs
        decoder_hx_k = hx_k
        decoder_input = go_symbol

        outputs = []
        start = inputs.shape[0] - self.decoder_model.horizon
        for t in range(self.decoder_model.horizon):
            decoder_output, decoder_hx_k = self.decoder_model(decoder_input[start + t],
                                                              decoder_hx_k)
            outputs.append(decoder_output)
        outputs = torch.stack(outputs)
        return outputs

    def forward(self, inputs, labels=None, batches_seen=None):
        """
        seq2seq forward pass
        :param inputs: shape (seq_len, batch_size, num_sensor * input_dim)
        :param labels: shape (horizon, batch_size, num_sensor * output)
        :param batches_seen: batches seen till now
        :return: output: (self.horizon, batch_size, self.num_nodes * self.output_dim)
        """
        encoder_outputs, hx_k = self.encoder(inputs)
        self._logger.debug("Encoder complete, starting decoder")
        outputs = self.decoder(encoder_outputs, hx_k, labels, batches_seen=batches_seen)
        self._logger.debug("Decoder complete")
        if batches_seen == 0:
            self._logger.info(
                "Total trainable parameters {}".format(count_parameters(self))
            )
        return outputs
