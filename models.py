
import torch
import torch.nn as nn
import numpy as np

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.0):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :]) # Get the output of the last time step
        return out

class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.0):
        super(GRUModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.gru(x, h0)
        out = self.fc(out[:, -1, :]) # Get the output of the last time step
        return out

class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()

class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation)
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(n_outputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation)
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

class TCNModel(nn.Module):
    def __init__(self, input_size, output_size, num_channels, kernel_size=2, dropout=0.2):
        super(TCNModel, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = input_size if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size, padding=(kernel_size-1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)
        self.linear = nn.Linear(num_channels[-1], output_size)

    def forward(self, x):
        # x needs to be (batch_size, input_channels, sequence_length)
        x = x.permute(0, 2, 1) # (batch_size, sequence_length, input_channels) -> (batch_size, input_channels, sequence_length)
        out = self.network(x)
        return self.linear(out[:, :, -1]) # Take the last output

class TransformerModel(nn.Module):
    def __init__(self, input_size, num_heads, num_encoder_layers, dim_feedforward, output_size, dropout=0.0):
        super(TransformerModel, self).__init__()
        self.model_type = 'Transformer'
        self.src_mask = None
        self.pos_encoder = PositionalEncoding(dim_feedforward, dropout)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=dim_feedforward, nhead=num_heads, dim_feedforward=dim_feedforward, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_encoder_layers)
        self.decoder = nn.Linear(dim_feedforward, output_size)
        self.init_weights()

        self.input_linear = nn.Linear(input_size, dim_feedforward)

    def init_weights(self):
        initrange = 0.1
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src):
        src = self.input_linear(src)
        # Apply positional encoding directly to src, ensuring sequence length matches
        # The positional encoding should be applied to the sequence length dimension, not the batch dimension.
        # The `pe` tensor is (1, max_len, d_model)
        # The `src` tensor is (batch_size, sequence_length, d_model)
        # We need to ensure that the sequence length of `src` does not exceed `max_len`
        src = src + self.pos_encoder.pe[:, :src.size(1), :]
        src = self.pos_encoder.dropout(src)
        output = self.transformer_encoder(src)
        output = self.decoder(output[:, -1, :]) # Take the output of the last token
        return output

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0) # Add batch dimension for broadcasting
        self.register_buffer('pe', pe)

    def forward(self, x):
        # This forward method is not directly called by TransformerModel's forward.
        # The positional encoding is added directly in TransformerModel's forward.
        # This method is kept for completeness if PositionalEncoding were to be used standalone.
        # The error was due to `x.size(0)` (batch_size) being used for indexing `pe` which has `1` as its first dimension.
        # We need to use `x.size(1)` (sequence_length) for the second dimension of `pe`.
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

if __name__ == '__main__':
    # Example Usage
    input_size = 3  # Hs, Tp, SLP
    sequence_length = 20
    hidden_size = 64
    num_layers = 2
    output_size = 3 # Predicting next Hs, Tp, SLP
    batch_size = 16

    dummy_input = torch.randn(batch_size, sequence_length, input_size)

    # LSTM Test
    lstm_model = LSTMModel(input_size, hidden_size, num_layers, output_size)
    lstm_output = lstm_model(dummy_input)
    print(f"LSTM Output shape: {lstm_output.shape}")

    # GRU Test
    gru_model = GRUModel(input_size, hidden_size, num_layers, output_size)
    gru_output = gru_model(dummy_input)
    print(f"GRU Output shape: {gru_output.shape}")

    # TCN Test
    num_channels = [64, 64, 64]
    kernel_size = 2
    tcn_model = TCNModel(input_size, output_size, num_channels, kernel_size=kernel_size, dropout=0.2)
    tcn_output = tcn_model(dummy_input)
    print(f"TCN Output shape: {tcn_output.shape}")

    # Transformer Test
    num_heads = 1
    num_transformer_layers = 1
    dim_feedforward = 128
    transformer_model = TransformerModel(input_size, num_heads, num_transformer_layers, dim_feedforward, output_size)
    transformer_output = transformer_model(dummy_input)
    print(f"Transformer Output shape: {transformer_output.shape}")


