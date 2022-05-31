import torch
import torch.nn as nn
import torch.nn.functional as F

from polyphonic_music_modeling.RTransformer import RTransformer


class NeuralNet(nn.Module):
    def __init__(self):
        pass

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return X


class LSTM(nn.Module):
    def __init__(self, input_dim, lstm_layers=3, embedding_dim=100, hidden_dim=100):
        """
        https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html
        Creates network for music generation. Consists of LSTM layers and Linear Layer for multi classfication.

            Parameters:
                    input_dim (int): number of unique notes
                    lstm_layers (int): number of lstm layers, default 3
                    embedding_dim (int): param, default 100
                    hidden_dim (int): param, default 100

        """
        super(LSTM, self).__init__()
        self.hidden_dim = hidden_dim
        # Add embedding layer (map from tensor of positive int)
        self.embedded_notes = nn.Embedding(input_dim, embedding_dim)
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=embedding_dim, hidden_size=hidden_dim, num_layers=lstm_layers
        )
        # Linear space from hidden state space to space of notes
        self.from_hidden_to_notes = nn.Linear(hidden_dim, input_dim)

    def forward(self, notes: torch.Tensor) -> torch.Tensor:
        embeds = self.embedded_notes(notes)
        lstm_out, _ = self.lstm(embeds.view(notes.size(-1), 1, -1))
        notes_space = self.from_hidden_to_notes(lstm_out.view(notes.size(-1), -1))
        notes_scores = F.log_softmax(notes_space, dim=1)
        return notes_scores


class GRU(nn.Module):
    def __init__(self, input_dim, gru_layers=3, embedding_dim=100, hidden_dim=100):
        """
        https://pytorch.org/docs/stable/generated/torch.nn.GRU.html

        Creates network for music generation. Consists of GRU layers:

            Parameters:
                    input_dim (int): number of unique notes
                    lstm_layers (int): number of gru layers, default 3
                    embedding_dim (int): param, default 100
                    hidden_dim (int): param, default 100

        """
        super(GRU, self).__init__()
        self.hidden_dim = hidden_dim
        # Add embedding layer (map from tensor of positive int)
        self.embedded_notes = nn.Embedding(input_dim, embedding_dim)
        # GRU layers
        self.gru = nn.GRU(
            input_size=embedding_dim, hidden_size=hidden_dim, num_layers=gru_layers
        )
        # Linear space from hidden state space to space of notes
        self.from_hidden_to_notes = nn.Linear(hidden_dim, input_dim)

    def forward(self, notes: torch.Tensor) -> torch.Tensor:
        embeds = self.embedded_notes(notes)
        gru_out, _ = self.gru(embeds.view(notes.size(-1), 1, -1))
        notes_space = self.from_hidden_to_notes(gru_out.view(notes.size(-1), -1))
        notes_scores = F.log_softmax(notes_space, dim=1)
        return notes_scores


class RT(nn.Module):
    def __init__(
        self,
        input_size,
        d_model,
        output_size,
        h,
        rnn_type,
        ksize,
        n,
        n_level,
        dropout,
        emb_dropout,
    ):
        super(RT, self).__init__()
        self.encoder = nn.Linear(input_size, d_model)
        self.rt = RTransformer(d_model, rnn_type, ksize, n_level, n, h, dropout)
        self.linear = nn.Linear(d_model, output_size)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        x = torch.nn.functional.one_hot(x, 88).float()  # ).requires_grad_()
        x = self.encoder(x).cpu()
        output = self.rt(x)
        output = self.linear(output).double()
        output = self.sig(output)

        output = (torch.argmax(output, dim=2, keepdim=False).float()).requires_grad_()
        return output.squeeze(0)
