import torch
import torch.nn as nn
import torch.nn.functional as F

class NeuralNet(nn.Module):
    def __init__(self):
        pass

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return X
    
class LSTM(nn.Module):
    def __init__(self, input_dim, lstm_layers=3, embedding_dim=64, hidden_dim=256):
        '''
        Creates network for music generation. Consists of LSTM layers and Linear Layer for multi classfication. 

            Parameters:
                    input_dim (int): number of unique notes
                    lstm_layers (int): number of lstm layers, default 3 
                    embedding_dim (int): param, default 64
                    hidden_dim (int): param, default 256

        '''
        super(LSTM, self).__init__()
        self.hidden_dim = hidden_dim
        # Add embedding layer (map from tensor of positive int)
        self.embedded_notes = nn.Embedding(input_dim, embedding_dim)
        # LSTM layers
        self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_dim, num_layers=lstm_layers)
        
        # Linear space from hidden state space to space of notes 
        self.from_hidden_to_notes = nn.Linear(hidden_dim, input_dim)
        

    def forward(self, notes : torch.Tensor) -> torch.Tensor:
        embeds = self.embedded_notes(notes)
        lstm_out, _ = self.lstm(embeds.view(notes.size(-1), 1, -1))
        notes_space = self.from_hidden_to_notes(lstm_out.view(notes.size(-1), -1))
        notes_scores = F.log_softmax(notes_space, dim=1)
        return notes_scores
