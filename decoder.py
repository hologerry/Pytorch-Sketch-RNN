import torch
import torch.nn as nn
import torch.nn.functional as F


class DecoderRNN(nn.Module):
    def __init__(self, hp, N_max, device):
        super().__init__()
        self.hp = hp
        self.N_max = N_max
        self.device = device
        # to init hidden and cell from z:
        self.fc_hc = nn.Linear(hp.Nz, 2 * hp.dec_hidden_size)
        # unidirectional lstm:
        self.lstm = nn.LSTM(hp.Nz + 5, hp.dec_hidden_size, dropout=hp.dropout)
        # create probability distribution parameters from hidden vectors:
        self.fc_params = nn.Linear(hp.dec_hidden_size, 6 * hp.M + 3)

    def forward(self, inputs, z, hidden_cell=None):
        if hidden_cell is None:
            # then we must init from z
            hidden, cell = torch.split(torch.tanh(self.fc_hc(z)), self.hp.dec_hidden_size, 1)
            hidden_cell = (hidden.unsqueeze(0).contiguous(), cell.unsqueeze(0).contiguous())
        outputs, (hidden, cell) = self.lstm(inputs, hidden_cell)
        # in training we feed the lstm with the whole input in one shot
        # and use all outputs contained in 'outputs', while in generate
        # mode we just feed with the last generated sample:
        if self.training:
            y = self.fc_params(outputs.view(-1, self.hp.dec_hidden_size))
        else:
            y = self.fc_params(hidden.view(-1, self.hp.dec_hidden_size))
        # separate pen and mixture params:
        params = torch.split(y, 6, 1)
        params_mixture = torch.stack(params[:-1])  # trajectory
        params_pen = params[-1]  # pen up/down
        # identify mixture params:
        pi, mu_x, mu_y, sigma_x, sigma_y, rho_xy = torch.split(params_mixture, 1, 2)
        # preprocess params::
        len_out = self.N_max + 1 if self.training else 1

        pi = F.softmax(pi.transpose(0, 1).squeeze(), dim=-1).view(len_out, -1, self.hp.M)
        sigma_x = torch.exp(sigma_x.transpose(0, 1).squeeze()).view(len_out, -1, self.hp.M)
        sigma_y = torch.exp(sigma_y.transpose(0, 1).squeeze()).view(len_out, -1, self.hp.M)
        rho_xy = torch.tanh(rho_xy.transpose(0, 1).squeeze()).view(len_out, -1, self.hp.M)
        mu_x = mu_x.transpose(0, 1).squeeze().contiguous().view(len_out, -1, self.hp.M)
        mu_y = mu_y.transpose(0, 1).squeeze().contiguous().view(len_out, -1, self.hp.M)
        q = F.softmax(params_pen, dim=-1).view(len_out, -1, 3)
        return pi, mu_x, mu_y, sigma_x, sigma_y, rho_xy, q, hidden, cell
