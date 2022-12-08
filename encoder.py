import torch
import torch.nn as nn


class EncoderRNN(nn.Module):
    def __init__(self, hp, device):
        super().__init__()
        self.hp = hp
        self.device = device
        # bidirectional lstm:
        self.lstm = nn.LSTM(5, hp.enc_hidden_size, dropout=hp.dropout, bidirectional=True)
        # create mu and sigma from lstm's last output:
        self.fc_mu = nn.Linear(2 * hp.enc_hidden_size, hp.Nz)
        self.fc_sigma = nn.Linear(2 * hp.enc_hidden_size, hp.Nz)
        # active dropout:
        self.train()

    def forward(self, inputs, batch_size, hidden_cell=None):
        if hidden_cell is None:
            # then must init with zeros
            hidden = torch.zeros(2, batch_size, self.hp.enc_hidden_size).to(self.device)
            cell = torch.zeros(2, batch_size, self.hp.enc_hidden_size).to(self.device)
            hidden_cell = (hidden, cell)
        _, (hidden, cell) = self.lstm(inputs.float(), hidden_cell)
        # hidden is (2, batch_size, hidden_size), we want (batch_size, 2*hidden_size):
        hidden_forward, hidden_backward = torch.split(hidden, 1, 0)
        hidden_cat = torch.cat([hidden_forward.squeeze(0), hidden_backward.squeeze(0)], 1)
        # mu and sigma:
        mu = self.fc_mu(hidden_cat)
        sigma_hat = self.fc_sigma(hidden_cat)
        sigma = torch.exp(sigma_hat / 2.0)
        # N ~ N(0,1)
        N = torch.randn_like(mu)
        # z_size = mu.size()  # N = torch.normal(torch.zeros(z_size), torch.ones(z_size)).to(self.device)
        z = mu + sigma * N
        # mu and sigma_hat are needed for LKL loss
        return z, mu, sigma_hat
