import os

import numpy as np
import torch

from torch import nn, optim

from data_utils import make_batch, make_image, sample_bivariate_normal
from decoder import DecoderRNN
from encoder import EncoderRNN
from lr_decay import lr_decay


class Model:
    def __init__(self, ckpt_save_path, image_save_path, hp, N_max, device):
        self.encoder = EncoderRNN(hp, device).to(device)
        self.decoder = DecoderRNN(hp, N_max, device).to(device)
        self.encoder_optimizer = optim.Adam(self.encoder.parameters(), hp.lr)
        self.decoder_optimizer = optim.Adam(self.decoder.parameters(), hp.lr)
        self.eta_step = hp.eta_min
        self.ckpt_save_path = ckpt_save_path
        self.image_save_path = image_save_path
        self.hp = hp
        self.device = device
        self.N_max = N_max

    def make_target(self, batch, lengths):
        eos = torch.stack([torch.Tensor([0, 0, 0, 0, 1])] * batch.size()[1]).to(self.device).unsqueeze(0)
        batch = torch.cat([batch, eos], 0)
        mask = torch.zeros(self.N_max + 1, batch.size()[1])
        for index, length in enumerate(lengths):
            mask[:length, index] = 1
        mask = mask.to(self.device)
        dx = torch.stack([batch.data[:, :, 0]] * self.hp.M, 2)
        dy = torch.stack([batch.data[:, :, 1]] * self.hp.M, 2)
        p1 = batch.data[:, :, 2]
        p2 = batch.data[:, :, 3]
        p3 = batch.data[:, :, 4]
        p = torch.stack([p1, p2, p3], 2)
        return mask, dx, dy, p

    def train(self, iter, data):
        self.encoder.train()
        self.decoder.train()
        batch, lengths = make_batch(data, self.hp.batch_size, self.N_max, self.device)
        # encode:
        z, self.mu, self.sigma = self.encoder(batch, self.hp.batch_size)
        # create start of sequence:
        sos = torch.stack([torch.Tensor([0, 0, 1, 0, 0])] * self.hp.batch_size).to(self.device).unsqueeze(0)
        # had sos at the beginning of the batch:
        batch_init = torch.cat([sos, batch], 0)
        # expend z to be ready to concatenate with inputs:
        z_stack = torch.stack([z] * (self.N_max + 1))
        # inputs is concatenation of z and batch_inputs
        inputs = torch.cat([batch_init, z_stack], 2)
        # decode:
        self.pi, self.mu_x, self.mu_y, self.sigma_x, self.sigma_y, self.rho_xy, self.q, _, _ = self.decoder(inputs, z)
        # prepare targets:
        mask, dx, dy, p = self.make_target(batch, lengths)
        # prepare optimizers:
        self.encoder_optimizer.zero_grad()
        self.decoder_optimizer.zero_grad()
        # update eta for LKL:
        self.eta_step = 1 - (1 - self.hp.eta_min) * self.hp.R
        # compute losses:
        kl_loss = self.kullback_leibler_loss()
        recon_loss = self.reconstruction_loss(mask, dx, dy, p, iter)
        loss = recon_loss + kl_loss
        # gradient step
        loss.backward()
        # gradient clipping
        nn.utils.clip_grad_norm_(self.encoder.parameters(), self.hp.grad_clip)
        nn.utils.clip_grad_norm_(self.decoder.parameters(), self.hp.grad_clip)
        # optim step
        self.encoder_optimizer.step()
        self.decoder_optimizer.step()
        # some print and save:
        if iter % 10 == 0:
            print(
                f"iter {iter:06d}, loss: {loss.item():.4f} recon_loss: {recon_loss.item():.4f} kl_loss: {kl_loss.item():.4f}"
            )
            self.encoder_optimizer = lr_decay(self.encoder_optimizer)
            self.decoder_optimizer = lr_decay(self.decoder_optimizer)
        if iter % 100 == 0:
            self.save(iter)
            self.conditional_generation(iter)

    def bivariate_normal_pdf(self, dx, dy):
        z_x = ((dx - self.mu_x) / self.sigma_x) ** 2
        z_y = ((dy - self.mu_y) / self.sigma_y) ** 2
        z_xy = (dx - self.mu_x) * (dy - self.mu_y) / (self.sigma_x * self.sigma_y)
        z = z_x + z_y - 2 * self.rho_xy * z_xy
        exp = torch.exp(-z / (2 * (1 - self.rho_xy**2)))
        norm = 2 * np.pi * self.sigma_x * self.sigma_y * torch.sqrt(1 - self.rho_xy**2)
        return exp / norm

    def reconstruction_loss(self, mask, dx, dy, p, iter):
        pdf = self.bivariate_normal_pdf(dx, dy)
        L_s = -torch.sum(mask * torch.log(1e-5 + torch.sum(self.pi * pdf, 2))) / float(self.N_max * self.hp.batch_size)
        L_p = -torch.sum(p * torch.log(self.q)) / float(self.N_max * self.hp.batch_size)
        return L_s + L_p

    def kullback_leibler_loss(self):
        KL_loss = (
            -0.5
            * torch.sum(1 + self.sigma - self.mu**2 - torch.exp(self.sigma))
            / float(self.hp.Nz * self.hp.batch_size)
        )
        KL_min = torch.Tensor([self.hp.KL_min]).to(self.device).detach()
        return self.hp.wKL * self.eta_step * torch.max(KL_loss, KL_min)

    def save(self, iter):
        torch.save(self.encoder.state_dict(), os.path.join(self.ckpt_save_path, f"encoder_iter_{iter:06d}.pth"))
        torch.save(self.decoder.state_dict(), os.path.join(self.ckpt_save_path, f"decoder_iter_{iter:06d}.pth"))

    def load(self, encoder_name, decoder_name):
        saved_encoder = torch.load(encoder_name)
        saved_decoder = torch.load(decoder_name)
        self.encoder.load_state_dict(saved_encoder)
        self.decoder.load_state_dict(saved_decoder)

    def conditional_generation(self, data, iter):
        print(f"conditional generation for iter {iter:06d}...")
        batch, lengths = make_batch(data, 1, self.N_max, self.device)
        # should remove dropouts:
        self.encoder.train(False)
        self.decoder.train(False)
        # encode:
        z, _, _ = self.encoder(batch, 1)
        sos = torch.Tensor([0, 0, 1, 0, 0]).view(1, 1, -1).to(self.device)
        s = sos
        seq_x = []
        seq_y = []
        seq_z = []
        hidden_cell = None
        for i in range(self.N_max):
            input = torch.cat([s, z.unsqueeze(0)], 2)
            # decode:
            (
                self.pi,
                self.mu_x,
                self.mu_y,
                self.sigma_x,
                self.sigma_y,
                self.rho_xy,
                self.q,
                hidden,
                cell,
            ) = self.decoder(input, z, hidden_cell)
            hidden_cell = (hidden, cell)
            # sample from parameters:
            s, dx, dy, pen_down, eos = self.sample_next_state()
            # ------
            seq_x.append(dx)
            seq_y.append(dy)
            seq_z.append(pen_down)
            if eos:
                print("end at ", i)
                break
        # visualize result:
        x_sample = np.cumsum(seq_x, 0)
        y_sample = np.cumsum(seq_y, 0)
        z_sample = np.array(seq_z)
        sequence = np.stack([x_sample, y_sample, z_sample]).T
        make_image(sequence, iter, self.image_save_path)

    def sample_next_state(self):
        def adjust_temp(pi_pdf):
            pi_pdf = np.log(pi_pdf) / self.hp.temperature
            pi_pdf -= pi_pdf.max()
            pi_pdf = np.exp(pi_pdf)
            pi_pdf /= pi_pdf.sum()
            return pi_pdf

        # get mixture index:
        pi = self.pi[0, 0, :].detach().cpu().numpy()
        pi = adjust_temp(pi)
        pi_idx = np.random.choice(self.hp.M, p=pi)
        # get pen state:
        q = self.q[0, 0, :].detach().cpu().numpy()
        q = adjust_temp(q)
        q_idx = np.random.choice(3, p=q)
        # get mixture params:
        mu_x = self.mu_x[0, 0, pi_idx].detach().cpu().numpy()
        mu_y = self.mu_y[0, 0, pi_idx].detach().cpu().numpy()
        sigma_x = self.sigma_x[0, 0, pi_idx].detach().cpu().numpy()
        sigma_y = self.sigma_y[0, 0, pi_idx].detach().cpu().numpy()
        rho_xy = self.rho_xy[0, 0, pi_idx].detach().cpu().numpy()
        x, y = sample_bivariate_normal(mu_x, mu_y, sigma_x, sigma_y, rho_xy, self.hp, greedy=False)
        next_state = torch.zeros(5).to(self.device)
        next_state[0] = x
        next_state[1] = y
        next_state[q_idx + 2] = 1
        return next_state.view(1, 1, -1), x, y, q_idx == 1, q_idx == 2
