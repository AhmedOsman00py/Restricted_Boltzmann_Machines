import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim
from torch.autograd import Variable


# --- Implementation of RBM
class RBM(nn.Module):
    """Restricted Boltzmann Machine"""

    def __init__(self, n_vis=784, n_hin=500, k=5):
        """
        :param n_vis: the number of visible units in the RBM
        :param n_hin: the number of hidden units in the RBM
        :param k: the number of Gibbs sampling steps to use during training
        """
        super(RBM, self).__init__()
        self.W = nn.Parameter(torch.randn(n_hin, n_vis) * 1e-2)

        self.v_bias = nn.Parameter(torch.zeros(n_vis))
        self.h_bias = nn.Parameter(torch.zeros(n_hin))
        self.k = k

    def sample_from_p(self, p):
        """
        this method takes a Tensor p as input and returns a tensor of the same shape,
        where each element is either 0 or 1
        :param p: Tensor of probabilities
        :return: Tensor of shape p
        """
        return F.relu(torch.sign(p - Variable(torch.rand(p.size()))))

    def v_to_h(self, v):
        """
        takes a Tensor v (visible units) and returns two Tensors,
        p_h (probability of the hidden units being activated given visible units)
            -> P(h=1|v) = σ(W.T * v + h_biais)
        sample_h (binary tensor representing a sample from the distribution of the hidden units, given the visible units)
        :param v: visible units
        :return: p_h, sample_h
        """
        p_h = F.sigmoid(F.linear(v, self.W, self.h_bias))
        sample_h = self.sample_from_p(p_h)
        return p_h, sample_h

    def h_to_v(self, h):
        """
        takes a Tensor h (hidden units) and returns two Tensors,
        p_v (probability of the visible units given hidden units)
            -> P(v=1|h) = σ(W * h + v_biais)
        sample_v (binary tensor which is a sample from the probability distribution p_v)
        :param h: hidden units
        :return: p_v, sample_v
        """
        p_v = F.sigmoid(F.linear(h, self.W.t(), self.v_bias))
        sample_v = self.sample_from_p(p_v)
        return p_v, sample_v

    def forward(self, v):
        """
        the forward() function takes in a visible tensor v, passes it through the v_to_h() function to compute the probability and the sample of the hidden units.
        then it initializes h_ to be the sample of the hidden units h1 obtained from the v_to_h() function.
        it alternates between sampling the visible units and the hidden units for k iterations
        :param v:
        :return:
        """
        _, h1 = self.v_to_h(v)

        h_ = h1
        for _ in range(self.k):
            _, v_ = self.h_to_v(h_)
            _, h_ = self.v_to_h(v_)

        return v, v_

    def free_energy(self, v):
        """
        function that takes a visible Tensor and returns the negative log-likelihood of the input under the model
        :param v: visible units
        :return: negative log-likelihood
        """
        vbias_term = v.mv(self.v_bias)
        wx_b = F.linear(v, self.W, self.h_bias)
        hidden_term = wx_b.exp().add(1).log().sum(1)
        return (-hidden_term - vbias_term).mean()


# --- Training of a RBM
def train_rbm(rbm, train_loader, learning_rate=1e-3, num_epochs=10):
    """train a RBM and compute loss at each epoch"""
    train_op = optim.SGD(rbm.parameters(), lr=learning_rate)

    loss_ = []
    loss_epochs = []
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for i, (batch, _) in enumerate(train_loader):
            sample_data = batch.view(-1, 784)
            v, v1 = rbm(sample_data)
            loss = rbm.free_energy(v) - rbm.free_energy(v1)
            loss_.append(loss.item())
            train_op.zero_grad()
            loss.backward()
            train_op.step()
            epoch_loss += loss.item()

            if (i + 1) % 100 == 0:
                print('Epoch: {:3d}  Step: {:4d}/{}  Loss: {:.4f}'.format(epoch + 1, i + 1, len(train_loader),
                                                                          loss.item()))

        epoch_loss /= len(train_loader)
        loss_epochs.append(epoch_loss)
        print('Epoch: {:3d} Loss: {:.4f}'.format(epoch + 1, epoch_loss))

    return v, v1, loss_, loss_epochs