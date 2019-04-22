import unittest

import numpy as np
import torch
from torch import nn
from torch.nn import functional


class GatedLayer(nn.Module):
    def __init__(self, blocks, config):
        super().__init__()
        self.blocks = blocks
        self.g_logits = config["g"]
        self.q_logits = config["q"]
        self.w = config["w"]
        self.sparse = config["sparse"]

    def forward(self, iput, emb, log_probs=0., extra_loss=0., labels=None):

        """
        :param iput: N x I
        :param emb: N x E
        :param log_probs: N
        :param extra_loss: N
        :param labels: N
        :return: N x O, N, N
        """

        g_logits = self.g_logits(emb)  # N x B
        w = self.w(emb)  # N x B
        if labels is not None:
            q_logits = self.q_logits(iput, emb, labels)
        else:
            q_logits = None

        b, new_log_probs, new_extra_loss = self.get_choices(g_logits, q_logits=q_logits)  # N x B, N, N
        output = torch.stack([block(iput) for block in self.blocks], dim=1)  # N x B x O
        if self.sparse:
            raise NotImplementedError
        else:
            output = output * w[:, :, None] * b[:, :, None]
        extra_loss += new_extra_loss
        if isinstance(new_extra_loss, torch.Tensor):
            extra_loss += new_extra_loss.detach() * log_probs
        return output.sum(dim=1), log_probs + new_log_probs, extra_loss

    def get_choices(self, g_logits, q_logits):
        raise NotImplementedError


class BlendingLayer(GatedLayer):
    def get_choices(self, g_logits, q_logits=None):
        return torch.sigmoid(g_logits), 0, 0


class SamplingLayer(GatedLayer):
    def __init__(self, blocks, config):
        super().__init__(blocks, config)
        self.reparam = config["reparam"]
        self.discrete = config["discrete"]
        self.temp = 0.1
        self.reg = config["reg"]

    def get_choices(self, g_logits, q_logits=None):
        if self.training:
            sample_penalty = (torch.sigmoid(g_logits) * self.reg).sum(dim=1)
            if self.reparam:
                distr = torch.distributions.RelaxedBernoulli(self.temp, logits=g_logits)
                r = distr.rsample()
                if not self.discrete:
                    return r, 0, sample_penalty
                else:
                    discr = torch.round(r)
                    return r + (discr - r).detach(), 0, sample_penalty
            else:
                distr = torch.distributions.Bernoulli(logits=g_logits)
                r = distr.sample()
                log_q_prob = functional.logsigmoid(q_logits) * r + functional.logsigmoid(-q_logits) * (1 - r)
                return r, log_q_prob.sum(dim=1), sample_penalty
        else:
            out = torch.round(torch.sigmoid(g_logits))
            return out, 0, 0


class VILayer(GatedLayer):
    def __init__(self, blocks, config):
        super().__init__(blocks, config)
        self.reparam = config["reparam"]
        self.discrete = config["discrete"]
        self.temp = 0.1
        self.reg = config["reg"]

    def get_choices(self, g_logits, q_logits=None):

        # q_logits is q(b|x, t, y), N x B
        # g_logits is p(b|t), N x B

        if self.training:
            sample_penalty = (torch.sigmoid(q_logits) * self.reg).sum(dim=1)

            log_p_prob0 = functional.logsigmoid(-g_logits)
            log_p_prob1 = functional.logsigmoid(g_logits)
            log_q_prob0 = functional.logsigmoid(-q_logits)
            log_q_prob1 = functional.logsigmoid(q_logits)
            q_prob = torch.sigmoid(q_logits)

            prob_one = q_prob * (log_q_prob1 - log_p_prob1)
            prob_zero = (1. - q_prob) * (log_q_prob0 - log_p_prob0)
            kl = (prob_one + prob_zero).sum(1)  # N

            if self.reparam:
                distr = torch.distributions.RelaxedBernoulli(self.temp, logits=q_logits)
                r = distr.rsample()
                if not self.discrete:
                    return r, 0, kl + sample_penalty
                else:
                    discr = torch.round(r)
                    return r + (discr - r).detach(), 0, kl + sample_penalty
            else:
                distr = torch.distributions.Bernoulli(logits=q_logits)  # N x B
                r = distr.sample()
                log_q_prob = functional.logsigmoid(q_logits) * r + functional.logsigmoid(-q_logits) * (1 - r)
                return r, log_q_prob.sum(dim=1), kl + sample_penalty
        else:
            out = torch.round(torch.sigmoid(g_logits))
            return out, 0, 0


class TestGatedLayer(unittest.TestCase):

    def test_blend_get_choices(self):
        config = {
            "q": None,
            "g": None,
            "w": None
        }

        self.convs = nn.ModuleList([nn.Sequential(
            nn.Conv2d(3, 3, 3),
            nn.ReLU(),
        ) for _ in range(3)])

        gating_layer = BlendingLayer(self.convs, config)

        arr = [[1., 2., 3.], [-1., 0., 1.], [-2., -1., 0.], [-2., -1., -3.]]
        soln = 1 / (1 + np.exp(-np.array(arr)))
        g_logits = torch.stack([torch.tensor(a) for a in arr])
        choices, _, _ = gating_layer.get_choices(g_logits)
        self.assertTrue(np.allclose(choices.numpy(), soln))

    def test_sample_get_choices(self):
        config = {
            "q": None,
            "g": None,
            "w": None,
            "reparam": True,
            "discrete": False
        }

        self.convs = nn.ModuleList([nn.Sequential(
            nn.Conv2d(3, 3, 3),
            nn.ReLU(),
        ) for _ in range(3)])

        gating_layer = SamplingLayer(self.convs, config)

        arr = [[100., -100., 100.], [-1., 0., 1.], [-2., -1., 0.], [-2., -1., -3.]]
        g_logits = torch.stack([torch.tensor(a) for a in arr])
        choices, probs, _ = gating_layer.get_choices(g_logits)
        self.assertTrue((choices[0].numpy() == np.array([1, 0, 1])).all())

        config = {
            "q": None,
            "g": None,
            "w": None,
            "reparam": True,
            "discrete": True
        }

        self.convs = nn.ModuleList([nn.Sequential(
            nn.Conv2d(3, 3, 3),
            nn.ReLU(),
        ) for _ in range(3)])

        gating_layer = SamplingLayer(self.convs, config)

        arr = [[100., -100., 100.], [-1., 0., 1.], [-2., -1., 0.], [-2., -1., -3.]]
        g_logits = torch.stack([torch.tensor(a) for a in arr])
        choices, probs, _ = gating_layer.get_choices(g_logits)
        self.assertTrue((choices[0].numpy() == np.array([1, 0, 1])).all())

        config = {
            "q": None,
            "g": None,
            "w": None,
            "reparam": False,
            "discrete": False
        }

        self.convs = nn.ModuleList([nn.Sequential(
            nn.Conv2d(3, 3, 3),
            nn.ReLU(),
        ) for _ in range(3)])

        gating_layer = SamplingLayer(self.convs, config)

        arr = [[5., -100., 5.], [-1., 0., 1.], [-2., -1., 0.], [-2., -1., -3.]]
        g_logits = torch.stack([torch.tensor(a) for a in arr])
        soln = np.log(1 / (1 + np.exp(-np.array(arr))))
        choices, probs, _ = gating_layer.get_choices(g_logits)
        self.assertTrue((choices[0].numpy() == np.array([1, 0, 1])).all())
        self.assertTrue(np.allclose(probs[0], np.array([soln[0][0], np.log(1 - np.exp(soln[0][1])), soln[0][2]])))

    def test_VI_get_choices(self):
        config = {
            "q": None,
            "g": None,
            "w": None,
            "reparam": True,
            "discrete": False
        }

        self.convs = nn.ModuleList([nn.Sequential(
            nn.Conv2d(3, 3, 3),
            nn.ReLU(),
        ) for _ in range(3)])

        gating_layer = VILayer(self.convs, config)

        g_logits = torch.stack([torch.tensor(a) for a in [[100., -100., 100.], [-1., 0., 1.]]])
        q_logits = torch.stack([torch.tensor(a) for a in [[-100., -100., 5.], [-1., 0., 1.]]])
        choices, probs, _ = gating_layer.get_choices(g_logits, q_logits=q_logits)
        self.assertTrue((choices[0].numpy() == np.array([0, 0, 1])).all())
        g_logits = torch.stack([torch.tensor(a) for a in [[100., -100., 100.], [-1., 0., 1.]]])
        choices, probs, _ = gating_layer.get_choices(g_logits, q_logits=None)
        self.assertTrue((choices[0].numpy() == np.array([1, 0, 1])).all())

        config = {
            "q": None,
            "g": None,
            "w": None,
            "reparam": True,
            "discrete": True
        }

        self.convs = nn.ModuleList([nn.Sequential(
            nn.Conv2d(3, 3, 3),
            nn.ReLU(),
        ) for _ in range(3)])

        gating_layer = VILayer(self.convs, config)

        g_logits = torch.stack([torch.tensor(a) for a in [[100., -100., 100.], [-1., 0., 1.]]])
        q_logits = torch.stack([torch.tensor(a) for a in [[-100., -100., 5.], [-1., 0., 1.]]])
        choices, probs, _ = gating_layer.get_choices(g_logits, q_logits=q_logits)
        self.assertTrue((choices[0].numpy() == np.array([0, 0, 1])).all())
        g_logits = torch.stack([torch.tensor(a) for a in [[100., -100., 100.], [-1., 0., 1.]]])
        choices, probs, _ = gating_layer.get_choices(g_logits, q_logits=None)
        self.assertTrue((choices[0].numpy() == np.array([1, 0, 1])).all())

        config = {
            "q": None,
            "g": None,
            "w": None,
            "reparam": False,
            "discrete": False
        }

        self.convs = nn.ModuleList([nn.Sequential(
            nn.Conv2d(3, 3, 3),
            nn.ReLU(),
        ) for _ in range(3)])

        gating_layer = VILayer(self.convs, config)

        q = [[-5., -5., 5.], [-1., 0., 1.]]
        p = [[100., -100., 5.], [-1., 0., 1.]]
        g_logits = torch.stack([torch.tensor(a) for a in p])
        q_logits = torch.stack([torch.tensor(a) for a in q])
        choices, q_probs, p_probs = gating_layer.get_choices(g_logits, q_logits=q_logits)
        self.assertTrue((choices[0].numpy() == np.array([0, 0, 1])).all())
        soln = np.log(1 / (1 + np.exp(-np.array(q))))
        self.assertTrue(np.allclose(q_probs[0], np.array(
            [np.log(1 - np.exp(soln[0][0])), np.log(1 - np.exp(soln[0][1])), soln[0][2]])))
        soln = np.log(1 / (1 + np.exp(-np.array(p))))
        self.assertTrue(np.allclose(p_probs[0], np.array([soln[0][1], np.log(1 - np.exp(soln[0][1])), soln[0][2]])))

        choices, q_probs, p_probs = gating_layer.get_choices(g_logits)
        self.assertTrue((choices[0].numpy() == np.array([1, 0, 1])).all())
        self.assertTrue(q_probs == 0)
        self.assertTrue(p_probs == 0)


if __name__ == '__main__':
    unittest.main()
