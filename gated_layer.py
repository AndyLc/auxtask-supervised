import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import unittest
import modules


class GatedLayer(nn.Module):
    def __init__(self, blocks, config):
        super().__init__()
        self.blocks = blocks
        self.g_logits = config["g"]#modules.catalog(config["g"])
        self.q_logits = config["q"]#modules.catalog(config["q"])
        self.w = config["w"]#modules.catalog(config["w"])

    def forward(self, iput, emb, labels=None):

        """
        :param iput: N x I
        :param emb: N x E
        :param labels: N x 1
        :return: N x O, T X B,
        """

        g_logits = self.g_logits(emb) # N x B
        w = self.w(emb) # N x B
        if labels is not None:
            q = self.q_logits(emb, labels)
        else:
            q = None

        b, log_probs, extra_loss = self.get_choices(g_logits, q_logits=q) # N x B
        output = torch.stack([block(iput) for block in self.blocks], dim=1) #N x B x O
        output = output * w[:, :, None] * b[:, :, None]
        return output.sum(dim=1), log_probs, extra_loss

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
            if self.reparam:
                if not self.discrete:
                    distr = torch.distributions.RelaxedBernoulli(self.temp, logits=g_logits)
                    r = distr.rsample()
                    sample_penalty = (torch.sigmoid(g_logits) * self.reg).sum(dim=1)
                    return r, 0, sample_penalty
                else:
                    distr = torch.distributions.RelaxedBernoulli(self.temp, logits=g_logits)
                    r = distr.rsample()
                    sample_penalty = (torch.sigmoid(g_logits) * self.reg).sum(dim=1)
                    discr = torch.round(r)
                    return r + (discr - r).detach(), 0, sample_penalty
            else:
                distr = torch.distributions.Bernoulli(logits=g_logits)
                r = distr.sample()
                sample_penalty = (torch.sigmoid(g_logits) * self.reg).sum(dim=1)

                return r, (F.logsigmoid(g_logits) * r + F.logsigmoid(-g_logits) * (1-r)).sum(dim=1), sample_penalty
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
            if self.reparam:
                if not self.discrete:
                    distr = torch.distributions.RelaxedBernoulli(self.temp, logits=q_logits)
                    r = distr.rsample()

                    sample_penalty = (torch.sigmoid(q_logits) * self.reg).sum(dim=1)

                    prob_one = torch.sigmoid(q_logits) * (F.logsigmoid(q_logits) - F.logsigmoid(g_logits))
                    prob_zero = (1 - torch.sigmoid(q_logits)) * (F.logsigmoid(-q_logits) - F.logsigmoid(-g_logits))
                    KL = prob_one + prob_zero
                    return r, 0, (KL).sum(1) + sample_penalty
                else:
                    distr = torch.distributions.RelaxedBernoulli(self.temp, logits=q_logits)
                    r = distr.rsample()
                    discr = torch.round(r)

                    sample_penalty = (torch.sigmoid(q_logits) * self.reg).sum(dim=1)

                    prob_one = torch.sigmoid(q_logits) * (F.logsigmoid(q_logits) - F.logsigmoid(g_logits))
                    prob_zero = (1 - torch.sigmoid(q_logits)) * (F.logsigmoid(-q_logits) - F.logsigmoid(-g_logits))
                    KL = prob_one + prob_zero
                    return r + (discr - r).detach(), 0, (KL).sum(1) + sample_penalty
            else:
                distr = torch.distributions.Bernoulli(logits=q_logits) # N x B
                r = distr.sample()
                log_q_prob = F.logsigmoid(q_logits) * r + F.logsigmoid(-q_logits) * (1 - r)
                log_p_prob = F.logsigmoid(g_logits) * r + F.logsigmoid(-g_logits) * (1 - r)

                sample_penalty = (torch.sigmoid(q_logits) * self.reg).sum(dim=1)

                return r, log_q_prob.sum(dim=1), ((-log_p_prob + log_q_prob).detach() * log_q_prob - log_p_prob).sum(dim=1) + sample_penalty
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
        soln = 1/(1+np.exp(-np.array(arr)))
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
        self.assertTrue(np.allclose(q_probs[0], np.array([np.log(1 - np.exp(soln[0][0])), np.log(1 - np.exp(soln[0][1])), soln[0][2]])))
        soln = np.log(1 / (1 + np.exp(-np.array(p))))
        self.assertTrue(np.allclose(p_probs[0], np.array([soln[0][1], np.log(1 - np.exp(soln[0][1])), soln[0][2]])))

        choices, q_probs, p_probs = gating_layer.get_choices(g_logits)
        self.assertTrue((choices[0].numpy() == np.array([1, 0, 1])).all())
        self.assertTrue(q_probs == 0)
        self.assertTrue(p_probs == 0)


if __name__ == '__main__':
    unittest.main()