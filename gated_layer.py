import torch
from torch import nn
from torch.nn import functional


class GatedLayer(nn.Module):
    def __init__(self, blocks, config, task_count, out_shape=None, cuda=None):
        super().__init__()
        self.blocks = blocks
        self.sparse = config["sparse"]
        self.task_count = task_count
        self.cuda = cuda
        self.q_net = None
        self.out_shape = out_shape
        self.weights = nn.Parameter(torch.randn([task_count, len(blocks)]))  # T X B
        # self.q_logits = nn.Parameter(torch.zeros([len(self.tasks), self.options, self.blocks]))

    def get_weight(self, emb):
        # emb: N x 1 for tasks
        # out: N x B
        N = emb.shape[0]
        B = self.weights.shape[1]
        selected = torch.gather(self.weights[None, :, :].expand(N, -1, -1), 1,
                                emb[:, :, None].expand(-1, -1, B))  # N x 1 x B
        return selected.reshape(N, -1)

    def get_g_logits(self, emb):
        # emb: N x 1 for tasks
        # out: N x B
        N = emb.shape[0]
        B = self.g_logits.shape[1]

        selected = torch.gather(self.g_logits[None, :, :].expand(N, -1, -1), 1,
                                emb[:, :, None].expand(-1, -1, B))  # N x 1 x B
        return selected.reshape(N, -1)

    def get_q_logits(self, x, emb, task_labels):
        # emb: N x 1 for tasks
        # task_labels: N x 1
        # out: N x B
        N = emb.shape[0]

        one_hot_emb = torch.zeros([N, self.task_count])
        one_hot_lab = torch.zeros([N, self.task_count])

        if self.cuda:
            one_hot_emb = one_hot_emb.cuda(self.cuda)
            one_hot_lab = one_hot_lab.cuda(self.cuda)

        one_hot_emb = one_hot_emb.scatter_(1, emb, 1)
        one_hot_lab = one_hot_lab.scatter_(1, task_labels, 1)

        concat = torch.cat((one_hot_emb, one_hot_lab), 1)

        return self.q_net(x, concat)

    def forward(self, iput, emb, log_probs=0., extra_loss=0., labels=None):

        """
        :param iput: N x I
        :param emb: N x E
        :param log_probs: N
        :param extra_loss: N
        :param labels: N
        :return: N x O, N, N
        """
        N = iput.shape[0]

        g_logits = self.get_g_logits(emb)  # N x B
        w = self.get_weight(emb)  # N x B
        if self.q_net is not None and self.training:
            q_logits = self.get_q_logits(iput, emb, labels)
        else:
            q_logits = None

        b, new_log_probs, new_extra_loss = self.get_choices(g_logits, q_logits=q_logits)  # N x B, N, N
        if self.sparse:
            chosen = b.sum(dim=0)
            zeros = torch.zeros([N, self.out_shape])
            if self.cuda:
                zeros = zeros.cuda(self.cuda)

            output = torch.stack([self.blocks[b](iput) if chosen[b] > 0 else zeros for b in range(len(chosen))], dim=1)  # N x B x O
        else:
            output = torch.stack([block(iput) for block in self.blocks], dim=1)  # N x B x O
        output = output * b[:, :, None]
        extra_loss += new_extra_loss

        if isinstance(new_extra_loss, torch.Tensor):
            extra_loss += new_extra_loss.detach() * log_probs

        return output.sum(dim=1), log_probs + new_log_probs, extra_loss

    def get_choices(self, g_logits, q_logits):
        raise NotImplementedError


class BlendingLayer(GatedLayer):
    def __init__(self, blocks, config, task_count, cuda=None):
        super().__init__(blocks, config, task_count, cuda=cuda)
        self.g_logits = nn.Parameter(torch.randn([task_count, len(blocks)]))

    def get_choices(self, g_logits, q_logits=None, cuda=None):
        return torch.sigmoid(g_logits), 0, 0


class SamplingLayer(GatedLayer):
    def __init__(self, blocks, config, task_count, out_shape, cuda=None):
        super().__init__(blocks, config, task_count, out_shape=out_shape, cuda=cuda)
        self.reparam = config["reparam"]
        self.discrete = config["discrete"]
        self.temp = 0.1
        self.reg = config["reg"]
        self.num_blocks = config["blocks"]
        self.avg_k = config["avg_k"]
        self.pick_one = config["pick_one"]
        self.g_logits = nn.Parameter(torch.zeros([task_count, len(blocks)]))

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
                log_p_prob = functional.logsigmoid(g_logits) * r + functional.logsigmoid(-g_logits) * (1 - r)
                return r, log_p_prob.sum(dim=1), sample_penalty
        else:
            sample_penalty = (torch.sigmoid(g_logits) * self.reg).sum(dim=1)

            expected_num = torch.sigmoid(g_logits).mean(dim=1)

            if self.avg_k:
                k_vals = torch.ones([expected_num.shape[0]]).fill_(torch.sigmoid(self.g_logits).mean(dim=1).mean(dim=0) * self.num_blocks)
            else:
                k_vals = torch.round(expected_num * self.num_blocks) # B

            res = torch.zeros(g_logits.shape[0], g_logits.shape[1])
            if self.cuda:
                res = res.cuda(self.cuda)

            for i in range(len(k_vals)):
                _, indices = torch.topk(g_logits[i], int(k_vals[i]))
                res[i] = res[i].scatter(0, indices, 1)

            return res, 0, sample_penalty


class VILayer(GatedLayer):
    def __init__(self, blocks, config, task_count, q_net, out_shape, cuda=None):
        super().__init__(blocks, config, task_count, out_shape=out_shape, cuda=cuda)
        self.reparam = config["reparam"]
        self.discrete = config["discrete"]
        self.temp = 0.1
        self.q_net = q_net
        self.reg = config["reg"]
        self.g_logits = nn.Parameter(torch.zeros([task_count, len(blocks)]))

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
            distr = torch.distributions.Bernoulli(logits=g_logits)
            r = distr.sample()
            return r, 0, 0
