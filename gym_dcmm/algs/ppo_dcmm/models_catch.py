import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, units, input_size):
        super(MLP, self).__init__()
        layers = []
        for output_size in units:
            layers.append(nn.Linear(input_size, output_size))
            layers.append(nn.ELU())
            input_size = output_size
        self.mlp = nn.Sequential(*layers)

        # orthogonal init of weights
        # hidden layers scale np.sqrt(2)
        self.init_weights(self.mlp, [np.sqrt(2)] * len(units))

    def forward(self, x):
        return self.mlp(x)

    @staticmethod
    def init_weights(sequential, scales):
        [torch.nn.init.orthogonal_(module.weight, gain=scales[idx]) for idx, module in
         enumerate(mod for mod in sequential if isinstance(mod, nn.Linear))]


class ActorCritic(nn.Module):
    def __init__(self, kwargs):
        nn.Module.__init__(self)
        separate_value_mlp = kwargs.pop('separate_value_mlp')
        self.separate_value_mlp = separate_value_mlp

        actions_num = kwargs.pop('actions_num')
        input_shape = kwargs.pop('input_shape')
        self.units = kwargs.pop('actor_units')
        mlp_input_shape = input_shape[0]

        out_size = self.units[-1]
        print("mlp_input_shape: ", mlp_input_shape)
        self.actor_mlp_t = MLP(units=self.units, input_size=mlp_input_shape-12)
        self.actor_mlp_c = MLP(units=self.units, input_size=mlp_input_shape-2)
        # if self.separate_value_mlp:
        self.value_mlp = MLP(units=self.units, input_size=mlp_input_shape)
        self.value = torch.nn.Linear(out_size, 1)
        self.mu_t = torch.nn.Linear(out_size, actions_num-12)
        self.mu_c = torch.nn.Linear(out_size, actions_num-6)
        self.sigma_t = nn.Parameter(
            torch.zeros(actions_num-12, requires_grad=True, dtype=torch.float32), requires_grad=True)
        self.sigma_c = nn.Parameter(
            torch.zeros(actions_num-6, requires_grad=True, dtype=torch.float32), requires_grad=True)

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d):
                fan_out = m.kernel_size[0] * m.out_channels
                m.weight.data.normal_(mean=0.0, std=np.sqrt(2.0 / fan_out))
                if getattr(m, 'bias', None) is not None:
                    torch.nn.init.zeros_(m.bias)
            if isinstance(m, nn.Linear):
                if getattr(m, 'bias', None) is not None:
                    torch.nn.init.zeros_(m.bias)
        nn.init.constant_(self.sigma_t, 0)
        nn.init.constant_(self.sigma_c, 0)
        # policy output layer with scale 0.01
        # value output layer with scale 1
        torch.nn.init.orthogonal_(self.mu_t.weight, gain=0.01)
        torch.nn.init.orthogonal_(self.mu_c.weight, gain=0.01)
        torch.nn.init.orthogonal_(self.value.weight, gain=1.0)

    @torch.no_grad()
    def act(self, obs_dict):
        # used specifically to collection samples during training
        # it contains exploration so needs to sample from distribution
        mu, logstd, value = self._actor_critic(obs_dict)
        sigma = torch.exp(logstd)
        distr = torch.distributions.Normal(mu, sigma)
        selected_action = distr.sample()
        result = {
            'neglogpacs': -distr.log_prob(selected_action).sum(1),
            'values': value,
            'actions': selected_action,
            'mus': mu,
            'sigmas': sigma,
        }
        return result

    @torch.no_grad()
    def act_inference(self, obs_dict):
        # used for testing
        mu, logstd, value = self._actor_critic(obs_dict)
        return mu

    def _actor_critic(self, obs_dict):
        obs = obs_dict['obs']
        obs_t = obs_dict['obs_t']
        obs_c = obs_dict['obs_c']

        x_t = self.actor_mlp_t(obs_t)
        x_c = self.actor_mlp_c(obs_c)
        mu_t = self.mu_t(x_t)
        mu_c = self.mu_c(x_c)
        # if self.separate_value_mlp:
        x = self.value_mlp(obs)
        value = self.value(x)

        sigma_t = self.sigma_t
        sigma_c = self.sigma_c
        # Normalize to (-1,1)
        mu_t = torch.tanh(mu_t)
        mu_c = torch.tanh(mu_c)

        # Concatenate mu_t and mu_c
        mu = torch.cat((mu_t, mu_c), dim=1)

        return mu, torch.cat((mu_t * 0 + sigma_t, mu_c * 0 + sigma_c), dim=1), value

    def forward(self, input_dict):
        prev_actions = input_dict.get('prev_actions', None)
        mu, logstd, value = self._actor_critic(input_dict)
        sigma = torch.exp(logstd)
        distr = torch.distributions.Normal(mu, sigma)
        entropy = distr.entropy().sum(dim=-1)
        prev_neglogp = -distr.log_prob(prev_actions).sum(1)
        result = {
            'prev_neglogp': torch.squeeze(prev_neglogp),
            'values': value,
            'entropy': entropy,
            'mus': mu,
            'sigmas': sigma,
        }
        return result