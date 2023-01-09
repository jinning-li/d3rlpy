"""The Cross Entropy Method Adopted from `https://github.com/LemonPi/pytorch_cem`.
"""
import torch
import time
import logging
from torch.distributions.multivariate_normal import MultivariateNormal

logger = logging.getLogger(__name__)


def pytorch_cov(x, rowvar=False, bias=False, ddof=None, aweights=None):
    """Estimates covariance matrix like numpy.cov"""
    # ensure at least 2D
    if x.dim() == 1:
        x = x.view(-1, 1)

    # treat each column as a data point, each row as a variable
    if rowvar and x.shape[0] != 1:
        x = x.t()

    if ddof is None:
        if bias == 0:
            ddof = 1
        else:
            ddof = 0

    w = aweights
    if w is not None:
        if not torch.is_tensor(w):
            w = torch.tensor(w, dtype=torch.float)
        w_sum = torch.sum(w)
        avg = torch.sum(x * (w / w_sum)[:, None], 0)
    else:
        avg = torch.mean(x, 0)

    # Determine the normalization
    if w is None:
        fact = x.shape[0] - ddof
    elif ddof == 0:
        fact = w_sum
    elif aweights is None:
        fact = w_sum - ddof
    else:
        fact = w_sum - ddof * torch.sum(w * w) / w_sum

    xm = x.sub(avg.expand_as(x))

    if w is None:
        X_T = xm.t()
    else:
        X_T = torch.mm(torch.diag(w), xm).t()

    c = torch.mm(X_T, xm)
    c = c / fact

    return c.squeeze()

def batch_cov(points):
    """Compute covariance matrix with batch dim.

    Args:
        points: shape(batch_size, N_samples, data_dim)
    Returns:
        bcov: shape(batch_size, data_dim, data_dim)
    """
    B, N, D = points.size()
    mean = points.mean(dim=1).unsqueeze(1)
    diffs = (points - mean).reshape(B * N, D)
    prods = torch.bmm(diffs.unsqueeze(2), diffs.unsqueeze(1)).reshape(B, N, D, D)
    bcov = prods.sum(dim=1) / (N - 1)  # Unbiased estimate
    return bcov  # (B, D, D)

class CEM:
    """
    Cross Entropy Method control

    This implementation batch samples the trajectories and so scales well with the 
    number of samples K.
    """
    def __init__(
        self, 
        dynamics, 
        running_cost, 
        nx, 
        nu, 
        num_samples=100, 
        num_iterations=3, 
        num_elite=10, 
        horizon=15,
        device="cpu",
        terminal_state_cost=None,
        u_min=None,
        u_max=None,
        choose_best=False,
        init_cov_diag=1
    ):
        """
        :param dynamics: function(state, action) -> next_state (K x nx) 
            taking in batch state (K x nx) and action (K x nu)
        :param running_cost: function(state, action) -> cost (K x 1) 
            taking in batch state and action (same as dynamics)
        :param nx: state dimension
        :param num_samples: K, number of trajectories to sample
        :param horizon: T, length of each trajectory
        :param device: pytorch device
        :param terminal_state_cost: function(state) -> cost (K x 1) taking in batch state
        """
        self.d = device
        self.dtype = torch.double  # TODO determine dtype
        self.K = num_samples  # N_SAMPLES
        self.T = horizon  # TIMESTEPS
        self.M = num_iterations
        self.num_elite = num_elite
        self.choose_best = choose_best

        # dimensions of state and control
        self.nx = nx
        self.nu = nu
        self.batch_size = None

        self.mean = None
        self.cov = None

        self.F = dynamics
        self.running_cost = running_cost
        self.terminal_state_cost = terminal_state_cost
        self.init_cov_diag = init_cov_diag
        self.u_min = u_min
        self.u_max = u_max
        # make sure if any of them is specified, both are specified
        if self.u_max is not None and self.u_min is None:
            self.u_min = -self.u_max
        if self.u_min is not None and self.u_max is None:
            self.u_max = -self.u_min
        if self.u_min is not None:
            self.u_min = self.u_min.to(device=self.d)
            self.u_max = self.u_max.to(device=self.d)
        self.action_distribution = None

        # regularize covariance
        self.cov_reg = torch.eye(self.T * self.nu, 
            device=self.d, dtype=self.dtype) * init_cov_diag * 1e-5

    def reset(self):
        """
        Clear controller state after finishing a trial
        """
        # action distribution, initialized as N(0,I)
        # we do Hp x 1 instead of H x p because covariance will be Hp x Hp 
        # matrix instead of some higher dim tensor
        self.mean = torch.zeros((self.batch_size, self.T * self.nu), 
            device=self.d, dtype=self.dtype)
        self.cov = self.init_cov_diag * torch.eye(self.T * self.nu, 
            device=self.d, dtype=self.dtype).repeat(self.batch_size, 1, 1)

    def _bound_samples(self, samples):
        if self.u_max is not None:
            samples = torch.clamp(samples, min=self.u_min, max=self.u_max)


            # for t in range(self.T):
            #     u = samples[:, self._slice_control(t)]
            #     cu = torch.max(torch.min(u, self.u_max), self.u_min)
            #     samples[:, self._slice_control(t)] = cu
        return samples

    def _slice_control(self, t):
        return slice(t * self.nu, (t + 1) * self.nu)

    def flatten_batch_dim(self, tensor):
        return torch.flatten(tensor, end_dim=1)

    def recover_batch_dim(self, tensor):
        return tensor.view((self.batch_size, self.K) + tensor.shape[1:])

    def _evaluate_trajectories(self, samples, init_state):
        if self.T == 1:
            cost_total = torch.squeeze(self.running_cost(init_state, samples))
        else:
            cost_total = torch.zeros(self.K, device=self.d, dtype=self.dtype)
            for t in range(self.T):
                u = samples[:, self._slice_control(t)]
                state = self.F(state, u)
                cost_total += self.running_cost(state, u)
        if self.terminal_state_cost:
            cost_total += self.terminal_state_cost(state)
        return cost_total

    def _sample_top_trajectories(self, state, num_elite):
        # sample K action trajectories
        # in case it's singular
        self.action_distribution = MultivariateNormal(self.mean, covariance_matrix=self.cov)
        samples = self.action_distribution.sample((self.K,))
        samples = samples.transpose(0, 1)  # shape = (batch_size, K, ac_dim)
        # bound to control maximums
        samples = self._bound_samples(samples)

        init_state = state.unsqueeze(1).repeat(1, self.K, 1)  # shape = (batch_size, K, ac_dim)
        init_state = self.flatten_batch_dim(init_state)
        samples = self.flatten_batch_dim(samples)

        # cost_total must be 1 dim
        cost_total = self._evaluate_trajectories(samples, init_state)   
        # select top k based on score
        cost_total = self.recover_batch_dim(cost_total)
        top_costs, topk = torch.topk(cost_total, num_elite, dim=1, largest=False, sorted=False)
        top_samples = samples[topk]
        return top_samples

    def command(self, state, choose_best=False):
        if not torch.is_tensor(state):
            state = torch.tensor(state)
        state = state.to(dtype=self.dtype, device=self.d)

        if len(state.shape) == 1:
            state = state.unsqueeze(0)
        assert len(state.shape) == 2  # state.shape = (batch_size, K, obs_dim)
        self.batch_size = state.shape[0]

        self.reset()

        for m in range(self.M):
            top_samples = self._sample_top_trajectories(state, self.num_elite)
            # fit the gaussian to those samples
            self.mean = torch.mean(top_samples, dim=1)  
            # self.cov = pytorch_cov(top_samples, rowvar=False)  
            self.cov = batch_cov(top_samples)  

            cov_rk = torch.matrix_rank(self.cov) < self.T * self.nu
            cov_rk = cov_rk.view(-1, 1, 1)
            self.cov += self.cov_reg.reshape(1, self.T * self.nu, 
                self.T * self.nu) * cov_rk

        if choose_best and self.choose_best:
            top_sample = self._sample_top_trajectories(state, 1)
        else:
            top_sample = self.action_distribution.sample((1,))

        # only apply the first action from this trajectory
        # u = top_sample[0, self._slice_control(0)] # TODO: add traj
        u = top_sample[0]

        return u


def run_cem(
    cem, 
    env, 
    retrain_dynamics, 
    retrain_after_iter=50, 
    iter=1000, 
    render=True, 
    choose_best=False
):
    dataset = torch.zeros((retrain_after_iter, cem.nx + cem.nu), dtype=cem.dtype, device=cem.d)
    total_reward = 0
    for i in range(iter):
        state = env.state.copy()
        command_start = time.perf_counter()
        action = cem.command(state, choose_best=choose_best)
        elapsed = time.perf_counter() - command_start
        s, r, _, _ = env.step(action.cpu().numpy())
        total_reward += r
        logger.debug("action taken: %.4f cost received: %.4f time taken: %.5fs", 
            action, -r, elapsed)
        if render:
            env.render()

        di = i % retrain_after_iter
        if di == 0 and i > 0:
            retrain_dynamics(dataset)
            # don't have to clear dataset since it'll be overridden, but useful for debugging
            dataset.zero_()
        dataset[di, :cem.nx] = torch.tensor(state, dtype=cem.dtype)
        dataset[di, cem.nx:] = action
    return total_reward, 