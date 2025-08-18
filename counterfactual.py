import numpy as np
import torch
from torch.autograd import Function
from stable_baselines3 import DQN
from tqdm import tqdm_notebook as tqdm

def calc_reward(obs_samps, discount=0.9):
    # Column 0 is a time index, column 6 is the reward
    discounted_reward = (discount**obs_samps[..., 0] * obs_samps[..., 6])
    return discounted_reward.sum(axis=-1)  # Take the last axis

def eval_on_policy(obs_samps, discount=0.9, bootstrap=False, n_bootstrap=None):
    obs_rewards = calc_reward(obs_samps, discount).squeeze()  # 1D array
    assert obs_rewards.ndim == 1

    if bootstrap:
        assert n_bootstrap is not None, "Please specify n_bootstrap"
        bs_rewards = np.random.choice(
            obs_rewards,
            size=(n_bootstrap, obs_rewards.shape[0]),
            replace=True)
        return bs_rewards.mean(axis=1)
    else:
        return obs_rewards.mean()

def counterfactual(env, policy_kwargs, trajectories, discount=0.95, device='cpu'):
    class CounterfactualFn(torch.autograd.Function):
        @staticmethod
        def forward(ctx, model_parameters):
            with torch.enable_grad():
                model = DQN("MlpPolicy", env, policy_kwargs=policy_kwargs, verbose=0, gamma=discount, buffer_size=0, device=device)
                model.policy.load_from_vector(model_parameters)
                model.policy.train()

                # Generate CF trajectories using env's method
                cf_trajectories = env.get_cf_trajectory(trajectories, model)

                # Compute cf_eval
                cf_rewards = []
                for cf_traj_list in cf_trajectories:
                    cf_samps = np.array(cf_traj_list)  # Convert list to array if needed
                    cf_reward = eval_on_policy(cf_samps, discount=discount, bootstrap=False)
                    cf_rewards.append(cf_reward)
                cf_eval = np.mean(cf_rewards)

                # Compute gradient for cf_eval
                model.policy.optimizer.zero_grad()
                cf_eval_tensor = torch.tensor(cf_eval, requires_grad=True)  # Make tensor for backward
                cf_eval_tensor.backward()
                cf_gradient = [param.grad.flatten() for param in model.policy.q_net.parameters()] # consider the gradient of q_net only
                cf_gradient = torch.cat(cf_gradient).detach()
                cf_gradient = torch.cat([cf_gradient, torch.zeros_like(cf_gradient)])

                # print('CF gradient sum', torch.sum(cf_gradient))

                cf_gradient[cf_gradient.isnan()] = 0
                model.policy.optimizer.zero_grad()

            ctx.save_for_backward(cf_gradient)
            for param in model.policy.parameters():
                del param.grad

            return torch.tensor(cf_eval).detach()

        def backward(ctx, dl_dcf):
            cf_gradient, = ctx.saved_tensors
            return dl_dcf * cf_gradient

    return CounterfactualFn.apply
