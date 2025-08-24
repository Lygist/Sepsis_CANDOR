import torch
from torch import nn
from torch.autograd import Function
from torchrl.collectors import SyncDataCollector
from torchrl.modules import MLP, QValueActor
from torchrl.data.replay_buffers import ReplayBuffer, LazyTensorStorage
from stable_baselines3 import DQN
from diffdqn_transition import compute_policy_hessian_inverse_vector_product
from torchrl.envs import GymWrapper, TransformedEnv, DTypeCastTransform
from torchrl.data import CompositeSpec
from torchrl.data.tensor_specs import MultiDiscreteTensorSpec, DiscreteTensorSpec

import gym
import copy

# ============================= Environment Wrappers =================================
class CorrectedGymWrapper(GymWrapper):
    def __init__(self, env, *args, **kwargs):
        super().__init__(env, *args, **kwargs)
        NUM_PATIENTS = env.observation_space.shape[0]
        self.observation_spec = CompositeSpec(
            observation=MultiDiscreteTensorSpec([324] * NUM_PATIENTS)
        )

class SepsisWorldAdapter(gym.Env):
    def __init__(self, env):
        self.env = env

    def __getattr__(self, name):
        return getattr(self.env, name)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        return obs, reward, done, False, info

    def reset(self):
        obs = self.env.reset()
        return obs, {}

# ============================= DiffCQL =================================
class CQL(DQN):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def train(self, gradient_steps: int, batch_size: int = 100) -> None:
        # Update learning rate according to scheduler
        self._update_learning_rate(self.policy.optimizer)

        losses = []
        for _ in range(gradient_steps):
            # Sample replay buffer
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)  # type: ignore[union-attr]

            with torch.no_grad():
                next_q_values = self.q_net_target(replay_data.next_observations)[0]  # Take [0] if tuple
                next_q_values, _ = next_q_values.max(dim=1)
                next_q_values = next_q_values.reshape(-1, 1)
                # 1-step TD target
                target_q_values = replay_data.rewards + (1 - replay_data.dones) * self.gamma * next_q_values

            current_q_values = self.q_net(replay_data.observations)[0]  # Take [0] if tuple
            chosen_q_values = torch.gather(current_q_values, dim=1, index=replay_data.actions.long())

            # Huber loss for Q
            loss_qvalue = nn.functional.smooth_l1_loss(chosen_q_values, target_q_values)

            # CQL loss
            logsumexp_q = current_q_values.logsumexp(dim=1, keepdim=True)
            loss_cql = (logsumexp_q - chosen_q_values).mean()

            # Total loss
            loss = loss_qvalue + loss_cql

            losses.append(loss.item())


            self.policy.optimizer.zero_grad()
            loss.backward()
            # Clip gradient norm
            nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.policy.optimizer.step()

        # Increase update counter
        self._current_progress_remaining = 1.0 - float(self._n_updates) / float(self.train_freq.frequency)

        # Update target network
        if self._n_updates % self.target_update_interval == 0:
            # Do a complete copy
            self.q_net_target.load_state_dict(self.q_net.state_dict())

def DiffCQL(
    env_wrapper,
    policy_kwargs,
    learning_rate=0.001,
    learning_starts=20000,
    min_num_iters=100000,
    model_initial_parameters=None,
    discount=0.99,
    target_update_interval=1000,
    buffer_size=100000,
    device="cpu",
    verbose=0,
    load_replay_buffer=False,
    save_replay_buffer=False,
    replay_buffer_dict=None,
    data_id=None,
    baseline=0,
    backprop_method=1,
    seed=0,
):
    class DiffCQLFn(Function):
        @staticmethod
        def forward(ctx, env_parameter): # reward
            env = env_wrapper(env_parameter.detach())

            # CQL based on SB3 DQN
            model = CQL("MlpPolicy", env, learning_rate=learning_rate, learning_starts=learning_starts, policy_kwargs=policy_kwargs, verbose=verbose, gamma=discount, strict=True,
                    target_update_interval=target_update_interval,
                    buffer_size=buffer_size,
                    optimize_memory_usage=False,
                    seed=seed,
                    )

            if model_initial_parameters is not None:
                model.policy.load_from_vector(model_initial_parameters)

            if load_replay_buffer:
                try:
                    model.replay_buffer = replay_buffer_dict[data_id]
                except:
                    print('Failed to load the replay buffer...')

            with torch.enable_grad():
                model.learn(total_timesteps=min_num_iters, log_interval=50)

            model_parameters = torch.from_numpy(model.policy.parameters_to_vector()) # pytorch autograd doesn't support non-tensor output so we have to create one :(
            if save_replay_buffer:
                replay_buffer_dict[data_id] = model.replay_buffer

            ctx.save_for_backward(env_parameter.detach(), model_parameters.detach())
            return model_parameters

        @staticmethod
        def backward(ctx, dl_dmodel):
            env_parameter, model_parameters = ctx.saved_tensors
            env = env_wrapper(env_parameter.detach())

            model = CQL("MlpPolicy", env, policy_kwargs=policy_kwargs, verbose=0, gamma=discount, device=device, buffer_size=0)
            model.policy.load_from_vector(model_parameters.detach()) # loading model parameters

            with torch.enable_grad():
                env_parameter_var = torch.autograd.Variable(env_parameter, requires_grad=True)
                env_var = env_wrapper(env_parameter_var)

                dl_denv = compute_policy_hessian_inverse_vector_product(-dl_dmodel, env_parameter_var, env_var, model, num_episodes=100, discount=discount, baseline=baseline, backprop_method=backprop_method)

            # clearning model parameter gradient
            for param in model.policy.parameters():
                del param.grad
            del env_parameter_var.grad

            return dl_denv
    return DiffCQLFn.apply

# ============================= DiffIQL =================================
def expectile_loss(diff, tau=0.7):
    weight = torch.abs(tau - (diff < 0).float())
    return (weight * (diff ** 2)).mean()

# def expectile_loss(diff, tau=0.7):
#     indicator = torch.sigmoid(-diff * 50)  # Soft approximation for (diff < 0), 50 for sharpness
#     weight = tau * (1 - indicator) + (1 - tau) * indicator
#     return (weight * (diff ** 2)).mean()

def DiffIQL(
    env_wrapper,
    policy_kwargs,
    learning_rate=0.001,
    learning_starts=20000,
    min_num_iters=100000,
    model_initial_parameters=None,
    discount=0.99,
    target_update_interval=1000,
    buffer_size=100000,
    device="cpu",
    verbose=0,
    load_replay_buffer=False,
    save_replay_buffer=False,
    replay_buffer_dict=None,
    data_id=None,
    baseline=0,
    backprop_method=1,
    seed=0,
    tau=0.7,
    beta=3.0,
):
    class DiffIQLFn(Function):
        @staticmethod
        def forward(ctx, env_parameter):
            def create_env_fn():
                sepsis_env = env_wrapper(env_parameter.detach())
                adapted_env = SepsisWorldAdapter(sepsis_env)
                wrapped_env = CorrectedGymWrapper(adapted_env)
                cast_transform = DTypeCastTransform(
                    dtype_in=torch.int64,
                    dtype_out=torch.float32,
                    in_keys=["observation"]
                )
                transformed_env = TransformedEnv(wrapped_env, cast_transform)
                return transformed_env

            env = GymWrapper(env_wrapper(env_parameter.detach()))

            obs_dim = env.observation_space.shape[0]
            action_dim = env.action_space.n

            # Networks for IQL
            q_net = MLP(
                in_features=obs_dim + action_dim,
                out_features=1,
                depth=2,
                num_cells=policy_kwargs.get("net_arch", [64])[0],
            ).to(device)

            v_net = MLP(
                in_features=obs_dim,
                out_features=1,
                depth=2,
                num_cells=policy_kwargs.get("net_arch", [64])[0],
            ).to(device)

            actor = MLP(
                in_features=obs_dim,
                out_features=action_dim,
                depth=2,
                num_cells=policy_kwargs.get("net_arch", [64])[0],
            ).to(device)

            # Warm start
            if model_initial_parameters is not None:
                actor.load_state_dict(model_initial_parameters)

            # Optimizers (only update actor in backward)
            optimizer_q = torch.optim.Adam(q_net.parameters(), lr=learning_rate)
            optimizer_v = torch.optim.Adam(v_net.parameters(), lr=learning_rate)
            optimizer_actor = torch.optim.Adam(actor.parameters(), lr=learning_rate)

            # Replay buffer
            if load_replay_buffer and replay_buffer_dict is not None and data_id in replay_buffer_dict:
                rb = replay_buffer_dict[data_id]
            else:
                rb = ReplayBuffer(storage=LazyTensorStorage(max_size=buffer_size))

            policy_actor = QValueActor(
                module=actor,
                in_keys=["observation"],
                spec=DiscreteTensorSpec(n=action_dim)
            ).to(device)

            collector = SyncDataCollector(
                create_env_fn=create_env_fn,
                policy=policy_actor,
                frames_per_batch=64,
                total_frames=min_num_iters,
                init_random_frames=learning_starts,
                reset_at_each_iter=True,
                device=device,
            )

            total_frames = 0
            for i, data in enumerate(collector):
                rb.extend(data)
                if total_frames < learning_starts:
                    total_frames += data.numel()
                    continue
                for _ in range(4):
                    batch = rb.sample(64)
                    observation = batch["observation"]
                    action = batch["action"].float()
                    reward = batch["next"]["reward"]
                    done = batch["next"]["done"]
                    next_observation = batch["next"]["observation"]

                    # Q loss
                    with torch.enable_grad():
                        sa = torch.cat([observation, action], dim=1)
                        q_values = q_net(sa)
                        with torch.no_grad():
                            next_v = v_net(next_observation)
                        target_q = reward + discount * next_v * (1 - done.float())
                        loss_q = nn.functional.mse_loss(q_values, target_q)

                    optimizer_q.zero_grad()
                    loss_q.backward()
                    optimizer_q.step()

                    # V loss
                    with torch.enable_grad():
                        v_values = v_net(observation)
                        with torch.no_grad():
                            sa = torch.cat([observation, action], dim=1)
                            q_detach = q_net(sa)
                        diff = q_detach - v_values
                        loss_v = expectile_loss(diff, tau)

                    optimizer_v.zero_grad()
                    loss_v.backward()
                    optimizer_v.step()

                    # Actor loss
                    with torch.enable_grad():
                        sa = torch.cat([observation, action], dim=1)
                        q_detach = q_net(sa)
                        v_detach = v_net(observation)
                        adv = q_detach - v_detach
                        weight = torch.exp(beta * adv)
                        temperature = 2.0  # To smooth the distribution in softmax
                        logits = actor(observation)
                        log_prob = nn.functional.log_softmax(logits/temperature, dim=1)
                        action_index = action.argmax(dim=1, keepdim=True)
                        chosen_log_prob = torch.gather(log_prob, dim=1, index=action_index)
                        loss_actor = -weight * chosen_log_prob
                        loss_actor = loss_actor.mean()

                    optimizer_actor.zero_grad()
                    loss_actor.backward()
                    optimizer_actor.step()

                total_frames += data.numel()
                if total_frames >= min_num_iters:
                    break

            if save_replay_buffer and replay_buffer_dict is not None:
                replay_buffer_dict[data_id] = rb

            dummy_model = DQN("MlpPolicy", env, policy_kwargs=policy_kwargs, verbose=0, gamma=discount, device=device,
                              buffer_size=0)
            dummy_model.policy.q_net = actor

            flat_params = torch.from_numpy(dummy_model.policy.parameters_to_vector())

            model_state_dict = actor.state_dict()
            v_state = v_net.state_dict()
            ctx.save_for_backward(env_parameter.detach(), flat_params.detach())
            ctx.model_state = model_state_dict
            ctx.v_state = v_state
            return flat_params, v_state

        @staticmethod
        def backward(ctx, dl_dmodel):
            env_parameter, model_parameters = ctx.saved_tensors
            env = env_wrapper(env_parameter.detach())

            # Create dummy DQN for Hessian compatibility
            dummy_model = DQN("MlpPolicy", env, policy_kwargs=policy_kwargs, verbose=0, gamma=discount, device=device, buffer_size=0)

            # Load the actor state into dummy_model.policy.q_net
            obs_dim = env.observation_space.shape[0]
            action_dim = env.action_space.n
            actor = MLP(in_features=obs_dim, out_features=action_dim, depth=2, num_cells=64).to(device)
            actor.load_state_dict(ctx.model_state)
            dummy_model.policy.q_net = actor

            v_net = MLP(
                in_features=obs_dim,
                out_features=1,
                depth=2,
                num_cells=64,
            ).to(device)
            v_net.load_state_dict(ctx.v_state)

            with torch.enable_grad():
                env_parameter_var = torch.autograd.Variable(env_parameter, requires_grad=True)
                env_var = env_wrapper(env_parameter_var)

                dl_denv = compute_policy_hessian_inverse_vector_product(
                    -dl_dmodel, env_parameter_var, env_var, dummy_model,
                    num_episodes=100,
                    discount=discount,
                    baseline=baseline,
                    backprop_method=backprop_method,
                    v_net=v_net  # Pass v_net for baseline
                )

            # Clearing gradients
            for param in dummy_model.policy.parameters():
                if param.grad is not None:
                    del param.grad
            if env_parameter_var.grad is not None:
                del env_parameter_var.grad

            return dl_denv

    return DiffIQLFn.apply