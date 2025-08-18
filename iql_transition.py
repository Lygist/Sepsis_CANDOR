import torch
import numpy as np
from d3rlpy.models.encoders import VectorEncoderFactory
from d3rlpy.dataset import MDPDataset
from discrete_iql_config import DiscreteIQLConfig

class IQLSolver:
    def __init__(
        self,
        env_wrapper,
        real_trajectories,
        discount=0.99,
        n_steps=1000,
        warmup_steps=100,
        batch_size=256,
        device="cpu",
        net_arch=(64, 64),
        tau=5.0,
        ess_const=10.0,
    ):
        self.env_wrapper = env_wrapper
        self.real_trajectories = real_trajectories
        self.discount = discount
        self.device = device
        self.tau = tau
        self.n_steps = n_steps
        self.warmup_steps = warmup_steps
        self.ess_const = ess_const

        encoder = VectorEncoderFactory(hidden_units=net_arch)
        self.config = DiscreteIQLConfig(
            gamma=discount,
            actor_learning_rate=1e-4,
            critic_learning_rate=1e-4,
            batch_size=batch_size,
        )
        self.trained_model = None

    def run_epoch(self, prediction, label, warm_start=False, previous_model=None):
        env = self.env_wrapper(transition_prob=prediction.detach(), true_transition_prob=label.detach())
        obs, acts, rews, terms = [], [], [], []
        for traj in self.real_trajectories:
            for s, a, r, _, done in traj:
                obs.append(s.detach().cpu().numpy())
                acts.append(int(a.detach().cpu().item()))
                rews.append(float(r))
                terms.append(bool(done))
        dataset = MDPDataset(
            observations=np.stack(obs),
            actions=np.array(acts, dtype=np.int64),
            rewards=np.array(rews, dtype=np.float32),
            terminals=np.array(terms, dtype=bool)
        )

        if isinstance(self.device, int):
            device_param = f"cuda:{self.device}"
        elif isinstance(self.device, str):
            device_param = self.device if ':' in self.device else self.device + ':0'
        elif isinstance(self.device, bool):
            device_param = "cuda:0" if self.device else "cpu:0"
        else:
            device_param = "cpu:0"

        # -------------------- training ------------------
        if warm_start and previous_model is not None:
            algo = previous_model if not isinstance(previous_model, str) else self.config.create(device=device_param)
            if isinstance(previous_model, str):
                algo.build_with_env(env)
                algo.load_model(previous_model)
        else:
            algo = self.config.create(device=device_param)
            algo.build_with_env(env)

        if warm_start:
            steps = self.warmup_steps
        else:
            steps = self.n_steps

        algo.fit(dataset, n_steps=steps, show_progress=False)
        self.trained_model = algo

        # ---------------- offline evaluation --------------
        rewards, weights = [], []
        N, T = len(self.real_trajectories), len(self.real_trajectories[0])
        for traj in self.real_trajectories:
            w, rsum = 1.0, 0.0
            for t, (s, a, r, _, behav_prob) in enumerate(traj):
                q_fn = algo.impl.q_function
                if isinstance(q_fn, (list, torch.nn.ModuleList)):
                    q_fn = q_fn[0]
                with torch.no_grad():
                    s_tensor = torch.tensor(s.cpu().numpy(), dtype=torch.float32, device=self.device).unsqueeze(0)
                    q_vals = q_fn(s_tensor).q_value.cpu()[0]
                prob = torch.softmax(q_vals / self.tau, dim=0)[int(a)]
                w *= prob / behav_prob
                rsum += (self.discount ** t) * float(r)
            rewards.append(rsum)
            weights.append(w)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        weights = torch.tensor(weights, dtype=torch.float32)
        cwpdis = torch.sum(weights * rewards) / (weights.sum() + 1e-6)
        ess = (weights.sum() ** 2) / (weights ** 2).sum()
        soft_eval = cwpdis - self.ess_const / torch.sqrt(ess + 1e-3)

        # -------------------- simulation ------------------
        total_rewards = []
        for _ in range(100):
            obs = env.reset()
            done, total, t = False, 0.0, 0
            while not done and t < 100:
                with torch.no_grad():
                    if isinstance(obs, torch.Tensor):
                        obs_tensor = obs.detach().clone().float().unsqueeze(0).to(self.device)
                    else:
                        obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(self.device)
                    q_fn = algo.impl.q_function
                    if isinstance(q_fn, torch.nn.ModuleList):
                        q_fn = q_fn[0]
                    q_vals = q_fn(obs_tensor).q_value.cpu().numpy()[0]
                    action = int(np.argmax(q_vals))
                obs, r, done, _ = env.step(action)
                total += (self.discount ** t) * r
                t += 1
            total_rewards.append(total)
        strict_eval = float(np.mean(total_rewards))

        return {
            'model': algo,
            'soft_eval': soft_eval.item(),
            'strict_eval': strict_eval,
            'cwpdis': cwpdis.item(),
            'ess': ess.item(),
        }
