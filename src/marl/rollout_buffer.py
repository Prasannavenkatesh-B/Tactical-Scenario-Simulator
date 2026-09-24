"""On-policy PPO rollout buffer with GAE computation.

Preallocates numpy arrays for memory efficiency. Supports continuous,
discrete, and hybrid action types.
"""

import typing

import numpy as np


class RolloutBuffer:
    """On-policy rollout buffer with preallocated numpy arrays and GAE.

    Stores transitions collected during policy rollout. Computes
    Generalized Advantage Estimation (GAE) and normalized advantages
    for PPO updates.

    Args:
        batch_size: Maximum number of transitions to store.
        obs_dim: Observation vector dimension.
        action_dim: Action vector dimension (total for hybrid).
        action_type: "continuous", "discrete", or "hybrid".
    """

    def __init__(
        self,
        batch_size: int,
        obs_dim: int,
        action_dim: int,
        action_type: str = "discrete",
    ) -> None:
        self.batch_size = batch_size
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.action_type = action_type
        self.ptr: int = 0

        # Preallocate arrays
        self.observations = np.zeros((batch_size, obs_dim), dtype=np.float32)
        self.actions = np.zeros((batch_size, action_dim), dtype=np.float32)
        self.rewards = np.zeros(batch_size, dtype=np.float32)
        self.dones = np.zeros(batch_size, dtype=np.float32)
        self.values = np.zeros(batch_size, dtype=np.float32)
        self.log_probs = np.zeros(batch_size, dtype=np.float32)
        self.advantages = np.zeros(batch_size, dtype=np.float32)
        self.returns = np.zeros(batch_size, dtype=np.float32)

        # Optional hidden states for GRU-based policies
        self.hidden_states: list[np.ndarray | None] = [None] * batch_size

        # For hybrid: separate log probs
        self.cont_log_probs = np.zeros(batch_size, dtype=np.float32)
        self.disc_log_probs = np.zeros(batch_size, dtype=np.float32)

    def add(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: float,
        done: bool,
        value: float,
        log_prob: float,
        hidden_state: np.ndarray | None = None,
        cont_log_prob: float = 0.0,
        disc_log_prob: float = 0.0,
    ) -> None:
        """Add a single transition to the buffer.

        Args:
            obs: Observation vector.
            action: Action vector.
            reward: Scalar reward.
            done: Episode terminal flag.
            value: Value function estimate V(s).
            log_prob: Log probability of the action.
            hidden_state: Optional GRU hidden state.
            cont_log_prob: Continuous action log prob (hybrid only).
            disc_log_prob: Discrete action log prob (hybrid only).
        """
        idx = self.ptr
        self.observations[idx] = obs
        self.actions[idx] = action
        self.rewards[idx] = reward
        self.dones[idx] = float(done)
        self.values[idx] = value
        self.log_probs[idx] = log_prob
        self.hidden_states[idx] = hidden_state
        self.cont_log_probs[idx] = cont_log_prob
        self.disc_log_probs[idx] = disc_log_prob
        self.ptr += 1

    def compute_advantages_and_returns(
        self,
        last_value: float,
        gamma: float,
        gae_lambda: float,
    ) -> None:
        """Compute GAE advantages and discounted returns.

        GAE: δ_t = r_t + γ * V(s_{t+1}) - V(s_t)
             Â_t = Σ_{l=0}^{∞} (γλ)^l * δ_{t+l}
        Terminal states: bootstrap_value = 0 when done=True.

        Advantages are normalized to mean=0, std=1 after computation.

        Args:
            last_value: V(s_T) for the state after the last stored transition.
            gamma: Discount factor.
            gae_lambda: GAE lambda.
        """
        n = self.ptr
        gae = 0.0

        for t in reversed(range(n)):
            if t == n - 1:
                next_value = last_value
                next_non_terminal = 1.0 - self.dones[t]
            else:
                next_value = self.values[t + 1]
                next_non_terminal = 1.0 - self.dones[t]

            delta = (
                self.rewards[t]
                + gamma * next_value * next_non_terminal
                - self.values[t]
            )
            gae = delta + gamma * gae_lambda * next_non_terminal * gae
            self.advantages[t] = gae
            self.returns[t] = gae + self.values[t]

        # Normalize advantages
        adv_slice = self.advantages[:n]
        adv_std = adv_slice.std()
        if adv_std > 1e-8:
            self.advantages[:n] = (adv_slice - adv_slice.mean()) / adv_std

    def get_minibatches(
        self,
        minibatch_size: int,
        shuffle: bool = True,
    ) -> typing.Iterator[dict[str, np.ndarray]]:
        """Iterate over random minibatches.

        Each minibatch is a dict of numpy arrays covering exactly
        minibatch_size samples. All samples are covered exactly once.

        Args:
            minibatch_size: Number of samples per minibatch.
            shuffle: Whether to shuffle indices.

        Yields:
            Dict with keys: obs, actions, log_probs, advantages, returns,
            values, cont_log_probs, disc_log_probs.
        """
        n = self.ptr
        indices = np.arange(n)
        if shuffle:
            np.random.shuffle(indices)

        for start in range(0, n, minibatch_size):
            end = min(start + minibatch_size, n)
            idx = indices[start:end]
            yield {
                "obs": self.observations[idx],
                "actions": self.actions[idx],
                "log_probs": self.log_probs[idx],
                "advantages": self.advantages[idx],
                "returns": self.returns[idx],
                "values": self.values[idx],
                "cont_log_probs": self.cont_log_probs[idx],
                "disc_log_probs": self.disc_log_probs[idx],
            }

    def clear(self) -> None:
        """Reset buffer pointer and clear data."""
        self.ptr = 0
        self.observations[:] = 0.0
        self.actions[:] = 0.0
        self.rewards[:] = 0.0
        self.dones[:] = 0.0
        self.values[:] = 0.0
        self.log_probs[:] = 0.0
        self.advantages[:] = 0.0
        self.returns[:] = 0.0
        self.cont_log_probs[:] = 0.0
        self.disc_log_probs[:] = 0.0
        self.hidden_states = [None] * self.batch_size

    def __len__(self) -> int:
        """Return the number of stored transitions."""
        return self.ptr
