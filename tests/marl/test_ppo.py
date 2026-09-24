"""Tests for PPO-Clip loss functions and GAE computation.

6 tests covering: clip loss identity, manual computation match,
clipping activation, value loss, entropy, GAE reference match.
"""

import sys
sys.path.insert(0, ".")

import numpy as np
import torch
import pytest

from src.marl.ppo import ppo_clip_loss, value_loss, entropy_bonus, compute_gae


class TestPPOClipLoss:
    """Tests for ppo_clip_loss."""

    def test_identity_ratio(self) -> None:
        """When old_log_probs == new_log_probs, ratio = 1, loss = mean(advantages)."""
        log_probs = torch.tensor([-1.0, -2.0, -0.5, -1.5])
        advantages = torch.tensor([1.0, -0.5, 0.3, -0.1])
        loss = ppo_clip_loss(log_probs, log_probs, advantages)
        # ratio = 1 everywhere, so surr1 = surr2 = advantages
        assert loss.item() == pytest.approx(advantages.mean().item(), abs=1e-5)

    def test_manual_computation(self) -> None:
        """PPO clip loss matches manual computation."""
        new_lp = torch.tensor([-0.5, -1.0])
        old_lp = torch.tensor([-1.0, -0.5])
        advantages = torch.tensor([1.0, 1.0])
        eps = 0.2

        ratio = torch.exp(new_lp - old_lp)  # [e^0.5, e^-0.5]
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - eps, 1 + eps) * advantages
        expected = torch.min(surr1, surr2).mean()

        loss = ppo_clip_loss(new_lp, old_lp, advantages, eps)
        assert loss.item() == pytest.approx(expected.item(), abs=1e-5)

    def test_clipping_activates(self) -> None:
        """When ratio is far from 1, clipping is active."""
        new_lp = torch.tensor([0.0])   # log_prob = 0 => prob = 1
        old_lp = torch.tensor([-3.0])  # log_prob = -3 => prob ≈ 0.05
        advantages = torch.tensor([1.0])
        eps = 0.2

        ratio = torch.exp(new_lp - old_lp)  # ≈ 20
        assert ratio.item() > 1 + eps  # Clipping should activate

        loss = ppo_clip_loss(new_lp, old_lp, advantages, eps)
        # Clipped: min(20 * 1, 1.2 * 1) = 1.2
        assert loss.item() == pytest.approx(1.2, abs=1e-5)


class TestValueLoss:
    """Tests for value_loss."""

    def test_zero_error(self) -> None:
        """Value loss is 0 when predictions equal returns."""
        values = torch.tensor([1.0, 2.0, 3.0])
        returns = torch.tensor([1.0, 2.0, 3.0])
        loss = value_loss(values, values, returns)
        assert loss.item() == pytest.approx(0.0, abs=1e-6)

    def test_positive_error(self) -> None:
        """Value loss is positive when predictions differ from returns."""
        new_values = torch.tensor([1.0, 2.0])
        old_values = torch.tensor([0.9, 1.9])
        returns = torch.tensor([1.5, 2.5])
        loss = value_loss(new_values, old_values, returns)
        assert loss.item() > 0


class TestEntropyBonus:
    """Tests for entropy_bonus."""

    def test_uniform_max_entropy(self) -> None:
        """Uniform distribution has maximum entropy."""
        from torch.distributions import Categorical
        uniform_logits = torch.zeros(4, 10)  # 10 actions, uniform
        peaked_logits = torch.zeros(4, 10)
        peaked_logits[:, 0] = 100.0  # Very peaked

        uniform_dist = Categorical(logits=uniform_logits)
        peaked_dist = Categorical(logits=peaked_logits)

        uniform_ent = entropy_bonus(uniform_dist)
        peaked_ent = entropy_bonus(peaked_dist)

        assert uniform_ent.item() > peaked_ent.item()


class TestComputeGAE:
    """Tests for compute_gae."""

    def test_gae_reference(self) -> None:
        """GAE computation matches reference implementation."""
        rewards = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        values = np.array([0.5, 1.0, 1.5], dtype=np.float32)
        dones = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        last_value = 2.0
        gamma = 0.99
        lam = 0.95

        advantages, returns = compute_gae(rewards, values, dones, last_value, gamma, lam)

        # Hand compute:
        # δ_2 = 3 + 0.99*2.0 - 1.5 = 3.48
        # A_2 = 3.48
        # δ_1 = 2 + 0.99*1.5 - 1.0 = 2.485
        # A_1 = 2.485 + 0.99*0.95*3.48 = 5.75494
        # δ_0 = 1 + 0.99*1.0 - 0.5 = 1.49
        # A_0 = 1.49 + 0.99*0.95*5.75494 = 6.90247
        assert advantages[2] == pytest.approx(3.48, abs=1e-3)
        assert advantages[1] == pytest.approx(5.75494, abs=1e-2)
        assert advantages[0] == pytest.approx(6.90247, abs=1e-2)

        # Returns = advantages + values
        np.testing.assert_allclose(returns, advantages + values, atol=1e-5)

    def test_gae_terminal(self) -> None:
        """GAE correctly handles terminal transitions."""
        rewards = np.array([1.0, 2.0], dtype=np.float32)
        values = np.array([0.5, 1.0], dtype=np.float32)
        dones = np.array([0.0, 1.0], dtype=np.float32)
        last_value = 999.0  # Should not matter because t=1 is terminal

        advantages, returns = compute_gae(rewards, values, dones, last_value, 0.99, 0.95)

        # At t=1 (terminal): δ_1 = 2.0 + 0 - 1.0 = 1.0, A_1 = 1.0
        assert advantages[1] == pytest.approx(1.0, abs=1e-5)

        # At t=0: δ_0 = 1.0 + 0.99*1.0*1.0 - 0.5 = 1.49
        # A_0 = 1.49 + 0.99*0.95*1.0*1.0 = 2.4305
        assert advantages[0] == pytest.approx(2.4305, abs=1e-3)
