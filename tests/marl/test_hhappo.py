"""Tests for HHAPPO hybrid action PPO.

3 tests covering: loss = cont + disc, independent advantage streams,
no gradient leakage between continuous and discrete heads.
"""

import sys
sys.path.insert(0, ".")

import torch
import numpy as np
import pytest

from src.marl.hhappo import hhappo_loss
from src.marl.ppo import ppo_clip_loss


class TestHHAPPOLoss:
    """Tests for hhappo_loss."""

    def test_loss_is_sum_of_cont_and_disc(self) -> None:
        """HHAPPO loss equals sum of continuous and discrete clip losses."""
        new_cont_lp = torch.tensor([-0.5, -1.0])
        old_cont_lp = torch.tensor([-0.6, -1.1])
        cont_adv = torch.tensor([1.0, -0.5])

        new_disc_lp = torch.tensor([-0.3, -0.8])
        old_disc_lp = torch.tensor([-0.4, -0.9])
        disc_adv = torch.tensor([0.5, -0.2])

        combined = hhappo_loss(
            new_cont_lp, old_cont_lp, cont_adv,
            new_disc_lp, old_disc_lp, disc_adv,
        )

        cont_only = ppo_clip_loss(new_cont_lp, old_cont_lp, cont_adv)
        disc_only = ppo_clip_loss(new_disc_lp, old_disc_lp, disc_adv)

        assert combined.item() == pytest.approx(
            cont_only.item() + disc_only.item(), abs=1e-5,
        )

    def test_independent_advantage_streams(self) -> None:
        """Changing continuous advantages does not affect discrete loss component."""
        new_cont_lp = torch.tensor([-0.5, -1.0])
        old_cont_lp = torch.tensor([-0.6, -1.1])
        new_disc_lp = torch.tensor([-0.3, -0.8])
        old_disc_lp = torch.tensor([-0.4, -0.9])

        cont_adv_1 = torch.tensor([1.0, -0.5])
        cont_adv_2 = torch.tensor([10.0, -5.0])  # Different advantages
        disc_adv = torch.tensor([0.5, -0.2])

        loss1 = hhappo_loss(new_cont_lp, old_cont_lp, cont_adv_1,
                            new_disc_lp, old_disc_lp, disc_adv)
        loss2 = hhappo_loss(new_cont_lp, old_cont_lp, cont_adv_2,
                            new_disc_lp, old_disc_lp, disc_adv)

        # Discrete component should be same in both cases
        disc_loss = ppo_clip_loss(new_disc_lp, old_disc_lp, disc_adv)

        # Total losses differ because cont advantages differ
        assert loss1.item() != pytest.approx(loss2.item(), abs=1e-3)

        # But disc part is unchanged
        cont_loss_1 = ppo_clip_loss(new_cont_lp, old_cont_lp, cont_adv_1)
        cont_loss_2 = ppo_clip_loss(new_cont_lp, old_cont_lp, cont_adv_2)
        assert (loss1 - cont_loss_1).item() == pytest.approx(
            (loss2 - cont_loss_2).item(), abs=1e-5,
        )

    def test_no_gradient_leakage(self) -> None:
        """Gradients from discrete loss don't flow into continuous parameters."""
        # Create separate parameters for continuous and discrete
        cont_param = torch.randn(2, requires_grad=True)
        disc_param = torch.randn(2, requires_grad=True)

        old_cont_lp = torch.tensor([-0.5, -1.0])
        old_disc_lp = torch.tensor([-0.3, -0.8])

        new_cont_lp = cont_param  # Gradients will flow through cont_param
        new_disc_lp = disc_param  # Gradients will flow through disc_param

        cont_adv = torch.tensor([1.0, -0.5])
        disc_adv = torch.tensor([0.5, -0.2])

        loss = hhappo_loss(
            new_cont_lp, old_cont_lp, cont_adv,
            new_disc_lp, old_disc_lp, disc_adv,
        )
        loss.backward()

        # Both should have gradients (independent paths)
        assert cont_param.grad is not None
        assert disc_param.grad is not None

        # Verify that disc_param gradient comes only from disc loss
        disc_param_2 = disc_param.detach().clone().requires_grad_(True)
        disc_loss_only = ppo_clip_loss(disc_param_2, old_disc_lp, disc_adv)
        disc_loss_only.backward()

        torch.testing.assert_close(disc_param.grad, disc_param_2.grad)
