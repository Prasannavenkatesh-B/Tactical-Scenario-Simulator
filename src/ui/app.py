"""Main tactical simulation desktop application loop, event loop, and frame pacing."""

import sys
from typing import Any
import numpy as np
import pygame

from src.database.db import Database
from src.simulator.config import DEFAULT_EPISODE_HORIZON
from src.simulator.env import TacticalEnv
from src.simulator.scenarios import ScenarioConfig, generate_scenario
from src.ui.config import (
    MAX_SIM_STEPS_PER_FRAME,
    TARGET_FPS,
    WINDOW_HEIGHT,
    WINDOW_WIDTH,
)
from src.ui.controls import InputHandler
from src.ui.renderer import Renderer
from src.ui.state import UIState


class TacticalUIApp:
    """Main desktop application window managing life-cycle, event dispatch, and frame rendering."""

    def __init__(
        self,
        scenario_config: ScenarioConfig | None = None,
        db: Database | None = None,
        seed: int = 0,
    ) -> None:
        if not pygame.get_init():
            pygame.init()

        pygame.display.set_caption("Tactical Simulation System (TSS) - DRDO")
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        self.clock = pygame.time.Clock()

        self.seed = seed
        self.rng = np.random.default_rng(seed)

        if scenario_config is None:
            self.scenario_config = generate_scenario(level=3, rng=self.rng, seed=seed)
        else:
            self.scenario_config = scenario_config

        # Instantiate environment in interactive mode
        self.env = TacticalEnv(
            scenario_config=self.scenario_config,
            mode="interactive",
            seed=seed,
        )

        # Initialize State, Renderer, and Controls
        self.state = UIState(
            env=self.env,
            scenario_config=self.scenario_config,
            rng=self.rng,
            db=db,
        )
        self.renderer = Renderer(self.screen, self.state)
        self.input_handler = InputHandler(
            state=self.state,
            map_view=self.renderer.map_view,
            sidebar=self.renderer.sidebar,
            timeline=self.renderer.timeline,
            renderer=self.renderer,
        )

        self.is_running: bool = False
        self.step_accumulator: float = 0.0

    def update(self, dt: float) -> None:
        """Advance physics and decision models according to active playback mode and speed multiplier."""
        if self.state.mode != "simulate":
            return

        self.step_accumulator += self.state.speed_multiplier
        steps_taken = 0
        while self.step_accumulator >= 1.0 and steps_taken < MAX_SIM_STEPS_PER_FRAME:
            self.state.step_simulation()
            self.step_accumulator -= 1.0
            steps_taken += 1

    def render(self) -> None:
        """Composite and render current frame."""
        self.renderer.render()
        self.renderer.render_fps(self.clock.get_fps())

    def run(self, max_frames: int | None = None) -> None:
        """Execute interactive event and render loop."""
        self.is_running = True
        frame_count = 0

        try:
            while self.is_running:
                dt = self.clock.tick(TARGET_FPS) / 1000.0

                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        self.is_running = False
                        break
                    elif event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_q and (
                            pygame.key.get_mods() & pygame.KMOD_CTRL or pygame.key.get_mods() & pygame.KMOD_META
                        ):
                            self.is_running = False
                            break

                    self.input_handler.handle_event(event)

                if not self.is_running:
                    break

                self.update(dt)
                self.render()
                pygame.display.flip()

                frame_count += 1
                if max_frames is not None and frame_count >= max_frames:
                    self.is_running = False
                    break

        finally:
            self.close()

    def close(self) -> None:
        """Cleanly terminate application and release resources."""
        self.is_running = False
