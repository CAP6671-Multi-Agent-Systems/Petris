import pandas as pd
import numpy as np
from tf_agents.trajectories import Trajectory
from src.scenes.scenes import State

import logging

logger = logging.getLogger(__name__)

class MetricsObserver:
    def __init__(self):
        self.tetris_heatmap = np.zeros((20, 10), dtype=int)
        self.reward_per_block = pd.DataFrame(columns=['block','reward'])
        self.total_placements = 0
        self.lines_placed = 0

    def __call__(self, trajectory: Trajectory, collision_detected: bool = False, block_placed: int = 0) -> None:
        if collision_detected:
            map = np.reshape(trajectory.observation, (20, 10))
            self.tetris_heatmap += (map != 0).astype(int)
            self.total_placements += 1
            if block_placed != 0:
                append = pd.DataFrame(data=[[block_placed,trajectory.reward]], columns=['block','reward'])
                self.reward_per_block = pd.concat([self.reward_per_block,append], ignore_index=True)
        if trajectory.is_last():
            self.lines_placed = State.full_line_no

    def get_heatmap_dataframe(self) -> pd.DataFrame:
        max_placements = np.max(self.tetris_heatmap)
        avg_heatmap = self.tetris_heatmap / max_placements if max_placements > 0 else self.tetris_heatmap
        return pd.DataFrame(avg_heatmap, columns=[f"col_{i}" for i in range(10)])
    
    def get_reward_per_block_dataframe(self) -> pd.DataFrame:
        avg_rewards = self.reward_per_block.groupby('block')['reward'].mean()
        return pd.DataFrame({'block': avg_rewards.index, 'avg_reward': avg_rewards.values})
    
    def reset(self) -> None:
        self.tetris_heatmap = np.zeros((20, 10), dtype=int)
        self.reward_per_block = pd.DataFrame(columns=['block','reward'])
        self.total_placements = 0
        self.lines_placed = -1