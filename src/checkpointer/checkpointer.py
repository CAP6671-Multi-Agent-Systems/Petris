"""Checkpointer module"""

import logging

from tensorflow import Variable
from tf_agents.utils.common import Checkpointer
from tf_agents.agents.tf_agent import TFAgent
from tf_agents.replay_buffers.replay_buffer import ReplayBuffer

from src import paths

logger = logging.getLogger(__name__)

CHECKPOINT_DIR = paths.BASE_DIR / "checkpoint"


def create_checkpointer(name: str,
                        agent: TFAgent,
                        replay_buffer: ReplayBuffer,
                        global_step: Variable,
                        max_to_keep: int = 1) -> Checkpointer:
    """
    Creates the checkpointer object as well as the 'checkpoint' directory.

    Args:
        name (str): Name of the training agent
        agent (TFAgent): Agent that is being trained
        replay_buffer (ReplayBuffer): Replay buffer for the agent
        global_step (Variable): Global step needed for the checkpoint
        max_to_keep (int, optional): Not sure what this is but keep it at 1.

    Returns:
        Checkpointer: Checkpointer object to use.
    """
    
    CHECKPOINT_DIR.mkdir(exist_ok=True)
    checkpointer_dir = CHECKPOINT_DIR / name
    checkpointer_dir.mkdir(exist_ok=True)
    
    train_checkpointer = Checkpointer(
        ckpt_dir=CHECKPOINT_DIR,
        max_to_keep=max_to_keep,
        agent=agent,
        policy=agent.policy,
        replay_buffer=replay_buffer,
        global_step=global_step
    )
    
    return train_checkpointer
