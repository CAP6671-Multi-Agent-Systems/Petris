"""
Another main script for training the avaialable agents.
"""

import os
import sys
import argparse
import logging
from pathlib import Path

# NOTE: Going to set the paths in here before importing packages. 
# NOTE: This is important for importing packages.
sys.path.append(str(Path().parent))

# Third party libs
import pygame
from pygame.time import Clock
from pygame.surface import Surface

from src import paths
from src.log.log import initialize_logger
from src.scenes.scenes import GameMetaData, TitleScene, Scenes
from src.agents.dqn import train_dqn
from src.agents.reinforce_agent import train_reinforce

logger = logging.getLogger(__name__)

TRAIN_LOG_FILE = "train.log"
TRAIN_LOG_PATH = paths.LOG_PATH / TRAIN_LOG_FILE

# TODO: Need to use pickle to serialize the trained model for the other main script to reuse.


def main(agent: str, speed: int, debug: bool = False) -> int:
    """
    Main function that handles all exceptions and begins the training process.

    Args:
        agent: The agent to train.
        debug: Flag to display debug logs
    Returns:
        int: Exit code
    """
    exit_code = 0

    try:
        initialize_logger(log_path=TRAIN_LOG_PATH, debug=debug)

        logger.info("Beginning Training Agent: %s", agent.upper())

        # Positioned Window on the screen
        os.environ['SDL_VIDEO_WINDOW_POS'] = "(100,100)"

        # Main game clock that allows the piece to drop.
        clock: Clock = pygame.time.Clock()
        
        screen_size = (GameMetaData.screen_width, GameMetaData.screen_height)
        main_screen: Surface = pygame.display.set_mode(size=screen_size)

        Scenes.titleScene = TitleScene()
        Scenes.active_scene = Scenes.titleScene

        if agent.lower() == "dqn":
            train_dqn(main_screen=main_screen, clock=clock, speed=speed, episodes=5, batch_size=5)
        elif agent.lower() == "reinforce":
            # train_reinforce()
            pass
    except Exception as ex:
        exit_code = 1
        logger.exception(ex)
    finally:
        logger.info("End of Training: %s", exit_code)

    return exit_code


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("-a", "--agent", action="store", required=True, type=str,
                        help="Agent to train.")
    parser.add_argument("-s", "--speed", action="store", required=False, default=50, type=int,
                        help="The speed at which the tetris piece gets dropped. "
                        "Higher is faster. Default is 50.")
    parser.add_argument("-d", "--debug", action="store_true", required=False, default=False,
                        help="Displays the debug logs.")

    args, _ = parser.parse_known_args()
    sys.exit(main(agent=args.agent, speed=args.speed, debug=args.debug))
