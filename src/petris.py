"""Main launcher script for the game"""

# Built-in libs
import os
import sys
import argparse
import logging
from pathlib import Path
from typing import Optional
from functools import partial

# NOTE: Going to set the paths in here before importing packages. 
# NOTE: This is important for importing packages.
sys.path.append(str(Path().parent))

# Third party libs
import pygame
from pygame.time import Clock
from pygame.surface import Surface
from bayes_opt import BayesianOptimization

# Custom packages.
from src import paths
from src.log.log import initialize_logger
from src.scenes.scenes import GameMetaData, TitleScene, Scenes
from src.params.parameters import Parameters, get_nested_value
from src.petris_environment.petris_environment import PetrisEnvironment
from src.game_runner.game_runner import play_game
from src.agents.random_agent import play_random_agent
from src.agents.dqn import train_dqn
from src.agents.ppo import train_ppo
from src.agents.reinforce_agent import train_reinforce
from src.metrics.metrics import Metrics
from tf_agents.environments.tf_py_environment import TFPyEnvironment

logger = logging.getLogger(__name__)

PETRIS_LOG_FILE = "petris.log"
PETRIS_LOG_PATH = paths.LOG_PATH / PETRIS_LOG_FILE


def main(speed: int, paramFile: Optional[str] = None ,rand_iter: int = 5, iter: int = 5, debug: bool = False) -> int:
    """
    Main function for the game

    Args:
        speed (int): Speed at which the tetris piece gets dropped.

    Returns:
        int: Exit code
    """
    
    exit_code = 0
    
    try:
        initialize_logger(log_path=PETRIS_LOG_PATH, debug=debug)
        
        logger.info("Starting Petris Game")
        logger.info("Args: (speed=%s, parameters=%s, random_iterations=%i, iterations=%i)", speed, paramFile)
        
        # Positioned Window on the screen
        os.environ['SDL_VIDEO_WINDOW_POS'] = "(100,100)"

        # Main game clock that allows the piece to drop.
        clock: Clock = pygame.time.Clock()
        
        screen_size = (GameMetaData.screen_width, GameMetaData.screen_height)
        main_screen: Surface = pygame.display.set_mode(size=screen_size)
        
        # Title of the window header
        pygame.display.set_caption("Petris")

        Scenes.titleScene = TitleScene()
        Scenes.active_scene = Scenes.titleScene
        
        logger.debug("Initialized Game Clock: %s", clock)
        logger.debug("Main Screen Built: %s", main_screen)
        logger.debug("Scene Setup: (titleScene=%s, gameScene=%s, active_scene=%s)", 
                     Scenes.titleScene, Scenes.gameScene, Scenes.active_scene)

        if paramFile and os.path.isfile(paramFile):
            parameters = Parameters(paramFile)
            metrics = Metrics(parameters=parameters)
            agent = parameters.agent
            opt_funct = None
            optimizer = None

            if agent and agent.lower() == "random":
                play_random_agent(env=PetrisEnvironment(parameters=parameters), main_screen=main_screen, clock=clock, speed=speed, num_episodes=parameters.agent.epoch)

            elif agent and agent.lower() == "dqn":
                logger.info("Training DQN")

                opt_funct =  partial(train_dqn,main_screen=main_screen, clock=clock, speed=speed, metrics=metrics, parameters=parameters, type=parameters.to_maximize)

                if parameters.to_maximize == "agent":
                    layer_0 = tuple(int(x) for x in parameters.bounds.agent.layer_0)
                    layer_1 = tuple(int(x) for x in parameters.bounds.agent.layer_1)
                    activation = tuple(float(x) for x in parameters.bounds.agent.activation)
                    learning_rate = tuple(float(x) for x in parameters.bounds.agent.learning_rate)
                    epochs = tuple(int(x) for x in parameters.bounds.agent.epochs)
                    epsilon = tuple(float(x) for x in parameters.bounds.agent.epsilon)
                    optimizer = BayesianOptimization(
                        f=opt_funct,
                        pbounds = {
                            'layer_0': layer_0,
                            'layer_1': layer_1,
                            'learning_rate': learning_rate,
                            'activation': activation,
                            'epochs': epochs,
                            'epsilon': epsilon
                        }
                    )
                elif parameters.to_maximize == "environment":
                    early_penalty = tuple(int(x) for x in parameters.bounds.environment.early_penalty)
                    holes_penalty = tuple(float(x) for x in parameters.bounds.environment.holes_penalty)
                    height_penalty = tuple(float(x) for x in parameters.bounds.environment.height_penalty)
                    game_over_penalty = tuple(int(x) for x in parameters.bounds.environment.game_over_penalty)
                    line_single_reward = tuple(int(x) for x in parameters.bounds.environment.line_single_reward)
                    line_double_reward = tuple(int(x) for x in parameters.bounds.environment.line_double_reward)
                    line_triple_reward = tuple(int(x) for x in parameters.bounds.environment.line_triple_reward)
                    line_tetris_reward = tuple(int(x) for x in parameters.bounds.environment.line_tetris_reward)
                    block_placed_reward = tuple(float(x) for x in parameters.bounds.environment.block_placed_reward)
                    press_down_reward =  tuple(float(x) for x in parameters.bounds.environment.press_down_reward)
                    optimizer = BayesianOptimization(
                        f=opt_funct,
                        pbounds= {
                            'early_penalty': early_penalty,
                            'holes_penalty': holes_penalty,
                            'height_penalty': height_penalty,
                            'game_over_penalty': game_over_penalty,
                            'line_single_reward': line_single_reward,
                            'line_double_reward': line_double_reward,
                            'line_triple_reward': line_triple_reward,
                            'line_tetris_reward': line_tetris_reward,
                            'block_placed_reward': block_placed_reward,
                            'press_down_reward': press_down_reward
                        }
                    )
                elif parameters.to_maximize == "all":
                    layer_0 = tuple(int(x) for x in parameters.bounds.agent.layer_0)
                    layer_1 = tuple(int(x) for x in parameters.bounds.agent.layer_1)
                    learning_rate = tuple(float(x) for x in parameters.bounds.agent.learning_rate)
                    epochs = tuple(int(x) for x in parameters.bounds.agent.epochs)
                    epsilon = tuple(float(x) for x in parameters.bounds.agent.epsilon)
                    early_penalty = tuple(int(x) for x in parameters.bounds.environment.early_penalty)
                    holes_penalty = tuple(float(x) for x in parameters.bounds.environment.holes_penalty)
                    height_penalty = tuple(float(x) for x in parameters.bounds.environment.height_penalty)
                    game_over_penalty = tuple(int(x) for x in parameters.bounds.environment.game_over_penalty)
                    line_single_reward = tuple(int(x) for x in parameters.bounds.environment.line_single_reward)
                    line_double_reward = tuple(int(x) for x in parameters.bounds.environment.line_double_reward)
                    line_triple_reward = tuple(int(x) for x in parameters.bounds.environment.line_triple_reward)
                    line_tetris_reward = tuple(int(x) for x in parameters.bounds.environment.line_tetris_reward)
                    block_placed_reward = tuple(float(x) for x in parameters.bounds.environment.block_placed_reward)
                    press_down_reward =  tuple(float(x) for x in parameters.bounds.environment.press_down_reward)
                    optimizer = BayesianOptimization(
                        f=opt_funct,
                        pbounds= {
                            'layer_0': layer_0,
                            'layer_1': layer_1,
                            'learning_rate': learning_rate,
                            'epochs': epochs,
                            'epsilon': epsilon,
                            'early_penalty': early_penalty,
                            'holes_penalty': holes_penalty,
                            'height_penalty': height_penalty,
                            'game_over_penalty': game_over_penalty,
                            'line_single_reward': line_single_reward,
                            'line_double_reward': line_double_reward,
                            'line_triple_reward': line_triple_reward,
                            'line_tetris_reward': line_tetris_reward,
                            'block_placed_reward': block_placed_reward,
                            'press_down_reward': press_down_reward
                        }
                    )
                elif parameters.to_maximize == "none":
                    logger.info("Manual testing activated")
                    result = opt_funct()
                    logger.info(f'Manual testing finished, average result: {result:.2f}')

                if(parameters.to_maximize != "none"):
                    optimizer.maximize(
                        init_points=rand_iter,
                        n_iter=iter
                    )
                    logger.info(optimizer.res)

            elif agent and agent.lower() == "reinforce":
                logger.info("Training Reinforce")

                opt_funct =  partial(train_reinforce,main_screen=main_screen, clock=clock, speed=speed, metrics=metrics, parameters=parameters, type=parameters.to_maximize)

                if parameters.to_maximize == "agent":
                    layer_0 = tuple(int(x) for x in parameters.bounds.agent.layer_0)
                    layer_1 = tuple(int(x) for x in parameters.bounds.agent.layer_1)
                    learning_rate = tuple(float(x) for x in parameters.bounds.agent.learning_rate)
                    epochs = tuple(int(x) for x in parameters.bounds.agent.epochs)
                    epsilon = tuple(float(x) for x in parameters.bounds.agent.epsilon)
                    optimizer = BayesianOptimization(
                        f=opt_funct,
                        pbounds = {
                            'layer_0': layer_0,
                            'layer_1': layer_1,
                            'learning_rate': learning_rate,
                            'epochs': epochs,
                            'epsilon': epsilon
                        }
                    )
                    optimizer.maximize(
                        init_points=5,
                        n_iter=5
                    )
                    logger.info(optimizer.max)
                elif parameters.to_maximize == "environment":
                    early_penalty = tuple(int(x) for x in parameters.bounds.environment.early_penalty)
                    holes_penalty = tuple(float(x) for x in parameters.bounds.environment.holes_penalty)
                    height_penalty = tuple(float(x) for x in parameters.bounds.environment.height_penalty)
                    game_over_penalty = tuple(int(x) for x in parameters.bounds.environment.game_over_penalty)
                    line_single_reward = tuple(int(x) for x in parameters.bounds.environment.line_single_reward)
                    line_double_reward = tuple(int(x) for x in parameters.bounds.environment.line_double_reward)
                    line_triple_reward = tuple(int(x) for x in parameters.bounds.environment.line_triple_reward)
                    line_tetris_reward = tuple(int(x) for x in parameters.bounds.environment.line_tetris_reward)
                    block_placed_reward = tuple(float(x) for x in parameters.bounds.environment.block_placed_reward)
                    press_down_reward =  tuple(float(x) for x in parameters.bounds.environment.press_down_reward)
                    optimizer = BayesianOptimization(
                        f=opt_funct,
                        pbounds= {
                            'early_penalty': early_penalty,
                            'holes_penalty': holes_penalty,
                            'height_penalty': height_penalty,
                            'game_over_penalty': game_over_penalty,
                            'line_single_reward': line_single_reward,
                            'line_double_reward': line_double_reward,
                            'line_triple_reward': line_triple_reward,
                            'line_tetris_reward': line_tetris_reward,
                            'block_placed_reward': block_placed_reward,
                            'press_down_reward': press_down_reward
                        }
                    )
                elif parameters.to_maximize == "all":
                    layer_0 = tuple(int(x) for x in parameters.bounds.agent.layer_0)
                    layer_1 = tuple(int(x) for x in parameters.bounds.agent.layer_1)
                    learning_rate = tuple(float(x) for x in parameters.bounds.agent.learning_rate)
                    epochs = tuple(int(x) for x in parameters.bounds.agent.epochs)
                    epsilon = tuple(float(x) for x in parameters.bounds.agent.epsilon)
                    early_penalty = tuple(int(x) for x in parameters.bounds.environment.early_penalty)
                    holes_penalty = tuple(float(x) for x in parameters.bounds.environment.holes_penalty)
                    height_penalty = tuple(float(x) for x in parameters.bounds.environment.height_penalty)
                    game_over_penalty = tuple(int(x) for x in parameters.bounds.environment.game_over_penalty)
                    line_single_reward = tuple(int(x) for x in parameters.bounds.environment.line_single_reward)
                    line_double_reward = tuple(int(x) for x in parameters.bounds.environment.line_double_reward)
                    line_triple_reward = tuple(int(x) for x in parameters.bounds.environment.line_triple_reward)
                    line_tetris_reward = tuple(int(x) for x in parameters.bounds.environment.line_tetris_reward)
                    block_placed_reward = tuple(float(x) for x in parameters.bounds.environment.block_placed_reward)
                    press_down_reward =  tuple(float(x) for x in parameters.bounds.environment.press_down_reward)
                    optimizer = BayesianOptimization(
                        f=opt_funct,
                        pbounds= {
                            'layer_0': layer_0,
                            'layer_1': layer_1,
                            'learning_rate': learning_rate,
                            'epochs': epochs,
                            'epsilon': epsilon,
                            'early_penalty': early_penalty,
                            'holes_penalty': holes_penalty,
                            'height_penalty': height_penalty,
                            'game_over_penalty': game_over_penalty,
                            'line_single_reward': line_single_reward,
                            'line_double_reward': line_double_reward,
                            'line_triple_reward': line_triple_reward,
                            'line_tetris_reward': line_tetris_reward,
                            'block_placed_reward': block_placed_reward,
                            'press_down_reward': press_down_reward
                        }
                    )
                elif parameters.to_maximize == "none":
                    logger.info("Manual testing activated")
                    result = opt_funct()
                    logger.info(f'Manual testing finished, average result: {result:.2f}')

                if(parameters.to_maximize != "none"):
                    optimizer.maximize(
                        init_points=rand_iter,
                        n_iter=iter
                    )
                    logger.info(optimizer.res)

            elif agent and agent.lower() == "ppo":
                logger.info("Training PPO")

                opt_funct = partial(train_ppo,main_screen=main_screen, clock=clock, speed=speed, parameters=parameters, metrics=metrics, type=parameters.to_maximize)

                if parameters.to_maximize == "agent":
                    actor_input_layer_0 = tuple(int(x) for x in parameters.bounds.agent.actor.input_layer_0)
                    actor_input_layer_1 = tuple(int(x) for x in parameters.bounds.agent.actor.input_layer_1)
                    actor_output_layer_0 = tuple(int(x) for x in parameters.bounds.agent.actor.output_layer_0)
                    actor_output_layer_1 = tuple(int(x) for x in parameters.bounds.agent.actor.output_layer_1)
                    actor_ltsm_size = tuple(int(x) for x in parameters.bounds.agent.actor.ltsm_size)
                    actor_activation = tuple(int(x) for x in parameters.bounds.agent.actor.activation)
                    value_input_layer_0 = tuple(int(x) for x in parameters.bounds.agent.value.input_layer_0)
                    value_input_layer_1 = tuple(int(x) for x in parameters.bounds.agent.value.input_layer_1)
                    value_output_layer_0 = tuple(int(x) for x in parameters.bounds.agent.value.output_layer_0)
                    value_output_layer_1 = tuple(int(x) for x in parameters.bounds.agent.value.output_layer_1)
                    value_ltsm_size = tuple(int(x) for x in parameters.bounds.agent.value.ltsm_size)
                    value_activation = tuple(int(x) for x in parameters.bounds.agent.value.activation)
                    learning_rate = tuple(float(x) for x in parameters.bounds.agent.learning_rate)
                    epochs = tuple(int(x) for x in parameters.bounds.agent.epochs)
                    epsilon = tuple(float(x) for x in parameters.bounds.agent.epsilon)
                    optimizer = BayesianOptimization(
                        f=opt_funct,
                        pbounds = {
                            'actor_input_layer_0': actor_input_layer_0,
                            'actor_input_layer_1': actor_input_layer_1,
                            'actor_output_layer_0': actor_output_layer_0,
                            'actor_output_layer_1': actor_output_layer_1,
                            'actor_ltsm_size': actor_ltsm_size,
                            'actor_activation': actor_activation,
                            'value_input_layer_0': value_input_layer_0,
                            'value_input_layer_1': value_input_layer_1,
                            'value_output_layer_0': value_output_layer_0,
                            'value_output_layer_1': value_output_layer_1,
                            'value_ltsm_size': value_ltsm_size,
                            'value_activation': value_activation,
                            'learning_rate': learning_rate,
                            'epochs': epochs,
                            'epsilon': epsilon
                        }
                    )
                    optimizer.maximize(
                        init_points=10,
                        n_iter=40
                    )
                    logger.info(optimizer.max)
                elif parameters.to_maximize == "environment":
                    early_penalty = tuple(int(x) for x in parameters.bounds.environment.early_penalty)
                    holes_penalty = tuple(float(x) for x in parameters.bounds.environment.holes_penalty)
                    height_penalty = tuple(float(x) for x in parameters.bounds.environment.height_penalty)
                    game_over_penalty = tuple(int(x) for x in parameters.bounds.environment.game_over_penalty)
                    line_single_reward = tuple(int(x) for x in parameters.bounds.environment.line_single_reward)
                    line_double_reward = tuple(int(x) for x in parameters.bounds.environment.line_double_reward)
                    line_triple_reward = tuple(int(x) for x in parameters.bounds.environment.line_triple_reward)
                    line_tetris_reward = tuple(int(x) for x in parameters.bounds.environment.line_tetris_reward)
                    block_placed_reward = tuple(float(x) for x in parameters.bounds.environment.block_placed_reward)
                    press_down_reward =  tuple(float(x) for x in parameters.bounds.environment.press_down_reward)
                    optimizer = BayesianOptimization(
                        f=opt_funct,
                        pbounds= {
                            'early_penalty': early_penalty,
                            'holes_penalty': holes_penalty,
                            'height_penalty': height_penalty,
                            'game_over_penalty': game_over_penalty,
                            'line_single_reward': line_single_reward,
                            'line_double_reward': line_double_reward,
                            'line_triple_reward': line_triple_reward,
                            'line_tetris_reward': line_tetris_reward,
                            'block_placed_reward': block_placed_reward,
                            'press_down_reward': press_down_reward
                        }
                    )
                    optimizer.maximize(
                        init_points=5,
                        n_iter=5
                    )
                elif parameters.to_maximize == "all":
                    actor_input_layer_0 = tuple(int(x) for x in parameters.bounds.agent.actor.input_layer_0)
                    actor_input_layer_1 = tuple(int(x) for x in parameters.bounds.agent.actor.input_layer_1)
                    actor_output_layer_0 = tuple(int(x) for x in parameters.bounds.agent.actor.output_layer_0)
                    actor_output_layer_1 = tuple(int(x) for x in parameters.bounds.agent.actor.output_layer_1)
                    actor_ltsm_size = tuple(int(x) for x in parameters.bounds.agent.actor.ltsm_size)
                    actor_activation = tuple(int(x) for x in parameters.bounds.agent.actor.activation)
                    value_input_layer_0 = tuple(int(x) for x in parameters.bounds.agent.value.input_layer_0)
                    value_input_layer_1 = tuple(int(x) for x in parameters.bounds.agent.value.input_layer_1)
                    value_output_layer_0 = tuple(int(x) for x in parameters.bounds.agent.value.output_layer_0)
                    value_output_layer_1 = tuple(int(x) for x in parameters.bounds.agent.value.output_layer_1)
                    value_ltsm_size = tuple(int(x) for x in parameters.bounds.agent.value.ltsm_size)
                    value_activation = tuple(int(x) for x in parameters.bounds.agent.value.activation)
                    learning_rate = tuple(float(x) for x in parameters.bounds.agent.learning_rate)
                    epochs = tuple(int(x) for x in parameters.bounds.agent.epochs)
                    epsilon = tuple(float(x) for x in parameters.bounds.agent.epsilon)
                    early_penalty = tuple(int(x) for x in parameters.bounds.environment.early_penalty)
                    holes_penalty = tuple(float(x) for x in parameters.bounds.environment.holes_penalty)
                    height_penalty = tuple(float(x) for x in parameters.bounds.environment.height_penalty)
                    game_over_penalty = tuple(int(x) for x in parameters.bounds.environment.game_over_penalty)
                    line_single_reward = tuple(int(x) for x in parameters.bounds.environment.line_single_reward)
                    line_double_reward = tuple(int(x) for x in parameters.bounds.environment.line_double_reward)
                    line_triple_reward = tuple(int(x) for x in parameters.bounds.environment.line_triple_reward)
                    line_tetris_reward = tuple(int(x) for x in parameters.bounds.environment.line_tetris_reward)
                    block_placed_reward = tuple(float(x) for x in parameters.bounds.environment.block_placed_reward)
                    press_down_reward =  tuple(float(x) for x in parameters.bounds.environment.press_down_reward)
                    optimizer = BayesianOptimization(
                        f=opt_funct,
                        pbounds = {
                            'actor_input_layer_0': actor_input_layer_0,
                            'actor_input_layer_1': actor_input_layer_1,
                            'actor_output_layer_0': actor_output_layer_0,
                            'actor_output_layer_1': actor_output_layer_1,
                            'actor_ltsm_size': actor_ltsm_size,
                            'actor_activation': actor_activation,
                            'value_input_layer_0': value_input_layer_0,
                            'value_input_layer_1': value_input_layer_1,
                            'value_output_layer_0': value_output_layer_0,
                            'value_output_layer_1': value_output_layer_1,
                            'value_ltsm_size': value_ltsm_size,
                            'value_activation': value_activation,
                            'learning_rate': learning_rate,
                            'epochs': epochs,
                            'epsilon': epsilon,
                            'early_penalty': early_penalty,
                            'holes_penalty': holes_penalty,
                            'height_penalty': height_penalty,
                            'game_over_penalty': game_over_penalty,
                            'line_single_reward': line_single_reward,
                            'line_double_reward': line_double_reward,
                            'line_triple_reward': line_triple_reward,
                            'line_tetris_reward': line_tetris_reward,
                            'block_placed_reward': block_placed_reward,
                            'press_down_reward': press_down_reward
                        }
                    )
                    optimizer.maximize(
                        init_points=10,
                        n_iter=40
                    )
                    logger.info(optimizer.max)
                elif parameters.to_maximize == "none":
                    logger.info("Manual testing activated")
                    result = opt_funct()
                    logger.info(f'Manual testing finished, average result: {result:.2f}')

                if parameters.to_maximize != "none":
                    optimizer.maximize(
                        init_points=rand_iter,
                        n_iter=iter
                    )
                    logger.info(optimizer.res)

            metrics.finish_training(optimizer.res)
        else:
            logger.info('No parameters found, playing game instead')
            play_game(main_screen=main_screen, clock=clock, speed=speed)
        pygame.quit()
    except Exception as ex:
        exit_code = 1
        logger.exception(ex)
    finally:
        logger.info("End of Petris Game! Code: %s", exit_code)
        
    return exit_code

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument("-s", "--speed", action="store", required=False, default=50, type=int,
                        help="The speed at which the tetris piece gets dropped. "
                        "Higher is faster. Default is 50.")
    parser.add_argument("-d", "--debug", action="store_true", required=False, default=False,
                        help="Displays the debug logs.")
    parser.add_argument("-p", "--parameters", action="store", required=False,type=str,
                        help="JSON file that contains the parameters for running the agent and environment.")
    parser.add_argument("-r", "--random-iterations",action="store", required=False, default=5, type=int,
                        help="The amount of random iterations you want the Bayesian Optimization to take. More can help diversify the exploration space.")
    parser.add_argument("-i", "--iterations",action="store", required=False, default=5, type=int,
                        help="The amount of iterations you want the Bayesian Optimization to take. More steps lead to better maximums.")
    
    args, _ = parser.parse_known_args()
    
    sys.exit(main(speed=args.speed, 
                  paramFile = args.parameters,
                  debug=args.debug,
                  rand_iter=args.random_iterations,
                  iter=args.iterations))
