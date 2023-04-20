"""Script containing the random petris agent."""

import logging
from typing import List

import pygame
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from pygame.time import Clock
from pygame.surface import Surface
from pygame.event import Event
from tf_agents.environments.py_environment import PyEnvironment
from tf_agents.trajectories import TimeStep

from src.scenes.scenes import GameScene, Scenes, TitleScene
from src.game_runner.game_runner import render_active_scene
from src.evaluation_metrics.evaluation_metrics import save_line_graph

logger = logging.getLogger(__name__) 
   

def play_random_agent(env: PyEnvironment, main_screen: Surface, clock: Clock, speed: int, num_episodes: int = 5) -> None:
    """
    Runs multiple episodes the game scene for the agent to run.
    
    NOTE: Player an interfere the agent by pressing the keys.

    Args:
        env (PyEnvironment): _description_
        main_screen (Surface): _description_
        clock (Clock): _description_
        speed (int): _description_
    """

    rewards = []
    cumulative_rewards = 0

    # Runs multiple games without quiting the pygame
    for episode in range(1, num_episodes + 1):
        logger.info("Starting Episode %s", episode)
        episode_reward = 0
        
        # Display episode
        pygame.display.set_caption(f"Episode {episode}")
        
        time_step = env.reset()
        
        events: List[Event] = []
        while not time_step.is_last():
            events = pygame.event.get() 
            Scenes.active_scene.process_input(events=events)
            
            # Press escape to stop the entire game.            
            for event in events:
                if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    logger.info("Stopping Agent...")
                    return
            
            render_active_scene(main_screen=main_screen, clock=clock, speed=speed)
            
            # [1] == (1, )
            random_action = tf.random.uniform([1], 0, 4, dtype=tf.int32)
            time_step: TimeStep = env.step(action=random_action)
            cumulative_rewards += time_step.reward
            episode_reward += time_step.reward
            
            # If it switches to the title scene that means the game episode is over.
            # Recreate GameScene and run the next episode.
            if isinstance(Scenes.active_scene, TitleScene):
                logger.info("End of Episode %s", episode)
                break

        rewards.append(episode_reward)

    logger.info("End of Random Agent")
    logger.info("Cumulative Reward: %s", cumulative_rewards)
    logger.info("Mean Reward: %s", np.array(rewards).mean())
    logger.info("Standard Deviation: %s", np.array(rewards).std())
    logger.info("Rewards: %s", rewards)

    plt.title("Random Agent")
    plt.xlabel("Number of Episodes")
    plt.ylabel("Reward")
    plt.plot(rewards, marker='o')
    plt.show()

    save_line_graph(title="", data=rewards)
