"""Script containing the code for the DQN petris Agent"""

from __future__ import absolute_import, division, print_function

import sys
import logging
from typing import List, Tuple

import reverb
from tf_agents.agents.dqn import dqn_agent
from tf_agents.drivers import py_driver
from tf_agents.environments import suite_gym
from tf_agents.environments import tf_py_environment
from tf_agents.replay_buffers.tf_uniform_replay_buffer import TFUniformReplayBuffer
from tf_agents.replay_buffers.reverb_utils import ReverbAddEpisodeObserver
from tf_agents.eval import metric_utils
from tf_agents.metrics import tf_metrics
from tf_agents.networks import sequential
from tf_agents.policies import py_tf_eager_policy
from tf_agents.policies import random_tf_policy
from tf_agents.policies import epsilon_greedy_policy
from tf_agents.replay_buffers import reverb_replay_buffer
from tf_agents.replay_buffers import reverb_utils
from tf_agents.trajectories import trajectory
from tf_agents.specs import tensor_spec
from tf_agents.utils import common
from tf_agents.utils.common import function
from tf_agents.drivers.dynamic_step_driver import DynamicStepDriver
from src.params.parameters import Parameters

import pygame
import tensorflow as tf
from tensorflow import keras
from pygame.time import Clock
from pygame.surface import Surface
from tf_agents.environments import utils
from tf_agents.environments.tf_py_environment import TFPyEnvironment

from src.scenes.scenes import GameScene, Scenes, TitleScene
from src.game_runner.game_runner import render_active_scene
from src.petris_environment.petris_environment import PetrisEnvironment
from src.custom_driver.petris_driver import PetrisDriver

from src.checkpointer.checkpointer import create_checkpointer
from src.policy_saver.policy_saver import TFPolicySaver

logger = logging.getLogger(__name__)
 
# TODO: Evaluate Average return and see if it is performing well during training 


def create_replay_buffer(agent: dqn_agent.DdqnAgent, replay_buffer_length: int = 100000):
    table_name = "uniform_table"

    replay_buffer_signature = tensor_spec.from_spec(
        agent.collect_data_spec
    )
    replay_buffer_signature = tensor_spec.add_outer_dim(
        replay_buffer_signature
    )

    table = reverb.Table(
        table_name,
        max_size=replay_buffer_length,
        sampler=reverb.selectors.Uniform(),
        remover=reverb.selectors.Fifo(),
        rate_limiter=reverb.rate_limiters.MinSize(1),
        signature=replay_buffer_signature
    )

    reverb_server = reverb.Server([table])

    replay_buffer = reverb_replay_buffer.ReverbReplayBuffer(
        agent.collect_data_spec,
        table_name=table_name,
        sequence_length=None, 
        local_server=reverb_server
    )

    rb_observer = reverb_utils.ReverbAddEpisodeObserver(
        replay_buffer.py_client,
        table_name,
        replay_buffer_length
    )

    return replay_buffer, rb_observer


def create_dqn(env: TFPyEnvironment, train_step_counter: tf.Variable = tf.Variable(0)) -> dqn_agent.DqnAgent:
    """_summary_

    Args:
        env (TFPyEnvironment): _description_

    Returns:
        dqn_agent.DqnAgent: _description_
    """
    q_net = sequential.Sequential([
        keras.layers.Dense(
            100, 
            activation=keras.activations.relu, 
            kernel_initializer=tf.keras.initializers.VarianceScaling(scale=2.0, mode='fan_in', distribution='truncated_normal')
        ),
        keras.layers.Dense(
            50, 
            activation=keras.activations.relu, 
            kernel_initializer=tf.keras.initializers.VarianceScaling(scale=2.0, mode='fan_in', distribution='truncated_normal')
        ),
        tf.keras.layers.Flatten(), 
        keras.layers.Dense(4, activation="linear")
    ])
    
    agent = dqn_agent.DqnAgent(
        env.time_step_spec(),
        env.action_spec(),
        q_network=q_net,
        optimizer=keras.optimizers.Adam(learning_rate=0.2),
        td_errors_loss_fn=common.element_wise_squared_loss,
        train_step_counter=train_step_counter
    )
    
    agent.initialize()
    
    q_net.summary()
    
    return agent


def collect_episode(env: PetrisEnvironment, policy, observers, parameters, main_screen, clock, speed, epoch, iteration, agent):
    driver = PetrisDriver(
        env, 
        py_tf_eager_policy.PyTFEagerPolicy(
            epsilon_greedy_policy.EpsilonGreedyPolicy(
                policy=policy,
                epsilon=parameters['epsilon']
            ), use_tf_function=True
        ),
        observers,
        max_episodes=parameters['collect_num_episodes'],
        agent=agent
    )
    time_step = env.reset()
    policy_state = policy.get_initial_state(batch_size=1)
    driver.run(main_screen, clock, speed, epoch, iteration, time_step, policy_state)

# Metrics and evaluation function
def compute_avg_return(env: TFPyEnvironment, policy: TFPolicy, num_episodes: int, main_screen: Surface, clock: Clock, speed: int, epoch: int, iteration: int, agent: PPOAgent) -> float:
    total_return = 0.0

    for _ in range(num_episodes):
        pygame.display.set_caption(f"EVALUATION | {agent} | Iteration {iteration+1} | Epoch {epoch+1} | Episode {_+1}")
        keyboard_events : List[Event] = []
        time_step = env.reset()
        episode_return = 0.0
        policy_state = policy.get_initial_state(env.batch_size)

        while not time_step.is_last():
            Scenes.active_scene.process_input(events=keyboard_events)
            keyboard_events = pygame.event.get()
            action_step = policy.action(time_step=time_step,policy_state=policy_state)
            time_step = env.step(action_step.action)
            episode_return += time_step.reward
            render_active_scene(main_screen=main_screen, clock=clock, speed=speed)
        total_return += episode_return

    avg_return = total_return / num_episodes
    return avg_return.numpy()[0]


def train_dqn(main_screen: Surface, clock: Clock, speed: int, parameters: Parameters, iteration: int = 0) -> None:
    """Creates and Trains a DQN agent."""

    petris_environment = PetrisEnvironment(parameters=parameters)
    train_env = TFPyEnvironment(environment=petris_environment)

    num_iterations = parameters.iterations.num_iterations
    parameters = parameters.params.agent

    global_step = tf.compat.v1.train.get_or_create_global_step()

    # Set up agent 
    agent = create_dqn(env=train_env, train_step_counter=global_step)

    replay_buffer, rb_observer = create_replay_buffer(agent=agent)
    
    agent.train = function(agent.train)
    agent.train_step_counter.assign(0)

    checkpoint = create_checkpointer(name="dqn", 
                                     agent=agent, 
                                     replay_buffer=replay_buffer, 
                                     global_step=global_step,
                                     max_to_keep=num_iterations)
    policy_saver = TFPolicySaver(name="dqn", agent=agent)

    checkpoint.initialize_or_restore()

    avg_return =  compute_avg_return(eval_env, agent.policy, params['num_eval_episodes'], main_screen, clock, speed, 0, metrics._iteration, "PPO")
    loss = 0.00
    output_data = DataFrame(data=[[0,avg_return,loss,0]], columns=['epoch','return','loss','lines_cleared'])


    logger.info("Running for %s epochs", parameters.epochs)

    for epoch in range(parameters.epochs):
        logger.info("Running Episode: %s", epoch)
        
        collect_episode(
            petris_environment, 
            agent.collect_policy, 
            rb_observer=rb_observer, 
            parameters=parameters, 
            main_screen=main_screen, 
            clock=clock, 
            speed=speed, 
            epoch=epoch, 
            iteration=iteration, 
            agent="DQN"
        )

        logger.info("Collecting Episode Information")
        iterator = iter(replay_buffer.as_dataset(sample_batch_size=1))
        trajectories, _ = next(iterator)
        trajectories = tf.nest.map_structure(lambda x: x[:, :2, ...], trajectories)
        logger.info("We are here")
        train_loss = agent.train(experience=trajectories)
        logger.info("Agent trained")

        replay_buffer.clear()
        step = agent.train_step_counter.numpy()
        
        if step % parameters.log_interval == 0:
            losses.append(train_loss.loss.numpy())
            logger.info('step = {0}: loss = {1}'.format(step, train_loss.loss))

        if step % parameters.save_interval == 0  and step != 0:
            checkpoint.save(global_step=global_step)
            policy_saver.save()
        
    
    logger.info("Finishing Training...")


def play_dqn_agent(env: TFPyEnvironment, main_screen: Surface, clock: Clock, speed: int, num_episodes: int = 5) -> None:
    """
    Runs multiple episodes the game scene for the agent to run.
    
    NOTE: Player an interfere the agent by pressing the keys.

    Args:
        env (PyEnvironment): _description_
        main_screen (Surface): _description_
        clock (Clock): _description_
        speed (int): _description_
    """

    cumulative_reward = 0
    
    agent: dqn_agent.DqnAgent = create_dqn(env=env)
    policy = agent.policy


    # Runs multiple games without quiting the pygame
    for episode in range(1, num_episodes + 1):
        logger.info("Starting Episode %s", episode)
        
        # Display episode
        pygame.display.set_caption(f"Agent - Episode {episode}")
        
        time_step = env.reset()
        
        keyboard_events: List[Event] = []
        while not time_step.is_last():
            keyboard_events = pygame.event.get() 
            Scenes.active_scene.process_input(events=keyboard_events)
            
            # Press escape to stop the entire game.            
            for event in keyboard_events:
                if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    logger.info("Stopping Agent...")
                    return
            
            render_active_scene(main_screen=main_screen, clock=clock, speed=speed)
            
            # NOTE: At this point, we have already defined the environment
            # in main() in petris.py 

            # Validate our environment using a random policy for 5 eps
            action = policy.action(time_step=time_step)
            env.step(action=action)
            cumulative_reward += time_step.reward
            
            # If it switches to the title scene that means the game episode is over.
            # Recreate GameScene and run the next episode.
            if isinstance(Scenes.active_scene, TitleScene):
                logger.info("End of Episode %s", episode)
                break
    
    logger.info("Cumulative Reward: %s", cumulative_reward)
    