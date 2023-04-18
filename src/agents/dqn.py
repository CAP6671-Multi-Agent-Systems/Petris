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
from tf_agents.policies.tf_policy import TFPolicy
from tf_agents.replay_buffers import reverb_utils
from tf_agents.trajectories import trajectory
from tf_agents.specs import tensor_spec
from tf_agents.utils import common
from tf_agents.utils.common import function
from src.params.parameters import Parameters

import pygame
import tensorflow as tf
from tensorflow import keras
from pygame.time import Clock
from pandas import DataFrame,concat
from pygame.surface import Surface
from tf_agents.environments.tf_py_environment import TFPyEnvironment

from src.scenes.scenes import GameScene, Scenes, TitleScene
from src.game_runner.game_runner import render_active_scene
from src.petris_environment.petris_environment import PetrisEnvironment
from src.custom_driver.petris_driver import PetrisDriver
from src.metrics.metrics import Metrics

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


def create_dqn(env: TFPyEnvironment, parameters, train_step_counter: tf.Tensor) -> dqn_agent.DqnAgent:
    """_summary_

    Args:
        env (TFPyEnvironment): _description_

    Returns:
        dqn_agent.DqnAgent: _description_
    """
    #investigate input_dim
    logger.info(parameters)
    q_net = sequential.Sequential([
        keras.layers.Dense(
            parameters['layer_0'], 
            activation=keras.activations.relu if parameters['activation'] == 'relu' else keras.activations.gelu, 
        ),
        keras.layers.Dense(
            parameters['layer_1'], 
            activation=keras.activations.relu if parameters['activation'] == 'relu' else keras.activations.gelu, 
        ),
        tf.keras.layers.Flatten(), 
        keras.layers.Dense(4, activation="linear")
    ])
    
    agent = dqn_agent.DqnAgent(
        env.time_step_spec(),
        env.action_spec(),
        q_network=q_net,
        optimizer=keras.optimizers.Adam(learning_rate=parameters['learning_rate']),
        td_errors_loss_fn=common.element_wise_squared_loss,
        train_step_counter=train_step_counter,
        epsilon_greedy= parameters['epsilon']
    )
    
    agent.initialize()
    
    q_net.summary()
    
    return agent


def collect_episode(env: PetrisEnvironment, policy, observers, parameters, main_screen, clock, speed, epoch, iteration, agent):
    driver = PetrisDriver(
        env, 
        py_tf_eager_policy.PyTFEagerPolicy(
            policy, 
            use_tf_function=True
        ),
        observers,
        max_episodes=parameters['collect_num_episodes'],
        agent=agent
    )
    time_step = env.reset()
    policy_state = policy.get_initial_state(batch_size=1)
    driver.run(main_screen, clock, speed, epoch, iteration, time_step, policy_state)

# Metrics and evaluation function
def compute_avg_return(env: TFPyEnvironment, policy: TFPolicy, num_episodes: int, main_screen: Surface, clock: Clock, speed: int, epoch: int, iteration: int, agent: dqn_agent.DqnAgent) -> float:
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


def train_dqn(main_screen: Surface, clock: Clock, speed: int, metrics: Metrics, parameters, type: str,**inputs) -> float:
    """Creates and Trains a DQN agent."""
    logger.info(type)
    params = parameters
    if type == "agent":
        params.params.agent = {
            'layer_0': int(inputs['layer_0']),
            'layer_1': int(inputs['layer_1']),
            'learning_rate': inputs['learning_rate'],
            'activation': "gelu" if int(inputs['activation']) < 0.5 else "relu",
            'learning_rate': inputs['learning_rate'],
            'epochs': parameters.params.agent['epochs'],
            'epsilon': inputs['epsilon'],
            'num_eval_episodes': parameters.params.agent['num_eval_episodes'],
            'eval_interval': parameters.params.agent['eval_interval'],
            'collect_num_episodes': parameters.params.agent['collect_num_episodes'],
            'save_interval': parameters.params.agent['save_interval']
        }
    elif type == "enviornment":
        params.params.enviornment = {
            'early_penalty': inputs['early_penalty'],
            'holes_penalty': inputs['holes_penalty'],
            'height_penalty': inputs['height_penalty'],
            'game_over_penalty': inputs['game_over_penalty'],
            'line_reward': [inputs['line_single_reward'],inputs['line_double_reward'],inputs['line_triple_reward'],inputs['line_tetris_reward']],
            'block_placed_reward': inputs['block_placed_reward'],
            'press_down_reward': inputs['press_down_reward']
        }

    petris_environment = PetrisEnvironment(parameters=params)
    train_env = TFPyEnvironment(environment=petris_environment)
    eval_env = TFPyEnvironment(environment=petris_environment)

    params = params.params.agent

    global_step = tf.compat.v1.train.get_or_create_global_step()

    # Set up agent 
    agent = create_dqn(env=train_env, parameters=params, train_step_counter=global_step)

    replay_buffer, rb_observer = create_replay_buffer(agent=agent)

    iterator = iter(replay_buffer.as_dataset(sample_batch_size=1))
    
    agent.train = function(agent.train)
    agent.train_step_counter.assign(0)

    #checkpoint = create_checkpointer(name="dqn", agent=agent, replay_buffer=replay_buffer, global_step=global_step, max_to_keep=params['save_interval'])
    #policy_saver = TFPolicySaver(name="dqn", agent=agent)

    #checkpoint.initialize_or_restore()

    avg_return =  compute_avg_return(eval_env, agent.policy, params['num_eval_episodes'], main_screen, clock, speed, 0, metrics._iteration, "PPO")
    loss = 0.00
    output_data = DataFrame(data=[[0,avg_return,loss,0]], columns=['epoch','return','loss','lines_cleared'])


    logger.info("Running for %s epochs", params['epochs'])

    for epoch in range(params['epochs']):
        logger.info("Running Episode: %s", epoch)
        avg_return = -1
        loss = 0.00
        
        collect_episode(
            petris_environment, 
            agent.collect_policy, 
            observers=[rb_observer,metrics.metrics_observer()], 
            parameters=params, 
            main_screen=main_screen, 
            clock=clock, 
            speed=speed, 
            epoch=epoch, 
            iteration=metrics._iteration, 
            agent="DQN"
        )

        trajectories, _ = next(iterator)
        trajectories = tf.nest.map_structure(lambda x: x[:, :2, ...], trajectories)
        train_loss = agent.train(experience=trajectories)

        replay_buffer.clear()
        step = agent.train_step_counter.numpy()
        

        # if step % params['save_interval'] == 0  and step != 0:
        #     checkpoint.save(global_step=global_step)
        #     policy_saver.save()

        loss = train_loss.loss.numpy()

        if step % params['eval_interval'] == 0 and step != 0:
            avg_return = compute_avg_return(eval_env, agent.policy, params['num_eval_episodes'], main_screen, clock, speed, epoch, metrics._iteration, "PPO")
            logger.info('Iteration = {} | Loss = {} | Average Return = {}'.format(epoch, loss, avg_return))

        append = DataFrame(data=[[epoch+1,avg_return,loss,metrics.metrics_observer().lines_placed]], columns=['epoch','return','loss','lines_cleared'])
        output_data = concat([output_data,append], ignore_index=True)

    metrics.finish_iteration(output_data)
    returns = output_data[output_data['return'] != -1]
    return returns['return'].mean()
    