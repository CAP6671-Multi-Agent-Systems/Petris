"""Script containing the code for the REINFORCE petris Agent"""

from __future__ import absolute_import, division, print_function

import logging
from typing import List

import matplotlib as plt
import tensorflow as tf
from pandas import DataFrame,concat
import numpy as np
import reverb
import pygame

from tf_agents.agents.reinforce import reinforce_agent
from tf_agents.drivers import py_driver
from pygame.time import Clock
from pygame.surface import Surface
from pygame.event import Event
from tf_agents.networks import actor_distribution_network
from tf_agents.policies import py_tf_eager_policy
from tf_agents.policies import epsilon_greedy_policy
from tf_agents.environments.tf_py_environment import TFPyEnvironment
from tf_agents.replay_buffers import reverb_replay_buffer
from tf_agents.replay_buffers import reverb_utils
from tf_agents.specs import tensor_spec
from tf_agents.utils import common

from src.params.parameters import Parameters
from src.metrics.metrics import Metrics
from src.custom_driver.petris_driver import PetrisDriver
from src.scenes.scenes import GameScene, Scenes, TitleScene
from src.game_runner.game_runner import render_active_scene
from src.petris_environment.petris_environment import PetrisEnvironment
from src.checkpointer.checkpointer import create_checkpointer
from src.policy_saver.policy_saver import TFPolicySaver

logger = logging.getLogger(__name__) 

def create_replay_buffer(agent: reinforce_agent.ReinforceAgent, replay_buffer_length: int = 100000):
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
    initial_time_step = env.reset()
    driver.run(main_screen, clock, speed, epoch, iteration, initial_time_step)

# Metrics and evaluation function
def compute_avg_return(env: TFPyEnvironment, policy, num_episodes, main_screen, clock, speed, epoch, iteration, agent):

    total_return = 0.0

    for _ in range(num_episodes):
        pygame.display.set_caption(f"EVALUATION | Iteration {iteration+1} | {agent} | Epoch {epoch+1} | Episode {_+1}")
        keyboard_events : List[Event] = []
        time_step = env.reset()
        episode_return = 0.0

        while not time_step.is_last():
            Scenes.active_scene.process_input(events=keyboard_events)
            keyboard_events = pygame.event.get()

            action_step = policy.action(time_step)
            #logger.info("Manual steps (avg return)")
            time_step = env.step(action_step.action)
            episode_return += time_step.reward
            render_active_scene(main_screen=main_screen, clock=clock, speed=speed)
        total_return += episode_return

    avg_return = total_return / num_episodes
    return avg_return.numpy()[0]

def create_reinforce(env: TFPyEnvironment, parameters: Parameters, train_step_counter: tf.Tensor) -> reinforce_agent.ReinforceAgent:
    # Actor network 
    actor_net = actor_distribution_network.ActorDistributionNetwork(
        env.observation_spec(),
        env.action_spec(),
        fc_layer_params=tuple(int(x) for x in parameters['layers'])
    )

    # NOTE: .001 lr was the example used by the docs
    optimizer = tf.keras.optimizers.Adam(learning_rate=parameters['learning_rate'])

    # Keeps track 
    train_step_counter = tf.Variable(0)

    agent = reinforce_agent.ReinforceAgent(
        env.time_step_spec(),
        env.action_spec(),
        actor_network=actor_net,
        optimizer=optimizer,
        normalize_returns=True,
        train_step_counter=train_step_counter
    )

    agent.initialize()

    return agent

def train_reinforce(main_screen: Surface, clock: Clock, speed: int, metrics: Metrics, parameters: Parameters, type: str,**inputs) -> float:
    params = {
        'params': {}
    }
    if type == "agent":
        params['params']['agent'] = {
            'layers': tuple([int(inputs['layer_0']),int(inputs['layer_1'])]),
            'learning_rate': inputs['learning_rate'],
            'epochs': parameters.params.agent['epochs'],
            'epsilon': inputs['epsilon'],
            'num_eval_episodes': parameters.params.agent['num_eval_episodes'],
            'eval_interval': parameters.params.agent['eval_interval'],
            'collect_num_episodes': parameters.params.agent['collect_num_episodes'],
        }
    elif type == "enviornment":
        params['params']['enviornment'] = {
            'early_penalty': inputs['early_penalty'],
            'holes_penalty': inputs['holes_penalty'],
            'height_penalty': inputs['height_penalty'],
            'game_over_penalty': inputs['game_over_penalty'],
            'line_reward': [inputs['line_single_reward'],inputs['line_double_reward'],inputs['line_triple_reward'],inputs['line_tetris_reward']],
            'block_placed_reward': inputs['block_placed_reward'],
            'press_down_reward': inputs['press_down_reward']
        }
    elif type == "all":
        params['params']['agent'] = {
            'layers': tuple([int(inputs['layer_0']),int(inputs['layer_1'])]),
            'learning_rate': inputs['learning_rate'],
            'epochs': parameters.params.agent['epochs'],
            'epsilon': inputs['epsilon'],
            'num_eval_episodes': parameters.params.agent['num_eval_episodes'],
            'eval_interval': parameters.params.agent['eval_interval'],
            'collect_num_episodes': parameters.params.agent['collect_num_episodes'],
        }
        params['params']['enviornment'] = {
            'early_penalty': inputs['early_penalty'],
            'holes_penalty': inputs['holes_penalty'],
            'height_penalty': inputs['height_penalty'],
            'game_over_penalty': inputs['game_over_penalty'],
            'line_reward': [inputs['line_single_reward'],inputs['line_double_reward'],inputs['line_triple_reward'],inputs['line_tetris_reward']],
            'block_placed_reward': inputs['block_placed_reward'],
            'press_down_reward': inputs['press_down_reward']
        }
    #logger.info(params)
    petris_environment = PetrisEnvironment(parameters=params)
    train_enivronment = TFPyEnvironment(environment=petris_environment)
    eval_environment = TFPyEnvironment(environment=petris_environment)
    
    params = params['params']['agent']

    global_step = tf.compat.v1.train.get_or_create_global_step()

    # Init the actor network, optimizer, and agent 
    reinforce_agent = create_reinforce(env=train_enivronment, parameters=params, train_step_counter=global_step)
    logger.info("Agent Created")

    # Init Replay Buffer
    replay_buffer, rb_observer = create_replay_buffer(agent=reinforce_agent)
    logger.info("Replay Buffer Created")

    reinforce_agent.train = common.function(reinforce_agent.train)

    # Reset the train step
    reinforce_agent.train_step_counter.assign(0)

    # checkpoint = create_checkpointer(name="reinforce", 
    #                                  agent=reinforce_agent, 
    #                                  replay_buffer=replay_buffer, 
    #                                  global_step=global_step,
    #                                  max_to_keep=10)
    # policy_saver = TFPolicySaver(name="reinforce", agent=reinforce_agent)

    # checkpoint.initialize_or_restore()

    # Evaluate the policy before training
    logger.info("Evaluating policy before training")
    
    avg_return =  compute_avg_return(eval_environment, reinforce_agent.policy, params['num_eval_episodes'], main_screen, clock, speed, 0, metrics._iteration, "Reinforce")
    loss = 0.00
    output_data = DataFrame(data=[[0,avg_return,loss,0]], columns=['epoch','return','loss','lines_cleared'])

    logger.info("Running for %s epochs", params['epochs'])

    for i in range(params['epochs']):
        logger.info("Running Epoch: %s", i)
        avg_return = -1
        loss = 0.00

        logger.info("Collecting Episode: %s", i)
        # Save episodes to the replay buffer
        collect_episode(
            petris_environment, 
            reinforce_agent.collect_policy, 
            observers=[rb_observer,metrics.metrics_observer()], 
            parameters=params, 
            main_screen=main_screen, 
            clock=clock, 
            speed=speed, 
            epoch=i,  
            iteration=metrics._iteration,
            agent="Reinforce"
        )
        
        logger.info("Gathering Trajectories")
        # Update the agent's network using the buffer data
        iterator = iter(replay_buffer.as_dataset(sample_batch_size=1))
        trajectories, _ = next(iterator)
        train_loss = reinforce_agent.train(experience=trajectories)

        replay_buffer.clear()

        logger.info("Checking Step")
        # Keeps track of how many times the agent has been trained
        step = reinforce_agent.train_step_counter.numpy()

        loss = train_loss.loss.numpy()
        if step % params['eval_interval'] == 0 and step != 0:
            avg_return = compute_avg_return(eval_environment, reinforce_agent.policy, params['num_eval_episodes'], main_screen, clock, speed, i, metrics._iteration, "Reinforce")
            logger.info('step = {0} | Average Return = {1} | loss = {2}'.format(step, avg_return,loss))

        logger.info("Saving Dataframe")
        append = DataFrame(data=[[i+1,avg_return,loss,metrics.metrics_observer().lines_placed]], columns=['epoch','return','loss','lines_cleared'])
        output_data = concat([output_data,append], ignore_index=True)
    
    metrics.finish_iteration(output_data)
    returns = output_data[output_data['return'] != -1]
    return returns['return'].mean()