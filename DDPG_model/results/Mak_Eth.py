import os
import sys
from os.path import dirname, abspath
sys.path.append(dirname(dirname(abspath(__file__))))
import gym
import numpy as np
from gym import spaces
import copy
from datetime import datetime
from talib import *
import pandas as pd
from agents.policy_gradient_agents.PPO import PPO
from agents.actor_critic_agents.DDPG import DDPG
from agents.actor_critic_agents.SAC import SAC
from agents.actor_critic_agents.TD3 import TD3
from agents.actor_critic_agents.A2C import A2C
from agents.DQN_agents.Dueling_DDQN import Dueling_DDQN
from agents.actor_critic_agents.SAC_Discrete import SAC_Discrete
from agents.actor_critic_agents.A3C import A3C
from agents.policy_gradient_agents.PPO import PPO
from agents.DQN_agents.DDQN import DDQN
from agents.DQN_agents.DDQN_With_Prioritised_Experience_Replay import DDQN_With_Prioritised_Experience_Replay
from agents.DQN_agents.DQN import DQN
from agents.DQN_agents.DQN_With_Fixed_Q_Targets import DQN_With_Fixed_Q_Targets
from agents.Trainer import Trainer
from utilities.data_structures.Config import Config


from environments.Market_Environments.MakEnv import MarketEnv
from environments.Market_Environments.MakEnv import EnvConfig


#coding:utf-8

import sys,os,re
import time


config = Config()
config.seed = 1
# height = 15
# width = 15
# random_goal_place = False
# num_possible_states = (height * width) ** (1 + 1*random_goal_place)
# embedding_dimensions = [[num_possible_states, 20]]


df = pd.read_csv('../environments/Market_Environments/Bitfinex_ETHUSD_1h.csv')
data = df.to_numpy()[:,[1,7,3,4,5,6]]
data = np.flip(data,axis=0)
for i in range(data.shape[0]):
    data[i, 0] = time.mktime(time.strptime(data[i, 0], '%Y/%m/%d %H:%M'))

data = (np.array(data,dtype=np.float64))
Envconfig = EnvConfig()
# 1)config.train_data
# 2)config.eval_data
# 3)config.mode
# 4)config.use_data_aug
Envconfig.train_data = data[:int(0.8*data.shape[0])]
# Envconfig.eval_data = data[:int(0.8*data.shape[0])]
Envconfig.eval_data = data[int(0.8*data.shape[0]):int(0.9*data.shape[0])]
Envconfig.mode = 'train'
Envconfig.use_data_aug = False
Envconfig.min_and_max = np.load('../environments/Market_Environments/min_and_max.npy', allow_pickle=True).item()

# Envconfig.test_data = data[int(0.8*data.shape[0]):int(0.9*data.shape[0])]

env = MarketEnv(Envconfig)
# env.reset()
# action = np.array([0.4,0.6])
# obs, reward, done, info = env.step(action)
 
config.environment = env

# config.num_episodes_to_run = 200
# config.file_to_save_data_results = "Data_and_Graphs/Mak_Eth.pkl"
# config.file_to_save_results_graph = "Data_and_Graphs/Mak_Eth.png"
# config.show_solution_score = False
# config.visualise_individual_results = False
# config.visualise_overall_agent_results = True
# config.standard_deviation_results = 1.0
# config.runs_per_agent = 3
# config.use_GPU = False
# config.overwrite_existing_results_file = False
# config.randomise_random_seed = True
# config.save_model = False

# config.hyperparameters = {
#     "DQN_Agents": {
#         # "linear_hidden_units": [30, 10],
#         # "learning_rate": 0.01,
#         # "buffer_size": 40000,
#         # "batch_size": 256,
#         # "final_layer_activation": "None",
#         # "columns_of_data_to_be_embedded": [0],
#         # "embedding_dimensions": embedding_dimensions,
#         # "batch_norm": False,
#         # "gradient_clipping_norm": 5,
#         # "update_every_n_steps": 1,
#         # "epsilon_decay_rate_denominator": 10,
#         # "discount_rate": 0.99,
#         # "learning_iterations": 1,
#         # "tau": 0.01,
#         # "exploration_cycle_episodes_length": None,
#         # "learning_iterations": 1,
#         # "clip_rewards": False

#         "learning_rate": 0.01,
#         "batch_size": 256,
#         "buffer_size": 40000,
#         "epsilon": 1.0,
#         "epsilon_decay_rate_denominator": 1,
#         "discount_rate": 0.99,
#         "tau": 0.01,
#         "alpha_prioritised_replay": 0.6,
#         "beta_prioritised_replay": 0.1,
#         "incremental_td_error": 1e-8,
#         "update_every_n_steps": 1,
#         "linear_hidden_units": [30, 15],
#         "final_layer_activation": "None",
#         "batch_norm": False,
#         "gradient_clipping_norm": 0.7,
#         "learning_iterations": 1,
#         "clip_rewards": False
#     },

#     # "SNN_HRL": {
#     #     "SKILL_AGENT": {
#     #         "num_skills": 20,
#     #         "regularisation_weight": 1.5,
#     #         "visitations_decay": 0.9999,
#     #         "episodes_for_pretraining": 300,
#     #         "batch_size": 256,
#     #         "learning_rate": 0.001,
#     #         "buffer_size": 40000,
#     #         "linear_hidden_units": [20, 10],
#     #         "final_layer_activation": "None",
#     #         "columns_of_data_to_be_embedded": [0, 1],
#     #         "embedding_dimensions": [embedding_dimensions[0],
#     #                                  [20, 6]],
#     #         "batch_norm": False,
#     #         "gradient_clipping_norm": 2,
#     #         "update_every_n_steps": 1,
#     #         "epsilon_decay_rate_denominator": 500,
#     #         "discount_rate": 0.999,
#     #         "learning_iterations": 1,
#     #         "tau": 0.01,
#     #         "clip_rewards": False
#     #     },

#     #     "MANAGER": {
#     #         "timesteps_before_changing_skill": 6,
#     #         "linear_hidden_units": [10, 5],
#     #         "learning_rate": 0.01,
#     #         "buffer_size": 40000,
#     #         "batch_size": 256,
#     #         "final_layer_activation": "None",
#     #         "columns_of_data_to_be_embedded": [0],
#     #         "embedding_dimensions": embedding_dimensions,
#     #         "batch_norm": False,
#     #         "gradient_clipping_norm": 5,
#     #         "update_every_n_steps": 1,
#     #         "epsilon_decay_rate_denominator": 50,
#     #         "discount_rate": 0.99,
#     #         "learning_iterations": 1,
#     #         "tau": 0.01,
#     #         "clip_rewards": False

#     #     }

#     # },

#     # "Actor_Critic_Agents": {

#     #     "learning_rate": 0.005,
#     #     "linear_hidden_units": [20, 10],

#     #     "columns_of_data_to_be_embedded": [0],
#     #     "embedding_dimensions": embedding_dimensions,
#     #     "final_layer_activation": ["SOFTMAX", None],
#     #     "gradient_clipping_norm": 5.0,
#     #     "discount_rate": 0.99,
#     #     "epsilon_decay_rate_denominator": 50.0,
#     #     "normalise_rewards": True,
#     #     "clip_rewards": False

#     # },


#     # "DIAYN": {

#     #     "num_skills": 5,
#     #     "DISCRIMINATOR": {
#     #         "learning_rate": 0.01,
#     #         "linear_hidden_units": [20, 10],
#     #         "columns_of_data_to_be_embedded": [0],
#     #         "embedding_dimensions": embedding_dimensions,
#     #     },

#     #     "AGENT": {
#     #         "learning_rate": 0.01,
#     #         "linear_hidden_units": [20, 10],
#     #     }
#     # },


#     # "HRL": {
#     #     "linear_hidden_units": [10, 5],
#     #     "learning_rate": 0.01,
#     #     "buffer_size": 40000,
#     #     "batch_size": 256,
#     #     "final_layer_activation": "None",
#     #     "columns_of_data_to_be_embedded": [0],
#     #     "embedding_dimensions": embedding_dimensions,
#     #     "batch_norm": False,
#     #     "gradient_clipping_norm": 5,
#     #     "update_every_n_steps": 1,
#     #     "epsilon_decay_rate_denominator": 400,
#     #     "discount_rate": 0.99,
#     #     "learning_iterations": 1,
#     #     "tau": 0.01

#     # }


# }

# if __name__== '__main__':


#     AGENTS = [DDQN, Dueling_DDQN, DQN, DQN_With_Fixed_Q_Targets,
#               DDQN_With_Prioritised_Experience_Replay, A2C, PPO, A3C ]
#     trainer = Trainer(config, AGENTS)
#     trainer.run_games_for_agents()

config.num_episodes_to_run = 450
config.file_to_save_data_results = './numerical_results/DDPG.pkl'
config.file_to_save_results_graph = './numerical_results/DDPG.png'
config.dir_to_save_models = './numerical_results'

config.show_solution_score = False
config.visualise_individual_results = False
config.visualise_overall_agent_results = True
config.standard_deviation_results = 1.0
config.runs_per_agent = 3
config.use_GPU = False
config.overwrite_existing_results_file = False
config.randomise_random_seed = True
config.save_model = True


config.hyperparameters = {
    "Policy_Gradient_Agents": {
            "learning_rate": 0.05,
            "linear_hidden_units": [30, 15],
            "final_layer_activation": "TANH",
            "learning_iterations_per_round": 10,
            "discount_rate": 0.9,
            "batch_norm": False,
            "clip_epsilon": 0.2,
            "episodes_per_learning_round": 10,
            "normalise_rewards": True,
            "gradient_clipping_norm": 5,
            "mu": 0.0,
            "theta": 0.15,
            "sigma": 0.2,
            "epsilon_decay_rate_denominator": 1,
            "clip_rewards": False
        },

    "Actor_Critic_Agents": {
            "Actor": {
                "learning_rate": 0.0003,
                "linear_hidden_units": [64, 64],
                "final_layer_activation": "sigmoid",
                "batch_norm": False,
                "tau": 0.005,
                "gradient_clipping_norm": 0.5,
                "initialiser": "Xavier"
            },

            "Critic": {
                "learning_rate": 0.0003,
                "linear_hidden_units": [64, 64],
                "final_layer_activation": None,
                "batch_norm": False,
                "buffer_size": 1000000,
                "tau": 0.005,
                "gradient_clipping_norm": 0.5,
                "initialiser": "Xavier"
            },

        "min_steps_before_learning": 1000, #for SAC only
        "batch_size": 256,
        "discount_rate": 0.99,
        "mu": 0.0,  # for O-H noise
        "theta": 0.15,  # for O-H noise
        "sigma": 0.25,  # for O-H noise
        "action_noise_std": 0.1,  # for TD3
        "action_noise_clipping_range": 0.2,  # for TD3
        "update_every_n_steps": 20,
        "learning_updates_per_learning_session": 10,
        "automatically_tune_entropy_hyperparameter": True,
        "entropy_term_weight": None,
        "add_extra_noise": True,
        "do_evaluation_iterations": True,
        "clip_rewards": False

    }

}

if __name__ == "__main__":
    AGENTS = [DDPG]
    trainer = Trainer(config, AGENTS)
    trainer.run_games_for_agents()
