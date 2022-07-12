import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env
from stable_baselines3.common.env_util import make_vec_env
import gym

env = gym.make("Hexy-v5")

while(True):

    env.reset_model()
    env.render()
    sim_contact = env.sim.data.contact
