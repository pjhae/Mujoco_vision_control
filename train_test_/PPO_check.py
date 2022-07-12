import gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.vec_env import VecFrameStack


date = "0709"
trial = "A"
steps = "3440000"

# 0701 / E / 7800000
# 0704 / D / 3600000  C map v6 원핫 -0.9
# 0627 / C / 2200000 직선 주로 v5 -1.2

## Make gym environment ##

#env = make_vec_env("MiniCheetah-v1", n_envs=1)
env = make_vec_env("Hexy-v5", n_envs=1)
# env = gym.make("Hexy-v6")
# env = VecFrameStack(env, n_stack=3,  channels_order = "first")
# env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.)

## Path ##

save_path='./save_model_'+date+'/'+trial+'/'

model = PPO.load(save_path+"Hexy_model_"+date+trial+"_"+steps+"_steps")

obs = env.reset()

## Rendering ##

while True:
    action, _states = model.predict(obs)
    obs, rewards, done, info = env.step(action)
    env.render()

#for i in range(12000):
#    action, _states = model.predict(obs)
#    line = []
#    line.append(str(action[0][0]))
#    for i in range(1, 12):
#        line.append(',')
#        line.append(str(action[0][i]))
#    line.append('\n')
#    line = ''.join(line)
#    obs, rewards, dones, info = env.step(action)
#    env.render()


