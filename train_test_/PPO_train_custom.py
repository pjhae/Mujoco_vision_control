import gym
import torch as th
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from typing import Callable
from stable_baselines3.common.callbacks import EveryNTimesteps, CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.vec_env import VecFrameStack


def lin_schedule(initial_value: float, final_value: float) -> Callable[[float], float]:
    def func(progress_remaining: float) -> float:
        return progress_remaining * (initial_value - final_value) + final_value
    return func


date = "0626"
trial = "C"

checkpoint_on_event = CheckpointCallback(
    save_freq=1,
    save_path='./save_model_'+date+'/'+trial,
    verbose=2,
    name_prefix='Hexy_model_'+date+trial
)

event_callback = EveryNTimesteps(
    n_steps=int(1e5),  # every n_steps, save the model
    callback=checkpoint_on_event
)


class CustomCNN(BaseFeaturesExtractor):
    """
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 6):
        super(CustomCNN, self).__init__(observation_space, features_dim)
        # We assume CxHxW images (channels first)
        # Re-ordering will be done by pre-preprocessing or wrapper
        n_input_channels = observation_space.shape[0]
        # print(observation_space.shape)
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=(4,32), stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=(2,8), stride=2, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        with th.no_grad():
            n_flatten = self.cnn(
                th.as_tensor(observation_space.sample()[None]).float()
            ).shape[1]

        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.linear(self.cnn(observations))

policy_kwargs = dict(
    features_extractor_class=CustomCNN,
    features_extractor_kwargs=dict(features_dim=64),
)


env = make_vec_env("Hexy-v5", n_envs=1)
env = VecFrameStack(env, n_stack=3,  channels_order = "first")

model = PPO("CnnPolicy", env=env , device = 'cuda', policy_kwargs=policy_kwargs, verbose=1,  tensorboard_log='./hexy_tb_log_'+ date,
            learning_rate=lin_schedule(3e-4, 3e-6), clip_range=lin_schedule(0.3, 0.1),
            n_epochs=10, ent_coef=1e-4, batch_size=256*4, n_steps=256)

model.learn(total_timesteps=20000000, 
		callback=event_callback,  # every n_steps, save the model.
		tb_log_name='hexy_tb_'+date+trial
		# ,reset_num_timesteps=False   # if you need to continue learning by loading existing model, use this option.
		
		)


model.save("Hexy_model")
del model # remove to demonstrate saving and loading
model = PPO.load("Hexy_model")

obs = env.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, done, info = env.step(action)
    
    env.render()
