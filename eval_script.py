import pickle
import torch
from rlkit.torch.sac.policies import MakeDeterministic, TanhGaussianPolicy
import gym
from rlkit.envs import make_env
from gym.wrappers import ResizeObservation, GrayScaleObservation, FlattenObservation
# from ae.autoencoder import load_ae
import numpy as np
import gym_donkeycar.envs 


with open('./data/qrsac-donkey-generated-roads-normal-iqn-neutral/qrsac_donkey-generated-roads_normal-iqn-neutral_2025_04_27_16_11_13_0000--s-0/params.pkl', 'rb') as f:
    state_dict = torch.load(f)

# Build a mini Donkey env (64×64 gray → flat) to get the right dims
env_raw = make_env('donkey-generated-roads-v0')
env_pre = ResizeObservation(env_raw, (64, 64))
env_pre = GrayScaleObservation(env_pre, keep_dim=True)
env_pre = FlattenObservation(env_pre)
obs_dim = env_pre.observation_space.low.size   # should be 4096
action_dim = env_pre.action_space.low.size      # should be 2
target_policy = TanhGaussianPolicy(
    obs_dim=obs_dim,
    action_dim=action_dim,
    hidden_sizes=[256, 256, 256, 256, 256],
    dropout_probability=0.1,
)

target_policy.load_state_dict(state_dict["trainer/policy"])
#donkeycar
# ae_path = "/home/pipelines/pipeline2/aae-train-donkeycar/logs/ae-32_1704492161_best.pkl"
# ae = load_ae(ae_path)
env = env_pre
env.seed(0)

obs=env.reset()
#donkeycar
# obs = ae.encode_from_raw_image(np.squeeze(obs[:, :, ::-1]))

return_ep=0
step_count=0
done = False
while done== False:
    action = target_policy.get_actions(obs, True)
    # print(f"action = {action}")
    action=action.flatten()
    state, reward, done, info = env.step(action)
    return_ep+=reward
    step_count+=1
    #donkeycar
    # obs = ae.encode_from_raw_image(np.squeeze(state[:, :, ::-1]))
    env.render()

obs=env.reset()
#donkeycar
# obs = ae.encode_from_raw_image(np.squeeze(obs[:, :, ::-1]))

print(return_ep, step_count)


