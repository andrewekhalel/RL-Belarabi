import gym
from DQN import DQNAgent
import numpy as np
import random
import time
from atari_wrappers import *

######## CONSTANTS ########
TRAIN_STEPS = 400000
NUM_ACTIONS = 2
LR = 0.0001
GAMMA = 0.99

# evaluation trials
TRIALS = 10
######## CONSTANTS ########



def wrap_dqn(env):
    """Apply a common set of wrappers for Atari games."""
    assert 'NoFrameskip' in env.spec.id
    env = EpisodicLifeEnv(env)
    env = NoopResetEnv(env, noop_max=30)
    env = MaxAndSkipEnv(env, skip=4)
    if 'FIRE' in env.unwrapped.get_action_meanings():
        env = FireResetEnv(env)
    env = ProcessFrame84(env)
    env = FrameStack(env, 4)
    env = ClippedRewardsWrapper(env)
    return env

env = wrap_dqn(gym.make('PongNoFrameskip-v4'))

agent = DQNAgent(env=env,num_actions=NUM_ACTIONS,lr=LR,discount=GAMMA)



# Load model
# agent.load_model(weights_file="snaps/model")

# Train agent
agent.train(TRAIN_STEPS,weights_file="snaps/model")

# Evaluate
success = 0
for tr in range(TRIALS):
	state = env.reset()
	t=0
	acc_r = 0
	while True:
		env.render()
		action = agent.act(state)
		state, reward, done, _ = env.step(action)
		acc_r += reward
		t +=1
		if done:
			print("Trial {} finished after {} timesteps".format(tr,t))
			if acc_r > 0:
				success += 1
			break
print ("Success: %d/%d"%(success,TRIALS))

env.close()