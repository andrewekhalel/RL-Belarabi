import gym
from QLearning import QLAgent
from helpers import discretize
import numpy as np

######## CONSTANTS ########
EPISODES = 2000

# due to discretization of continous state
N_STATES =  20 * 20

# push-left, no-push and push-right
N_ACTIONS = 3

# evaluation trials
TRIALS = 10
######## CONSTANTS ########

env = gym.make('MountainCar-v0')
agent = QLAgent(env=env,
				n_states=N_STATES,
				n_actions=N_ACTIONS,
				lr=0.5,
				discount=0.95)

# Train agent
agent.learn(episodes=EPISODES)

# Evaluate
success = 0
for tr in range(TRIALS):
	state = env.reset()
	t=0
	while True:
		env.render()
		action = agent.act(discretize(state))
		state, reward, done, info = env.step(action)
		t +=1
		if done:
			print("Trial {} finished after {} timesteps".format(tr,t))
			if t < 200:
				success += 1
			break
print ("Success: %d/%d"%(success,TRIALS))

env.close()