import numpy as np
from helpers import discretize

class QLAgent:
	'''
	Q-Learning Agent
	'''
	def __init__(self,env,n_actions,n_states,lr,discount):
		'''
		Agent Initialization

		:param env: OpenAI gym environment
		:param n_action: Number of available actions
		:param n_states: Number of available states
		:param lr: Learning rate (a.k.a alpha)
		:param discount: Discount factor (a.k.a gamma)
		'''
		self.n_actions = n_actions
		self.n_states = n_states
		self.lr = lr
		self.discount = discount
		self.env = env
		
		self.Qs = np.zeros((n_states,n_actions))


	def act(self,state,episode=None):
		'''
		Return best action based on current Q-states

		:param state: Disctretized current state
		:param episode: Current episode number (optional) - Only matters if learning
		:returns: Chosen action number
		'''		
		if episode is not None:
			# add random noise to Qs to explore at the beginning
			noise = np.random.randn(1, self.n_actions)*(1./(episode+1))**(0.75)
			return np.argmax(self.Qs[state,:] + noise)

		return np.argmax(self.Qs[state,:])

	def updateQ(self,s,a,r,s_dash):
		'''
		Update Q values based on previous step

		:param s: Current discretized state
		:param a: Taken action
		:param r: Gained reward
		:param s_dash: Next (new) discretized state
		'''
		self.Qs[s,a] += self.lr * (r + self.discount*max(self.Qs[s_dash,:])- self.Qs[s,a]) 

	def learn(self,episodes,visualize=False):
		'''
		Learn best Q-values

		:param episodes: Number of episodes to learn
		:param visualize: if True, render environment when learning
		'''
		for ep in range(episodes):
			state = self.env.reset()
			t = 0
			while True:
				if visualize:
					self.env.render()
				action = self.act(discretize(state),episode=ep)
				new_state, reward, done, info = self.env.step(action)
				self.updateQ(discretize(state),action,reward,discretize(new_state))
				state = new_state
				t += 1
				if done:
					print("Episode {} finished after {} timesteps".format(ep,t))
					break
