import numpy as np
import random
import keras
from collections import deque
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.optimizers import SGD , Adam
import cv2
import matplotlib.pyplot as plt
import os
import json
from tqdm import tqdm

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

class DQNAgent:
	'''
	Deep Q-Networks Agent
	'''
	def __init__(self,
				env,
				num_actions,
				lr,
				discount,
				batch_size=32,
				mem_size=100000,
				prefetch_size=10000,
				eps=1.,
				eps_decay=0.99,
				eps_min=0.02):
		'''
		Agent Initialization

		:param env: OpenAI gym environment
		:param lr: Learning rate (a.k.a alpha)
		:param num_actions: number of available actions
		:param discount: Discount factor (a.k.a gamma)
		:param batch_size: batch size for training
		:param mem_size: experience replay buffer size
		:param prefetch_size: size of buffer before training starts
		:param eps: epsilon (exploration vs exploitation)
		:param eps_decay: epsilon decay rate
		:param eps_min: minimum possible value of epsilon
		'''
		self.env = env
		self.n_actions = num_actions
		state = self.env.reset()
		self.state_shape = self.preprocess(state).shape
		print ("state shape:",self.state_shape)
		self.discount = discount

		# epsilon
		self.eps = eps
		self.eps_decay_rate = eps_decay
		self.eps_min = eps_min

		# training parameters
		self.lr = lr
		self.batch_size = batch_size
		self.prefetch_size = prefetch_size

		# memory
		self.mem_size = mem_size
		self.mem = deque(maxlen=self.mem_size)
		
		# network		
		self.model = self._build_model()


	def _build_model(self):
		'''
		Build the network using keras
		'''
		model = Sequential()
		model.add(Conv2D(32, (8, 8), strides=(4, 4), padding='same',input_shape=self.state_shape[1:]))
		model.add(Activation('relu'))
		model.add(Conv2D(64, (4, 4), strides=(2, 2), padding='same'))
		model.add(Activation('relu'))
		model.add(Conv2D(64, (3, 3), strides=(1, 1), padding='same'))
		model.add(Activation('relu'))
		model.add(Flatten())
		model.add(Dense(512))
		model.add(Activation('relu'))
		model.add(Dense(self.n_actions))
	   
		adam = Adam(lr=self.lr)
		model.compile(loss='mse',optimizer=adam)			
			
		return model
		
	def preprocess(self,state):
		'''
		Reshaping state to fit out model

		:param state: Original state
		'''
		return np.concatenate(state,axis=-1)[np.newaxis,:,:,:]
		
	def act(self,state,random_en=False):
		'''
		Return best action based on current Q-states

		:param state: current state
		:param random_en: enable random actions?
		:returns: Chosen action number
		'''
		if len(state) == 4:
			state = self.preprocess(state)

		if random_en:
			eps = self.eps
		else:
			eps = 0

		if random.random() <= eps:
			return random.randrange(self.n_actions) +2
		else:
			Qs = self.model.predict(state)[0]
			return np.argmax(softmax(Qs)) +2
	
	def remember(self,state, action, reward, next_state, done):
		'''
		Save to experience replay buffer
		'''
		self.mem.append((state, action, reward, next_state, done))

	def replay(self):
		'''
		Main experience replay routine
		'''
		minibatch = random.sample(self.mem, self.batch_size)

		states = []
		targets = []
		for state, action, reward, next_state, done in minibatch:
			Q = reward
			if not done:
				next_Qs = self.model.predict(next_state)
				Q = reward + self.discount*np.amax(next_Qs)
			target_Qs = self.model.predict(state)
			target_Qs[0][action-2] = Q
			states.append(state[0])
			targets.append(target_Qs[0])

		self.model.fit(np.stack(states,axis=0),np.stack(targets,axis=0),epochs=1,verbose=0)

	
	def train(self,timesteps,weights_file=None,save_step=10000):
		'''
		Train agent

		:param timesteps: timesteps to train agent with
		:param weights_file: file path to save trained weights
		:param save_step: when to save model
		'''
		state = self.env.reset()
		state = self.preprocess(state)

		for t in tqdm(range(1,timesteps+1)):
			action = self.act(state,random_en=True)
			
			# epsilon decay
			if t % 1000 == 0:
				self.eps = max(self.eps_min,self.eps*self.eps_decay_rate)

			next_state,reward,done,_ = self.env.step(action)
			next_state = self.preprocess(next_state)
			self.remember(state, action, reward, next_state, done)
			state = next_state

			# check if buffer is filled enough to do replay
			if len(self.mem) > self.prefetch_size:
				self.replay()

			# reset environment if reached terminal state
			if done:
				state = self.env.reset()
				state = self.preprocess(state)

			# save model
			if t % save_step == 0 and weights_file is not None:
				self.save_model(weights_file)

	def load_model(self,weights_file):
		'''
		Load model weights
		:param weights_file: weights file paths
		'''
		self.model.load_weights(weights_file+".h5")
		print ('Model loaded ...')

	def save_model(self,weights_file):
		'''
		Save model weights
		:param weights_file: weights file paths
		'''
		self.model.save_weights(weights_file+".h5", overwrite=True)
		with open(weights_file+".json", "w") as outfile:
			json.dump(self.model.to_json(), outfile)
		print('Model saved ...')