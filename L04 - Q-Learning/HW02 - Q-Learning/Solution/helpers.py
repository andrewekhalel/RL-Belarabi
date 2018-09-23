import numpy as np
import pandas as pd

######## CONSTANTS ########
N_BINS = 20
######## CONSTANTS ########

# bins for each input
position_bins = pd.cut([-1.2, 0.6], bins=N_BINS, retbins=True)[1][1:-1]
velocity_bins = pd.cut([-0.07, 0.07], bins=N_BINS, retbins=True)[1][1:-1]

def build_state(features):
	'''
	Convert from bins numbers to a universal state number

	:param features: Bin number for each input feature
	:returns: Universal state number
	'''
	state_no = 0
	for i,feat in enumerate(features):
		state_no += (N_BINS**i) * (feat-1)
	return state_no

def to_bin(value, bins):
	'''
	Map value to corresponding bin
	
	:param value: Value to map
	:param bins: Available bins
	:returns: Bin number for passed value
	'''
	return np.digitize(x=[value], bins=bins)[0]

def discretize(state):
	'''
	Find correpsonding state number in a discrete state space

	:param state: Continuous state
	:returns: Universal state number
	'''
	position,\
	velocity = state

	return build_state([to_bin(position, position_bins),
						 to_bin(velocity, velocity_bins)])