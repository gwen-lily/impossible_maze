from bedevere.markov import *
from typing import Tuple
import pylab
import matplotlib.pyplot as plt


def is_stochastic(P: np.ndarray, precision: float = arithmetic_precision) -> bool:
	"""Checks that a given transition matrix is square and probability-complete"""

	m, n = P.shape
	return m == n and all([math.fabs(sum(P[i, :]) - 1) < precision for i in range(n)])


class CategoricalSquareMatrix:
	"""Data structure to allow for intuitive access of categorical state matrices"""

	def __init__(self, matrix: np.ndarray, states: list):
		m, n = matrix.shape
		assert m == n and m == len(states)

		self.state_dict = {}
		self.matrix = matrix

		for index, state in enumerate(states):
			self.state_dict[state] = index

	def value(self, state_1, state_2):
		i, j = (self.state_dict[state_1], self.state_dict[state_2])
		return self.matrix[i, j]

	def set_value(self, state_1, state_2, val):
		i, j = (self.state_dict[state_1], self.state_dict[state_2])
		self.matrix[i, j] = val


def markov_generator(indents: int) -> AbsorbingMarkovChain:
	"""Generates an AbsorbingMarkovChain with categorical states and an absorbing transition matrix for an n-indent maze

	Parameters:
		indents (int): Left indents in the maze before the finish is reached

	Returns:
		_ (AbsorbingMarkovChain): Markov bussy
	"""

	state_indicators = ('a', 'b')     # tuple of string values that indicate the direction of entry
	forward, backward = state_indicators
	transient_states = [(0, backward)]
	absorbing_states = [(indents + 1, forward)]

	for n in range(1, indents + 1):

		for indicator in state_indicators:
			transient_states.append((n, indicator))

	transient_states.pop()    # final intersection before end of maze cannot be entered from secondary direction

	t = len(transient_states)
	r = len(absorbing_states)
	states = []

	for trans_state in transient_states:
		states.append(trans_state)

	for abs_state in absorbing_states:
		states.append(abs_state)

	P = CategoricalSquareMatrix(np.zeros(shape=(t+r, t+r)), states)

	# edge case
	state_1, state_2 = (states[0], (1, forward))
	P.set_value(state_1, state_2, 1)
	P.set_value(states[-1], states[-1], 1)

	# general cases
	for i in range(1, t+r):
		position_i, direction_i = states[i]

		for j in range(t+r):
			position_j, direction_j = states[j]

			position_tuple = (position_i, position_j)
			direction_tuple = (direction_i, direction_j)

			if position_j - position_i == 1:

				if all(d == forward for d in direction_tuple):
					P.matrix[i, j] = 5/8

				elif direction_i == backward and direction_j == forward:
					P.matrix[i, j] = 1/8

			elif position_i - position_j == 1:

				if all(d == backward for d in direction_tuple):
					P.matrix[i, j] = 7/8

				elif direction_i == forward and direction_j == backward:
					P.matrix[i, j] = 3/8

	Q = P.matrix[:t, :t]
	R = P.matrix[:t, t:]

	# ideally this doesn't need to exist but as the package exists right now states must by np.ndarray
	state_dict = {}
	states_array = np.empty(t+r)

	for index, (key, val) in enumerate(P.state_dict.items()):
		states_array[val] = index
		state_dict[index] = key

	MazeMarkovChain = AbsorbingMarkovChain(Q, R, states_array)
	assert is_stochastic(MazeMarkovChain.transition_matrix)

	return MazeMarkovChain


def get_expected_steps(indents: int) -> float:
	MazeMarkovChain = markov_generator(indents)
	starting_distribution = np.zeros(len(MazeMarkovChain.states))
	starting_distribution[0] = 1

	return MazeMarkovChain.expected_steps[0]


def plotxy(x, y):
	fig = plt.figure()
	ax = fig.add_subplot(1, 1, 1)
	pure_data, = ax.plot(x, y)
	ax.set_yscale('log')
	plt.xlabel('Indents')
	plt.ylabel('Expected steps')
	plt.show()


def main(indents: int):

	expected_steps = np.empty(indents)

	for i in range(1, indents+1):
		expected_steps[i-1] = get_expected_steps(i)

	x = np.arange(1, indents+1)
	y = expected_steps

	plotxy(x, y)


if __name__ == '__main__':
	maximum_indents = 100
	main(maximum_indents)
