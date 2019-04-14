from mdptoolbox import mdp
import numpy as np
import matplotlib.pyplot as plt

cooperate = 0
defect = 1

states = [cooperate, defect]
actions = [cooperate, defect]

# reward_matrix-> reward for (state, action) pairs
def run_tft(reward_matrix, probability, graph_string, algoIsValueIteration):
	print('\n\n%s\n\n' % 'Value Iteration' if algoIsValueIteration else 'Policy Iteration')

	# transition probabilities for (action, state current, state next)
	transition_matrix = {
			(cooperate, cooperate, cooperate): probability, 
			(cooperate, cooperate, defect): 1-probability, 
			(cooperate, defect, cooperate): probability, 
			(cooperate, defect, defect): 1-probability,
			(defect, cooperate, cooperate): 1-probability, 
			(defect, cooperate, defect): probability, 
			(defect, defect, cooperate): 1-probability, 
			(defect, defect, defect): probability
	}

	GAMMA_LIST = [val/100 for val in range(5, 100, 5)]

	reward_arr = list(reward_matrix.items())
	reward_arr.sort()
	# first state is cooperate, first action is cooperate
	R = np.array([val[1] for val in reward_arr]).reshape((2,2))


	transition_arr = list(transition_matrix.items())
	transition_arr.sort()
	# first element is transition matrix for action cooperate; second is for action defect
	P = np.array([val[1] for val in transition_arr]).reshape((2, 2, 2))

	# reward for (state, action) pairs
	state_0_values = []
	state_1_values = []

	state_0_policy = []
	state_1_policy = []

	for gamma in GAMMA_LIST:

		if algoIsValueIteration:
			tft = mdp.ValueIteration(P, R, gamma)
		else:
			tft = mdp.PolicyIteration(P, R, gamma)
		tft.run()

		state_0_values.append(tft.V[0])
		state_1_values.append(tft.V[1])

		state_0_policy.append(tft.policy[0])
		state_1_policy.append(tft.policy[1])

		print('\nGamma: %f' % gamma)
		print('Iterations: %d' % tft.iter)
		print('Time to solve MDP: %.3f' % tft.time)
		print('Policy: {Cooperate: %s, Defect: %s}' % (state_0_policy[-1], state_1_policy[-1]))
		print('Value: {Cooperate: %.3f, Defect: %.3f}\n' % (state_0_values[-1], state_1_values[-1]))

	plt.grid()

	plt.plot(GAMMA_LIST, state_0_values, '-', color="r")
	plt.plot(GAMMA_LIST, state_1_values, '-', color="g")

	plt.scatter(GAMMA_LIST, state_0_values, c=['b' if val == 0 else 'y' for val in state_0_policy])
	plt.scatter(GAMMA_LIST, state_1_values, c=['b' if val == 0 else 'y' for val in state_1_policy])

	plt.xlabel('Gamma Values')
	plt.ylabel('Utility Values')
	plt.title('Tit for Tat Results %s %s' % (graph_string, 'VI' if algoIsValueIteration else 'PI'))
	plt.legend(['Cooperate', 'Defect'], loc="best")
	plt.savefig("TFT %s %s.png" % (graph_string, 'VI' if algoIsValueIteration else 'PI'))
	plt.close()

run_tft({(cooperate, cooperate): -1, (cooperate, defect): 0, (defect, cooperate): -9, (defect, defect): -6}, 1, graph_string="Default", algoIsValueIteration=True)
run_tft({(cooperate, cooperate): -1, (cooperate, defect): 0, (defect, cooperate): -9, (defect, defect): -6}, 0.7, graph_string="Stochastic", algoIsValueIteration=True)
run_tft({(cooperate, cooperate): -1, (cooperate, defect): 5, (defect, cooperate): -9, (defect, defect): -6}, 1, graph_string="(c, d) = 5", algoIsValueIteration=True)

run_tft({(cooperate, cooperate): -1, (cooperate, defect): 0, (defect, cooperate): -9, (defect, defect): -6}, 1, graph_string="Default", algoIsValueIteration=False)
run_tft({(cooperate, cooperate): -1, (cooperate, defect): 0, (defect, cooperate): -9, (defect, defect): -6}, 0.7, graph_string="Stochastic", algoIsValueIteration=False)
run_tft({(cooperate, cooperate): -1, (cooperate, defect): 5, (defect, cooperate): -9, (defect, defect): -6}, 1, graph_string="(c, d) = 5", algoIsValueIteration=False)









