from __future__ import division

import tensorflow as tf
import numpy as np 
import gym
import itertools
import collections
from database.examples.point_env import PointEnv
from scipy.stats import multivariate_normal

# Create the environment\
env = PointEnv()

#MIXING_WEIGHT = 0.5

N_HIDDEN = 25
#var = [[0.3, 0], [0, 0.3]]
var = [0.3, 0.3]
full_var = [[0.3, 0], [0, 0.3]]
l_rate_agent = 0.000025
l_rate_value = 0.00001 
l_rate_gate = 0.000025


def gauss_kl(mu_1, mu_2, var_1, var_2): 

	eps = 1e-6

	return 0.5 * ( (np.log( np.linalg.det(var_2) + eps) / (np.linalg.det(var_1) + eps) ) - 2 \
		+ np.trace(np.matmul(np.linalg.inv(var_2), var_1)) \
		+ tf.matmul(tf.matmul(mu_2 - mu_1, np.linalg.inv(var_2).astype(np.float32)), tf.transpose(mu_2 - mu_1)) )  


# Coordinated Exploration
class MultiAgent(): 

	def __init__(self, learning_rate=l_rate_agent): 

		self.reg_constant_1 = 0
		self.reg_constant_2 = 0
		with tf.variable_scope('agent1'): 

			# Define policy for agent 1
			self.lambda_1 = tf.placeholder(tf.float32, name='lambda_1')
			self.state_1 = tf.placeholder(tf.float32, (env.observation_space.shape[0], ), "state_1")
			self.action_1 = tf.placeholder(dtype=tf.float32, name="action_1")
			self.target_1 = tf.placeholder(dtype=tf.float32, name="target_1")
			self.ratio_1 = tf.placeholder(dtype=tf.float32, name="ratio_1")
			self.other_mean_1 = tf.placeholder(dtype=tf.float32, name="other_mean_1")

			# Loss for policy 1
			#state_one_hot_1 = tf.one_hot(self.state_1, int(env.observation_space.n))
			
			self.p1_full_1 = tf.layers.dense(
				inputs=tf.expand_dims(self.state_1, 0),
				units=N_HIDDEN,
				activation=tf.nn.sigmoid, 
				kernel_initializer=tf.truncated_normal_initializer)

			'''
			self.p1_full_2 = tf.layers.dense(
				inputs=self.p1_full_1,
				units=N_HIDDEN,
				activation=tf.nn.relu)
			'''

			# Output the mean of a multi-variate gaussian
			#self.output_layer = tf.layers.dense(inputs=self.full_2, units=2*env.action_space.shape[0])
			self.output_layer_1 = tf.layers.dense(inputs=self.p1_full_1, 
				units=env.action_space.shape[0], 
				kernel_initializer=tf.truncated_normal_initializer,
				activation=tf.tanh)

			mean_1 = self.output_layer_1
			#mean = tf.gather(tf.squeeze(self.output_layer), 0)
			#var = tf.gather(tf.squeeze(self.output_layer), 1)

			dist_1 = tf.distributions.Normal(loc=mean_1, scale=var)
			self.picked_action_prob_1 = dist_1.prob(self.action_1)

			self.KL_1 = gauss_kl(mu_1=self.other_mean_1, mu_2=mean_1, var_1=full_var, var_2=full_var) * (1 - self.lambda_1)

			self.loss_1 = - self.lambda_1 * tf.log(self.picked_action_prob_1) * self.ratio_1 * self.target_1 \
						  - self.reg_constant_1 * self.KL_1

			self.optimizer_1 = tf.train.AdamOptimizer(learning_rate=learning_rate)
			self.train_op_1 = self.optimizer_1.minimize(
				self.loss_1, global_step=tf.contrib.framework.get_global_step())

		# Define policy for agent 2 
		with tf.variable_scope('agent2'): 
			# Define policy for agent 2
			self.lambda_2 = tf.placeholder(tf.float32, name='lambda_2')
			self.state_2 = tf.placeholder(tf.float32, (env.observation_space.shape[0], ), "state_2")
			self.action_2 = tf.placeholder(dtype=tf.float32, name="action_2")
			self.target_2 = tf.placeholder(dtype=tf.float32, name="target_2")
			self.ratio_2 = tf.placeholder(dtype=tf.float32, name="ratio_2")
			self.other_mean_2 = tf.placeholder(dtype=tf.float32, name="other_mean_2")

			
			self.p2_full_1 = tf.layers.dense(
				inputs=tf.expand_dims(self.state_2, 0),
				units=N_HIDDEN,
				activation=tf.nn.sigmoid,
				kernel_initializer=tf.truncated_normal_initializer)

			'''
			self.p2_full_2 = tf.layers.dense(
				inputs=self.p2_full_1,
				units=N_HIDDEN,
				activation=tf.nn.relu)
			'''


			# Output the mean of a multi-variate gaussian
			#self.output_layer = tf.layers.dense(inputs=self.full_2, units=2*env.action_space.shape[0])
			self.output_layer_2 = tf.layers.dense(inputs=self.p2_full_1, 
				units=env.action_space.shape[0],
				kernel_initializer=tf.truncated_normal_initializer, 
				activation=tf.tanh)

			mean_2 = self.output_layer_2
			#mean = tf.gather(tf.squeeze(self.output_layer), 0)
			#var = tf.gather(tf.squeeze(self.output_layer), 1)

			dist_2 = tf.distributions.Normal(loc=mean_2, scale=var)
			self.picked_action_prob_2 = dist_2.prob(self.action_2)

			self.KL_2 = gauss_kl(mu_1=self.other_mean_2, mu_2=mean_2, var_1=full_var, var_2=full_var) * (1 - self.lambda_2)

			self.loss_2 = - self.lambda_2 * tf.log(self.picked_action_prob_2) * self.ratio_2 * self.target_2 \
						  - self.reg_constant_2 * self.KL_2

			self.optimizer_2 = tf.train.AdamOptimizer(learning_rate=learning_rate)
			self.train_op_2 = self.optimizer_2.minimize(
				self.loss_2, global_step=tf.contrib.framework.get_global_step())


	def train(self, agent_idx, state, target, action, ratio, other_dist, weight, sess=None): 
		'''
		Train policy (policy_idx) to improve mixture policy performance
		'''
		sess = sess or tf.get_default_session()
		if agent_idx == 0: 
			feed_dict = { self.state_1: state, self.target_1: target, self.action_1: action, \
						  self.ratio_1: ratio, self.other_mean_1: other_dist, self.lambda_1: weight}
			_, loss = sess.run([self.train_op_1, self.loss_1], feed_dict)
			return loss
		else: 
			feed_dict = { self.state_2: state, self.target_2: target, self.action_2: action, \
						  self.ratio_2: ratio, self.other_mean_2: other_dist, self.lambda_2: weight}
			_, loss = sess.run([self.train_op_2, self.loss_2], feed_dict)
			return loss			


	def sample(self, agent_idx, state, sess=None): 
		'''
		Sample from policy (policy_idx) 
		'''
		sess = sess or tf.get_default_session()

		if agent_idx == 0: 
			return sess.run(self.output_layer_1, { self.state_1: state })

		else:
			return sess.run(self.output_layer_2, { self.state_2: state })


# Joint value function trained by all the agents
class ValueFunction(): 

	def __init__(self, learning_rate=l_rate_value, scope='value'):
		with tf.variable_scope(scope): 
			self.state_value = tf.placeholder(tf.float32, (env.observation_space.shape[0], ), "state")
			self.returns = tf.placeholder(tf.float32, name="returns")

			self.v_full_1 = tf.layers.dense(
				inputs=tf.expand_dims(self.state_value, 0),
				units=N_HIDDEN,
				activation=tf.nn.relu)

			self.v_full_2 = tf.layers.dense(
				inputs=self.v_full_1,
				units=N_HIDDEN,
				activation=tf.nn.relu)

			self.v_output_layer = tf.layers.dense(inputs=self.v_full_2, units=1)

			self.value_loss = tf.squared_difference(self.v_output_layer, self.returns)

			self.optimizer_value = tf.train.AdamOptimizer(learning_rate=learning_rate)
			self.train_op_value = self.optimizer_value.minimize(self.value_loss, 
				global_step=tf.contrib.framework.get_global_step())

	def train(self, state, returns, sess=None): 
		sess = sess or tf.get_default_session()
		feed_dict = {self.state_value: state, self.returns: returns}
		_, loss = sess.run([self.train_op_value, self.value_loss], feed_dict)
		return loss

	def predict(self, state, sess=None): 
		sess = sess or tf.get_default_session()
		return sess.run(self.v_output_layer, {self.state_value: state})


class GatingFunction():
	def __init__(self, learning_rate=l_rate_gate): 

		with tf.variable_scope('gate'): 
			# Define policy for agent 1
			self.state_gate = tf.placeholder(tf.float32, (env.observation_space.shape[0], ), "gate_1")
			self.agent_action = tf.placeholder(dtype=tf.int32, name="agent_action")

			self.output_layer = tf.contrib.layers.fully_connected(
				inputs=tf.expand_dims(self.state_gate, 0),
				num_outputs=2,
				activation_fn=None,
				weights_initializer=tf.zeros_initializer)

			self.agent_probs = tf.squeeze(tf.nn.softmax(self.output_layer))
			self.picked_agent_prob = tf.gather(self.agent_probs, self.agent_action)

			
			self.gate_loss = -tf.log(self.picked_agent_prob)

			self.optimizer_gate = tf.train.AdamOptimizer(learning_rate=learning_rate)
			self.train_op_gate = self.optimizer_gate.minimize(
				self.gate_loss, global_step=tf.contrib.framework.get_global_step())

	def train(self, state, agent, sess=None): 
		sess = sess or tf.get_default_session()
		feed_dict = {self.state_gate: state, self.agent_action: agent}
		loss, _ = sess.run([self.gate_loss, self.train_op_gate], feed_dict)
		return loss

	def sample(self, state, sess=None): 
		sess = sess or tf.get_default_session()
		return sess.run(self.agent_probs, {self.state_gate: state})


def get_gauss_prob(mean, var, x):
	return multivariate_normal.pdf(x, mean, var)


def get_env_dim(env): 
	'''
	Return the dimensions of the state, action space of a Gym environment
	'''

	state_dim = env.observation_space.shape[0]
	action_dim = env.action_space.shape[0]

	return state_dim, action_dim


# Get dimensions of the environment state, action space
num_states, num_actions = get_env_dim(env)

# Parameters for training
num_episodes = 10000
num_agents = 2
discount_factor = 0.99
high_threshold = 0.1
low_threshold = -0.1

'''
agent_parameters = {'hidden_units_1':, 
					'hidden_units_2':, 
					'learning_rate': }

value_function_parameters = {'hidden_units_1':,
							 'hidden_units_2':,
							 'learning_rate': }
'''

tf.reset_default_graph()

global_step = tf.Variable(0, name="global_step", trainable=False)

Transition = collections.namedtuple("Transition", ["state", "action", "reward", "next_state", "done"])

multi_agent = MultiAgent()
gate = GatingFunction()
#joint_value_function = ValueFunction()

episode_rewards_1 = np.zeros((num_episodes, ))
episode_rewards_2 = np.zeros((num_episodes, ))

with tf.Session() as sess: 

	sess.run(tf.global_variables_initializer())

	# Train agents over multiple episodes
	for i_episode in range(num_episodes): 


		# Train each agent with one episode of experience
		for i_agent in range(num_agents): 
			
			episode = []

			# Reset the initial state
			state = env.reset()

			# Reset policy probability array
			current_policy_probs = []
			mixture_policy_probs = []

			# Reset weights array 
			weights = []

			# Collection transitions from one rollout of current agent 
			for t in itertools.count(): 

				# Take a step in the environment
				action_mean = np.ndarray.flatten(multi_agent.sample(agent_idx=i_agent, state=state))
				action = multivariate_normal.rvs(mean=action_mean, cov=var)

				current_policy_probs.append(get_gauss_prob(mean=action_mean, var=var, x=action))

				'''
				max_idx = np.argmax(np.abs(action))
				a_max = action_mean[max_idx]

				if a_max > high_threshold or a_max < low_threshold: 
					action_clipped = action / np.abs(a_max)
				'''

				#action_clipped = action 

				action_clipped = [np.max([np.min([action[0], high_threshold]), low_threshold]), 
						  np.max([np.min([action[1], high_threshold]), low_threshold])]
	
				#print(action_clipped)

				next_state, reward, done = env.step(action_clipped)

				if t > 30: 
					done = True

				if i_agent == 0: 
					
					my_weight = gate.sample(state)[0]
					
					'''
					if i_episode > 100: 
						performance = np.sum(episode_rewards_1[i_episode - 100 : i_episode - 1]) \
										/ np.sum(episode_rewards_1[i_episode - 100 : i_episode - 1] \
											+ episode_rewards_2[i_episode - 100 : i_episode - 1])

						my_weight = (my_weight + performance) * 0.5
					'''

					other_action_mean = np.ndarray.flatten(multi_agent.sample(agent_idx=1, state=state))
					mixture_policy_probs.append(my_weight * get_gauss_prob(mean=action_mean, var=var, x=action) + \
						(1 - my_weight) * get_gauss_prob(mean=other_action_mean, var=var, x=action))
					episode_rewards_1[i_episode] += reward
					weights.append(my_weight)



				else: 

					my_weight = gate.sample(state)[1]
					
					'''
					if i_episode > 100: 
						performance = np.sum(episode_rewards_2[i_episode - 100 : i_episode - 1]) \
										/ np.sum(episode_rewards_1[i_episode - 100 : i_episode - 1] \
											+ episode_rewards_2[i_episode - 100 : i_episode - 1])

						my_weight = (my_weight + performance) * 0.5
					'''

					other_action_mean = np.ndarray.flatten(multi_agent.sample(agent_idx=0, state=state))
					mixture_policy_probs.append(my_weight * get_gauss_prob(mean=action_mean, var=var, x=action) + \
						(1 - my_weight) * get_gauss_prob(mean=other_action_mean, var=var, x=action))
					episode_rewards_2[i_episode] += reward	
					weights.append(my_weight)


				# Record the transition tuple 
				transition = Transition(state=state, action=action, reward=reward, 
										next_state=next_state, done=done)

				# Append to the transitions from an episode
				episode.append(transition)

				if done: 
					break
				else: 
					state = next_state

			baseline = np.mean([cur_transition.reward for cur_transition in episode])

			# Train the agent using one episode of experience 
			for t, transition in enumerate(episode): 

				# Get discount return from state t onwards 
				discounted_return = np.sum([cur_transition.reward*discount_factor**i for i, cur_transition in enumerate(episode[t:])])
				
				# Get value estimate 
				#baseline = joint_value_function.predict(transition.state)
				'''
				if i_agent == 0: 
					baseline = value_function_1.predict(transition.state)
					td_target = transition.reward + value_function_1.predict(transition.next_state)
					value_function_1.train(state=transition.state, returns=td_target)
				else:
					baseline = value_function_2.predict(transition.state)
					td_target = transition.reward + value_function_2.predict(transition.next_state)
					value_function_2.train(state=transition.state, returns=td_target)
				'''
				'''
				td_target = transition.reward + joint_value_function.predict(transition.next_state)

				# Train joint value function 
				joint_value_function.train(state=transition.state, returns=td_target)
				'''
					
				# Train gating function
				gate.train(state=transition.state, agent=i_agent)

				# Calculate the advantage
				target = discounted_return - baseline

				if i_agent == 0: 
					other_mean = np.ndarray.flatten(multi_agent.sample(agent_idx=1, state=transition.state))

				else: 
					other_mean = np.ndarray.flatten(multi_agent.sample(agent_idx=0, state=transition.state))

				'''
				ratio = np.prod(np.divide(mixture_policy_probs[:t] + mixture_policy_probs[t+1:], \
										  current_policy_probs[:t] + current_policy_probs[t+1:]))
				'''

				ratio = np.prod(np.divide(mixture_policy_probs[t+1:], \
										  current_policy_probs[t+1:], dtype=np.float32))


				# Update the agent's policy 
				#print(transition.action, ratio, other_mean)
				multi_agent.train(agent_idx=i_agent, state=transition.state, target=target, \
					action=transition.action, ratio=ratio, other_dist=other_mean, weight=weights[t])

			monitor_epoch = 50 

			if i_episode % monitor_epoch == 0 and i_episode != 0 and i_agent == 0:
				print('----------------Episode: %d----------------' %i_episode) 

			# Print out the performance of each agent after 100 episodes
			if i_episode % monitor_epoch == 0 and i_episode != 0: 
				if i_agent == 0: 
					print("Average return from Agent %d: %f" %(i_agent, np.mean(episode_rewards_1[i_episode-monitor_epoch:i_episode])))
					#print(sess.run(tf.trainable_variables()[0]))

				else: 
					print("Average return from Agent %d: %f" %(i_agent, np.mean(episode_rewards_2[i_episode-monitor_epoch:i_episode])))
					#print(sess.run(tf.trainable_variables()[2]))

					'''
					mixed_rewards = np.zeros((100, ))


					performance = np.sum(episode_rewards_1[i_episode - 100 : i_episode - 1]) \
									/ np.sum(episode_rewards_1[i_episode - 100 : i_episode - 1] \
										+ episode_rewards_2[i_episode - 100 : i_episode - 1])


					for k in range(100):
						state = env.reset()

						for t in itertools.count(): 

							# Take a step in the environment
							action_probs_1 = multi_agent.sample(agent_idx=0, state=state)
							action_probs_2 = multi_agent.sample(agent_idx=1, state=state)

							my_weight = 0.5 * (performance + gate.sample(state)[0])

							mixed_probs = my_weight * action_probs_1 \
										+ (1 - my_weight) * action_probs_2

							action = np.random.choice(np.arange(num_actions), p=mixed_probs)

							next_state, reward, done, _ = env.step(action)

							mixed_rewards[k] += reward

							if done: 
								break
							else: 
								state = next_state	


					print("Average return of mixed policy: %f" %np.mean(mixed_rewards))
					'''














