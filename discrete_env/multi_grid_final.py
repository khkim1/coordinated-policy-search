from __future__ import division

import tensorflow as tf
import numpy as np 
import gym
import itertools
import collections
from grid_env import GridEnv
from lib import plotting
from matplotlib import pyplot as plt


env = GridEnv()
obs_dim, act_dim, grid_size = env.get_dimensions()

class SingleAgent(): 

    def __init__(self, learning_rate=0.01, scope='agent1'): 

        self.reg_constant = 1 #0.0005 for hard env, 0.0002 for easy 
        with tf.variable_scope(scope): 

            # Define policy for agent 1
            self.gate = tf.placeholder(tf.float32, name='gate')
            self.state = tf.placeholder(tf.int32, [], "state_1")
            self.action = tf.placeholder(dtype=tf.int32, name="action_1")
            self.target = tf.placeholder(dtype=tf.float32, name="target_1")
            self.is_ratio = tf.placeholder(dtype=tf.float32, name="ratio_1")
            self.mix_prob = tf.placeholder(dtype=tf.float32, name="mix_prob_1")
            
            # Loss for policy 1
            state_one_hot = tf.one_hot(self.state, obs_dim)
            
            self.output_layer = tf.contrib.layers.fully_connected(
                inputs=tf.expand_dims(state_one_hot, 0),
                num_outputs=act_dim,
                activation_fn=None,
                weights_initializer=tf.truncated_normal_initializer)

            self.action_probs = tf.squeeze(tf.nn.softmax(self.output_layer))
            self.picked_action_prob = tf.gather(self.action_probs, self.action)

            self.loss = -self.gate * tf.log(self.picked_action_prob) * self.is_ratio \
           				*(self.target - self.reg_constant*tf.log(self.mix_prob))


            self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
            self.train_op = self.optimizer.minimize(
                self.loss, global_step=tf.contrib.framework.get_global_step())


    def train(self, state, target, action, ratio, mix_prob, gate, sess=None): 
        '''
        Train policy (policy_idx) to improve mixture policy performance
        '''
        sess = sess or tf.get_default_session()

        feed_dict = { self.state: state, self.target: target, self.action: action, \
                      self.ratio: ratio, self.mix_prob: mix_prob, self.gate: gate }
        _, loss = sess.run([self.train_op, self.loss], feed_dict)

        return loss


    def sample(self, state, sess=None): 
        '''
        Sample from policy (policy_idx) 
        '''
        sess = sess or tf.get_default_session()

		return sess.run(self.action_probs, { self.state: state })



class GatingFunction():
    def __init__(self, learning_rate=0.01): 

        with tf.variable_scope('gate'): 
            # Define policy for agent 1
            self.state_gate = tf.placeholder(tf.int32, [], "state_gate")
            self.agent_action = tf.placeholder(dtype=tf.int32, name="agent_action")
            self.target = tf.placeholder(dtype=tf.float32, name='target')

            state_gate_one_hot = tf.one_hot(self.state_gate, obs_dim)

            self.output_layer = tf.contrib.layers.fully_connected(
                inputs=tf.expand_dims(state_gate_one_hot, 0),
                num_outputs=2,
                activation_fn=None,
                weights_initializer=tf.truncated_normal_initializer)

            self.agent_probs = tf.squeeze(tf.nn.softmax(self.output_layer))
            self.picked_agent_prob = tf.gather(self.agent_probs, self.agent_action)

            
            self.gate_loss = -tf.log(self.picked_agent_prob) * target

            self.optimizer_gate = tf.train.AdamOptimizer(learning_rate=learning_rate)
            self.train_op_gate = self.optimizer_gate.minimize(
                self.gate_loss, global_step=tf.contrib.framework.get_global_step())


    def train(self, state, selected_agent, target, sess=None): 
        sess = sess or tf.get_default_session()
        feed_dict = {self.state_gate: state, self.agent_action: selected_agent, self.target: target}
        loss, _ = sess.run([self.gate_loss, self.train_op_gate], feed_dict)
        return loss


    def sample(self, state, sess=None): 
        sess = sess or tf.get_default_session()
        return sess.run(self.agent_probs, {self.state_gate: state})



# Get dimensions of the environment state, action space
num_states, num_actions, _ = env.get_dimensions()

# Parameters for training
num_episodes = 5000
num_agents = 2
discount_factor = 0.99
batch_size = 20 

tf.reset_default_graph()

global_step = tf.Variable(0, name="global_step", trainable=False)

Transition = collections.namedtuple("Transition", ["state", "action", "reward", "next_state", "done"])

for k in range(num_agents): 
	agents.append(SingleAgent(scope='agent'+str(k)))

gate = GatingFunction()

episode_rewards_1 = np.zeros((num_episodes, ))
episode_rewards_2 = np.zeros((num_episodes, ))

with tf.Session() as sess: 

    sess.run(tf.global_variables_initializer())

    # Train agents over multiple episodes
    for i_episode in range(num_episodes): 


        # Train each agent with batch of experience
        for i_agent in range(num_agents): 

        	cur_agent = agents[i_agent]

            for k in range(batch_size): 

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
                    action_probs = cur_agent.sample(state=state)
                    action = np.random.choice(np.arange(num_actions), p=action_probs)

                    current_policy_probs.append(action_probs[action])

                    next_state, reward, done = env.step(action)

                    if t > 50: 
                        done = True

                    my_weight = gate.sample(state)[i_agent]
                    all_weights = gate.sample(state)

                    mix_prob = 0 

                    for agent_idx in range(num_agents): 
                    	mix_prob += all_weights[agent_idx]*(agents[agent_idx].sample(state)[action])

                    mixture_policy_probs.append(mix_prob)
                    episode_rewards_1[i_episode*batch_size + k] += reward
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

                    target = discounted_return - baseline

                    ratio = np.prod(np.divide(mixture_policy_probs[t+1:], \
                                              current_policy_probs[t+1:]))

                    # Update the agent's policy 
                    cur_agent.train(agent_idx=i_agent, state=transition.state, target=target, \
                        action=transition.action, ratio=ratio, mix_prob=mixture_policy_probs[t], other_dist=other_dist, weight=weights[t])

            monitor_epoch = 4 

            if i_episode % monitor_epoch == 0 and i_episode != 0 and i_agent == 0:
                print('----------------Episode: %d----------------' %i_episode) 

            # Print out the performance of each agent after 100 episodes
            if i_episode % monitor_epoch == 0 and i_episode != 0: 
                if i_agent == 0: 
                    print("Average return from Agent %d: %f" %(i_agent, np.mean(episode_rewards_1[batch_size*(i_episode+1) - 50:batch_size*(i_episode+1)])))
                    #print(sess.run(tf.trainable_variables()[0]))

                else: 
                    print("Average return from Agent %d: %f" %(i_agent, np.mean(episode_rewards_2[batch_size*(i_episode+1) - 50:batch_size*(i_episode+1)])))
                    #print(sess.run(tf.trainable_variables()[2]))

            if i_episode % 20 == 0 and i_episode > 0: 
                if i_agent == 0: 
                    for transition in episode: 
                        plt.figure(1)
                        plt.plot(transition.state % grid_size, transition.state // grid_size, 'bo')
                        plt.xlim([-1, grid_size])
                        plt.ylim([-1, grid_size])
                    plt.plot(transition.next_state % grid_size, transition.next_state // grid_size, 'ro')
                    plt.title("Agent 1")
                    plt.show()

                else: 
                    for transition in episode: 
                        plt.figure(1)
                        plt.plot(transition.state % grid_size, transition.state // grid_size, 'bo')
                        plt.xlim([-1, grid_size])
                        plt.ylim([-1, grid_size])
                    plt.plot(transition.next_state % grid_size, transition.next_state // grid_size, 'ro')
                    plt.title("Agent 2")
                    plt.show()

        # Train the gating function 

        for k in range(batch_size): 

        	episode = []

            # Reset the initial state
            state = env.reset()

            # Collection transitions from one rollout of current agent 
            for t in itertools.count(): 

                # Take a step in the environment

                all_weights = gate.sample(state)
                mix_prob = 0 
                for agent_idx in range(num_agents): 
                	mix_prob += all_weights[agent_idx]*np.array((agents[agent_idx].sample(state)))

                action = np.random.choice(np.arange(num_actions), p=mix_prob)

                next_state, reward, done = env.step(action)

                if t > 50: 
                    done = True

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

                '''
                # Get value estimate 
                value_estimate = joint_value_function.predict(transition.state)

                # Train joint value function 
                joint_value_function.train(transition.state)
                '''

                #gate.train(state=transition.state, selected_agent=i_agent)

                target = discounted_return - baseline

                ratio = np.prod(np.divide(mixture_policy_probs[t+1:], \
                                          current_policy_probs[t+1:]))

                # Update the agent's policy 
                multi_agent.train(agent_idx=i_agent, state=transition.state, target=target, \
                    action=transition.action, ratio=ratio, mix_prob=mixture_policy_probs[t], other_dist=other_dist, weight=weights[t])









