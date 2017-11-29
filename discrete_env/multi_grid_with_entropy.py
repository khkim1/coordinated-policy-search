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


# Coordinated Exploration
class MultiAgent(): 

    def __init__(self, learning_rate=0.01): 

        self.reg_constant_1 = 1 #0.0005 for hard env, 0.0002 for easy 
        self.reg_constant_2 = 1
        with tf.variable_scope('agent1'): 

            # Define policy for agent 1
            self.lambda_1 = tf.placeholder(tf.float32, name='lambda_1')
            self.state_1 = tf.placeholder(tf.int32, [], "state_1")
            self.action_1 = tf.placeholder(dtype=tf.int32, name="action_1")
            self.target_1 = tf.placeholder(dtype=tf.float32, name="target_1")
            self.ratio_1 = tf.placeholder(dtype=tf.float32, name="ratio_1")
            self.mix_prob_1 = tf.placeholder(dtype=tf.float32, name="mix_prob_1")
            self.other_dist_1 = tf.placeholder(dtype=tf.float32, name="other_dist_1")

            # Loss for policy 1
            state_one_hot_1 = tf.one_hot (self.state_1, obs_dim)
            
            self.output_layer_1 = tf.contrib.layers.fully_connected(
                inputs=tf.expand_dims(state_one_hot_1, 0),
                num_outputs=act_dim,
                activation_fn=None,
                weights_initializer=tf.truncated_normal_initializer)

            self.action_probs_1 = tf.squeeze(tf.nn.softmax(self.output_layer_1))
            self.picked_action_prob_1 = tf.gather(self.action_probs_1, self.action_1)

            eps = 1e-6
            self.KL_1 = tf.reduce_sum(self.other_dist_1 * tf.log((self.other_dist_1 + eps) / (self.action_probs_1 + eps)))

            #self.entropy_1 = tf.reduce_sum(self.action_probs * tf.log())

            self.loss_1 = -self.lambda_1 * tf.log(self.picked_action_prob_1) * self.ratio_1 * (self.target_1 - self.reg_constant_1*tf.log(self.mix_prob_1))
   

            '''
            self.loss_1 = -tf.log(self.picked_action_prob_1) * self.target_1 \
                        - self.reg_constant_1 * self.KL_1 * (1 - self.lambda_1)
            '''

            self.optimizer_1 = tf.train.AdamOptimizer(learning_rate=learning_rate)
            self.train_op_1 = self.optimizer_1.minimize(
                self.loss_1, global_step=tf.contrib.framework.get_global_step())

        # Define policy for agent 2 
        with tf.variable_scope('agent2'): 
            # Define policy for agent 2
            self.lambda_2 = tf.placeholder(tf.float32, name='lambda_2')
            self.state_2 = tf.placeholder(tf.int32, [], "state_2")
            self.action_2 = tf.placeholder(dtype=tf.int32, name="action_2")
            self.target_2 = tf.placeholder(dtype=tf.float32, name="target_2")
            self.ratio_2 = tf.placeholder(dtype=tf.float32, name="ratio_2")
            self.mix_prob_2 = tf.placeholder(dtype=tf.float32, name="mix_prob_2")
            self.other_dist_2 = tf.placeholder(dtype=tf.float32, name="other_dist_2")

            state_one_hot_2 = tf.one_hot(self.state_2, obs_dim)
            self.output_layer_2 = tf.contrib.layers.fully_connected(
                inputs=tf.expand_dims(state_one_hot_2, 0),
                num_outputs=act_dim,
                activation_fn=None,
                weights_initializer=tf.truncated_normal_initializer)

            self.action_probs_2 = tf.squeeze(tf.nn.softmax(self.output_layer_2))
            self.picked_action_prob_2 = tf.gather(self.action_probs_2, self.action_2)

            eps = 1e-6
            self.KL_2 = tf.reduce_sum(self.other_dist_2 * tf.log((self.other_dist_2 + eps) / (self.action_probs_2 + eps)))

            #self.entropy_2 = -tf.reduce_sum(self.action_probs_2 * tf.log(self.action_probs_2))

            self.loss_2 = -self.lambda_2 * tf.log(self.picked_action_prob_2) * self.ratio_2 * (self.target_2 - self.reg_constant_2*tf.log(self.mix_prob_2))

            '''
            self.loss_2 = - tf.log(self.picked_action_prob_2) * self.target_2 \
                        - self.reg_constant_2 * self.KL_2 * (1 - self.lambda_2)
            '''

            self.optimizer_2 = tf.train.AdamOptimizer(learning_rate=learning_rate)
            self.train_op_2 = self.optimizer_2.minimize(
                self.loss_2, global_step=tf.contrib.framework.get_global_step())


    def train(self, agent_idx, state, target, action, ratio, mix_prob, other_dist, weight, sess=None): 
        '''
        Train policy (policy_idx) to improve mixture policy performance
        '''
        sess = sess or tf.get_default_session()
        if agent_idx == 0: 
            feed_dict = { self.state_1: state, self.target_1: target, self.action_1: action, \
                          self.ratio_1: ratio, self.mix_prob_1: mix_prob, self.other_dist_1: other_dist, \
                          self.lambda_1: weight}
            _, loss = sess.run([self.train_op_1, self.loss_1], feed_dict)
            return loss
        else: 
            feed_dict = { self.state_2: state, self.target_2: target, self.action_2: action, \
                          self.ratio_2: ratio, self.mix_prob_2: mix_prob, self.other_dist_2: other_dist, \
                          self.lambda_2: weight}
            _, loss = sess.run([self.train_op_2, self.loss_2], feed_dict)
            return loss         


    def sample(self, agent_idx, state, sess=None): 
        '''
        Sample from policy (policy_idx) 
        '''
        sess = sess or tf.get_default_session()

        if agent_idx == 0: 
            return sess.run(self.action_probs_1, { self.state_1: state })

        else:
            return sess.run(self.action_probs_2, { self.state_2: state })

'''
# Joint value function trained by all the agents
class ValueFunction(): 

    def __init__(self, network_parameters): 


    def train(self, state, returns): 

    def predict(self, state): 
'''

class GatingFunction():
    def __init__(self, learning_rate=0.05): 

        with tf.variable_scope('gate'): 
            # Define policy for agent 1
            self.state_gate = tf.placeholder(tf.int32, [], "state_gate_1")
            self.agent_action = tf.placeholder(dtype=tf.int32, name="agent_action")

            state_gate_one_hot = tf.one_hot(self.state_gate, obs_dim)

            self.output_layer = tf.contrib.layers.fully_connected(
                inputs=tf.expand_dims(state_gate_one_hot, 0),
                num_outputs=2,
                activation_fn=None,
                weights_initializer=tf.truncated_normal_initializer)

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



def get_env_dim(env): 
    '''
    Return the dimensions of the state, action space of a Gym environment
    '''

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    return state_dim, action_dim


# Get dimensions of the environment state, action space
num_states, num_actions, _ = env.get_dimensions()

# Parameters for training
num_episodes = 5000
num_agents = 2
discount_factor = 0.99
batch_size = 20 

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
#joint_value_function = ValueFunction(network_parameters=value_function_parameters)

episode_rewards_1 = np.zeros((num_episodes, ))
episode_rewards_2 = np.zeros((num_episodes, ))

with tf.Session() as sess: 

    sess.run(tf.global_variables_initializer())

    # Train agents over multiple episodes
    for i_episode in range(num_episodes): 


        # Train each agent with one episode of experience
        for i_agent in range(num_agents): 

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
                    action_probs = multi_agent.sample(agent_idx=i_agent, state=state)
                    action = np.random.choice(np.arange(num_actions), p=action_probs)

                    current_policy_probs.append(action_probs[action])

                    next_state, reward, done = env.step(action)

                    if t > 50: 
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

                        other_action_probs = multi_agent.sample(agent_idx=1, state=state)
                        mixture_policy_probs.append(my_weight * action_probs[action] + \
                            (1 - my_weight) * other_action_probs[action])
                        episode_rewards_1[i_episode*batch_size + k] += reward
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


                        other_action_probs = multi_agent.sample(agent_idx=0, state=state)
                        mixture_policy_probs.append(my_weight * action_probs[action] + \
                            (1 - my_weight) * other_action_probs[action])
                        episode_rewards_2[i_episode*batch_size + k] += reward  
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

                    '''
                    # Get value estimate 
                    value_estimate = joint_value_function.predict(transition.state)

                    # Train joint value function 
                    joint_value_function.train(transition.state)
                    '''

                    gate.train(state=transition.state, agent=i_agent)

                    target = discounted_return - baseline

                    if i_agent == 0: 
                        other_dist = multi_agent.sample(agent_idx=1, state=transition.state)
                    else: 
                        other_dist = multi_agent.sample(agent_idx=0, state=transition.state)

                    '''
                    ratio = np.prod(np.divide(mixture_policy_probs[:t] + mixture_policy_probs[t+1:], \
                                              current_policy_probs[:t] + current_policy_probs[t+1:]))
                    '''

                    ratio = np.prod(np.divide(mixture_policy_probs[t+1:], \
                                              current_policy_probs[t+1:]))

                    # Update the agent's policy 
                    multi_agent.train(agent_idx=i_agent, state=transition.state, target=target, \
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














