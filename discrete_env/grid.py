import gym
import itertools
import matplotlib
import numpy as np
import sys
import tensorflow as tf
import collections
from grid_env import GridEnv
from lib import plotting
from matplotlib import pyplot as plt

env = GridEnv()
obs_dim, act_dim, grid_size = env.get_dimensions()

N_HIDDEN = 300 

class PolicyEstimator():
    """
    Policy Function approximator. 
    """
    def __init__(self, learning_rate=0.01, scope="policy_estimator"):
        with tf.variable_scope(scope):

            self.h_constant = 0.1
            self.state = tf.placeholder(tf.int32, [], "state")
            self.action = tf.placeholder(dtype=tf.int32, name="action")
            self.target = tf.placeholder(dtype=tf.float32, name="target")


            # This is just table lookup estimator
            state_one_hot = tf.one_hot(self.state, obs_dim)

            self.output_layer = tf.contrib.layers.fully_connected(
            inputs=tf.expand_dims(state_one_hot, 0),
            num_outputs=act_dim,
            activation_fn=None,
            weights_initializer=tf.truncated_normal_initializer)

            self.action_probs = tf.squeeze(tf.nn.softmax(self.output_layer))
            self.picked_action_prob = tf.gather(self.action_probs, self.action)

            self.entropy = -tf.reduce_sum(self.action_probs * tf.log(self.action_probs))

            # Loss and train op
            self.loss = -tf.log(self.picked_action_prob) * self.target - self.h_constant * self.entropy

            self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
            self.train_op = self.optimizer.minimize(
                self.loss, global_step=tf.contrib.framework.get_global_step())
    
    def predict(self, state, sess=None):
        sess = sess or tf.get_default_session()
        return sess.run(self.action_probs, { self.state: state })

    def update(self, state, target, action, sess=None):
        sess = sess or tf.get_default_session()
        feed_dict = { self.state: state, self.target: target, self.action: action  }
        _, loss = sess.run([self.train_op, self.loss], feed_dict)
        return loss


class ValueEstimator():
    """
    Value Function approximator. 
    """
    
    def __init__(self, learning_rate=0.01, scope="value_estimator"):
        with tf.variable_scope(scope):
            self.state = tf.placeholder(tf.int32, [], "state")
            self.target = tf.placeholder(dtype=tf.float32, name="target")

             # This is just table lookup estimator
            state_one_hot = tf.one_hot(self.state, obs_dim)
            self.output_layer = tf.contrib.layers.fully_connected(
                inputs=tf.expand_dims(state_one_hot, 0),
                num_outputs=1,
                activation_fn=None,
                weights_initializer=tf.zeros_initializer, 
                scope='fc1')

            self.value_estimate = tf.squeeze(self.output_layer)
            self.loss = tf.squared_difference(self.value_estimate, self.target)

            self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
            self.train_op = self.optimizer.minimize(
                self.loss, global_step=tf.contrib.framework.get_global_step())        
    
    def predict(self, state, sess=None):
        sess = sess or tf.get_default_session()
        return sess.run(self.value_estimate, { self.state: state })

    def update(self, state, target, sess=None):
        sess = sess or tf.get_default_session()
        feed_dict = { self.state: state, self.target: target }
        _, loss = sess.run([self.train_op, self.loss], feed_dict)
        return loss

    def get_weights(self, sess=None, weights=False, biases=False):
        sess = sess or tf.get_default_session()
        all_vars = tf.global_variables()
        def get_var(name): 
            for i in range(len(all_vars)):
                if all_vars[i].name.startswith(name):
                    return all_vars[i]
            return None
        if weights:
            fc1_var = sess.run(get_var('value_estimator/fc1/weights'))
        
        elif biases: 
            fc1_var = sess.run(get_var('value_estimator/fc1/biases'))
        else:
            fc1_var = None

        return fc1_var



def reinforce(env, estimator_policy, estimator_value, num_episodes, discount_factor=1.0):
    """
    REINFORCE (Monte Carlo Policy Gradient) Algorithm. Optimizes the policy
    function approximator using policy gradient.
    
    Args:
        env: OpenAI environment.
        estimator_policy: Policy Function to be optimized 
        estimator_value: Value function approximator, used as a baseline
        num_episodes: Number of episodes to run for
        discount_factor: Time-discount factor
    
    Returns:
        An EpisodeStats object with two numpy arrays for episode_lengths and episode_rewards.
    """

    # Keeps track of useful statistics
    stats = plotting.EpisodeStats(
        episode_lengths=np.zeros(num_episodes),
        episode_rewards=np.zeros(num_episodes))    
    
    Transition = collections.namedtuple("Transition", ["state", "action", "reward", "next_state", "done"])
    

    parameter_list = []
    for i_episode in range(num_episodes):
        # Reset the environment and pick the fisrst action
        state = env.reset()
        
        episode = []
        
        # One step in the environment
        for t in itertools.count():
            
            # Take a step
            action_probs = estimator_policy.predict(state)
            action = np.random.choice(np.arange(act_dim), p=action_probs)
            next_state, reward, done = env.step(action)
            
            # Keep track of the transition
            episode.append(Transition(
              state=state, action=action, reward=reward, next_state=next_state, done=done))
            
            # Update statistics
            stats.episode_rewards[i_episode] += reward
            stats.episode_lengths[i_episode] = t
            
            '''
            # Print out which step we're on, useful for debugging.
            print("\rStep {} @ Episode {}/{} ({})".format(
                    t, i_episode + 1, num_episodes, stats.episode_rewards[i_episode - 1]), end="")
            #sys.stdout.flush()
            '''

            if done:
                break
                
            state = next_state
    
        if i_episode % 100 == 0 and i_episode > 0: 
            print("avg reward : %f" %(np.mean(stats.episode_rewards[i_episode-100:i_episode])))
        
        '''            
        horizon = 10 

        if i_episode > horizon:
            past_param = parameter_list[i_episode-horizon]
            past_w = past_param[0]
            past_b = past_param[1]
        '''

        # Go through the episode and make policy updates
        for t, transition in enumerate(episode):
            # The return after this timestep
            total_return = sum(discount_factor**i * t.reward for i, t in enumerate(episode[t:]))
           
            v_prev = estimator_value.predict(transition.state)
            td_target = transition.reward + estimator_value.predict(transition.next_state)


            # Update our value estimator
            estimator_value.update(transition.state, total_return)
            
            # Calculate baseline/advantage
            #baseline_value = estimator_value.predict(transition.state)            
            average = np.mean([cur_transition.reward for cur_transition in episode])
            advantage = total_return - v_prev
            #advantage = td_target - v_prev 

            # Update our policy estimator
            estimator_policy.update(transition.state, advantage, transition.action)
        
        if i_episode % 2000 == 0 and i_episode > 0: 
            for transition in episode: 
                plt.figure(1)
                plt.plot(transition.state % grid_size, transition.state // grid_size, 'bo')
                plt.xlim([-1, grid_size])
                plt.ylim([-1, grid_size])
            plt.plot(transition.next_state % grid_size, transition.next_state // grid_size, 'ro')
            plt.show()

        #w = estimator_value.get_weights(weights=True)
        #b = estimator_value.get_weights(biases=True)

        #parameter_list.append((w, b))

    return stats

tf.reset_default_graph()

global_step = tf.Variable(0, name="global_step", trainable=False)
policy_estimator = PolicyEstimator()
value_estimator = ValueEstimator()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    # Note, due to randomness in the policy the number of episodes you need to learn a good
    # policy may vary. ~2000-5000 seemed to work well for me.
    stats = reinforce(env, policy_estimator, value_estimator, 20000, discount_factor=1)

