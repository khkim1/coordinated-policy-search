from __future__ import division

import gym
import itertools
import numpy as np
import sys
import tensorflow as tf
import collections
from database.examples.point_env import PointEnv
from database.rllab.envs.normalized_env import normalize
from database.sandbox.rocky.tf.envs.base import TfEnv
from scipy.stats import norm

from matplotlib import pyplot as plt
import plotting

#env = CliffWalkingEnv()
#env = WindyGridworldEnv()
#env = GridworldEnv()
#env = PointEnv()
env = normalize(PointEnv())
env = TfEnv(env)

#var = 1
N_HIDDEN = 50
#sig = 0.2
#var = [sig, sig]
#full_var = np.array([[sig, 0], [0, sig]])
high_threshold = 0.01
low_threshold = -0.01


class PolicyEstimator():
    """
    Policy Function approximator. 
    """
    
    def __init__(self, learning_rate=0.00001, scope="policy_estimator"):
        
        with tf.variable_scope(scope):

            self.state = tf.placeholder(tf.float32, [env.observation_space.shape[0]], "state")
            self.action = tf.placeholder(dtype=tf.float32, name="action")
            self.target = tf.placeholder(dtype=tf.float32, name="target")

            # This is just linear classifier
            self.mu = tf.contrib.layers.fully_connected(
                inputs=tf.expand_dims(self.state, 0),
                num_outputs=2,
                activation_fn=None,
                weights_initializer=tf.random_normal_initializer(
                                      stddev=np.sqrt(1 / N_HIDDEN)))
            self.mu = tf.squeeze(self.mu)
            
            self.sigma = tf.contrib.layers.fully_connected(
                inputs=tf.expand_dims(self.state, 0),
                num_outputs=2,
                activation_fn=None,
                weights_initializer=tf.random_normal_initializer(
                                      stddev=np.sqrt(1 / N_HIDDEN)))
            
            self.sigma = tf.squeeze(self.sigma)
            self.sigma = tf.nn.softplus(self.sigma) + 1e-5
            self.normal_dist = tf.contrib.distributions.MultivariateNormalDiag(self.mu, self.sigma)
            self.action = self.normal_dist.sample()
            self.action = tf.clip_by_value(self.action, [env.action_space.low[0], env.action_space.low[0]], 
                [env.action_space.high[0], env.action_space.high[0]])

            # Loss and train op
            self.loss = -self.normal_dist.log_prob(self.action) * self.target
            
            # Add cross entropy cost to encourage exploration
            #self.loss -= 1e-1 * self.normal_dist.entropy()
            
            self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
            self.train_op = self.optimizer.minimize(
                self.loss, global_step=tf.contrib.framework.get_global_step())


            '''
            self.full_1 = tf.layers.dense(
                inputs=tf.expand_dims(self.state, 0),
                units=N_HIDDEN,
                activation=tf.nn.relu,
                kernel_initializer=tf.truncated_normal_initializer)

            self.full_2 = tf.layers.dense(
                inputs=self.full_1,
                units=N_HIDDEN,
                activation=tf.nn.relu,
                kernel_initializer=tf.truncated_normal_initializer)

            # Output the mean of a multi-variate gaussian
            #self.output_layer = tf.layers.dense(inputs=self.full_2, units=2*env.action_space.shape[0])
            self.output_layer = tf.layers.dense(inputs=self.full_2, units=env.action_space.shape[0], 
                activation=tf.tanh, kernel_initializer=tf.truncated_normal_initializer)

            self.mean = self.output_layer * high_threshold
            #mean = tf.gather(tf.squeeze(self.output_layer), 0)
            #var = tf.gather(tf.squeeze(self.output_layer), 1)

            action = tf.reshape(self.action, shape=tf.shape(self.mean))

            #ds = tf.contrib.distributions
            #dist = ds.MultivariateNormalDiag(loc=self.mean, scale_diag=var, validate_args=True, allow_nan_stats=False)
            #self.picked_action_prob = dist.prob(self.action)

            self.picked_action_prob = (2*np.pi)**(-1) * np.linalg.det(full_var)**(-1/2) \
            * tf.exp(-0.5 * tf.matmul(tf.matmul(action - self.mean, tf.matrix_inverse(full_var.astype(np.float32))), \
                tf.transpose(action - self.mean))) 

            # Loss and train op
            self.loss = -tf.log(self.picked_action_prob) * self.target

            self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
            self.train_op = self.optimizer.minimize(
                self.loss, global_step=tf.contrib.framework.get_global_step())
            '''

    def predict(self, state, sess=None):
        sess = sess or tf.get_default_session()
        return sess.run(self.action, { self.state: state })

    def update(self, state, target, action, sess=None):
        sess = sess or tf.get_default_session()
        feed_dict = { self.state: state, self.target: target, self.action: action  }
        _, loss = sess.run([self.train_op, self.loss], feed_dict)
        return loss


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
    
    for i_episode in range(num_episodes):
        # Reset the environment and pick the fisrst action
        state = env.reset()
        
        episode = []
        
        # One step in the environment
        for t in itertools.count():
            
            # Take a step
            #action_means = np.ndarray.flatten(estimator_policy.predict(state))
            #action = np.random.multivariate_normal(mean=action_means, cov=full_var)

            action = estimator_policy.predict(state)

            '''
            max_idx = np.argmax(np.abs(action))
            a_max = action[max_idx]

            if a_max > high_threshold or a_max < low_threshold: 
                action_clipped = action / (10*np.abs(a_max))
            '''


            '''
            action_clipped = [np.max([np.min([action[0], high_threshold]), low_threshold]), 
                          np.max([np.min([action[1], high_threshold]),
                           low_threshold])]
            '''

            next_state, reward, done, _ = env.step(action)

            '''
            if t > 50:
                done = True
            '''
            
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
            # sys.stdout.flush()
            '''

            if done:
                break
                
            state = next_state
        
        monitor_epoch = 50
        if i_episode % monitor_epoch == 0 and i_episode > 0: 
            print("avg reward : %f" %(np.mean(stats.episode_rewards[i_episode-monitor_epoch:i_episode])))


        baseline_value = np.mean([cur_trans.reward for cur_trans in episode])  

        # Go through the episode and make policy updates
        for t, transition in enumerate(episode):
            # The return after this timestep
            total_return = sum(discount_factor**i * t.reward for i, t in enumerate(episode[t:]))
       
            advantage = total_return - baseline_value
            #advantage += np.max([0, baseline_value - v_prev])

            # Update our policy estimator
            estimator_policy.update(transition.state, advantage, transition.action)

            #print(p_action)

            if i_episode % 200 == 0 and i_episode > 0:
                plt.figure(1)
                plt.plot(transition.state[0], transition.state[1], 'bo')
                if t == len(episode) - 1:
                    plt.show()
    
    
    return stats

tf.reset_default_graph()

global_step = tf.Variable(0, name="global_step", trainable=False)
policy_estimator = PolicyEstimator()
#value_estimator = ValueEstimator()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    # Note, due to randomness in the policy the number of episodes you need to learn a good
    # policy may vary. ~2000-5000 seemed to work well for me.
    print(env)
    stats = reinforce(env, policy_estimator, None, 10000, discount_factor=0.99)

