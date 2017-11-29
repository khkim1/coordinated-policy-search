import gym
import itertools
import numpy as np
import sys
import tensorflow as tf
import collections

import plotting

#env = CliffWalkingEnv()
#env = WindyGridworldEnv()
#env = GridworldEnv()
env = gym.make('InvertedPendulum-v1')

#var = 1
N_HIDDEN = 50
var = 0.3

class PolicyEstimator():
    """
    Policy Function approximator. 
    """
    
    def __init__(self, learning_rate=0.0001, scope="policy_estimator"):
        
        with tf.variable_scope(scope):

            self.state = tf.placeholder(tf.float32, (env.observation_space.shape[0], ), "state")
            self.action = tf.placeholder(dtype=tf.float32, name="action")
            self.target = tf.placeholder(dtype=tf.float32, name="target")


            self.full_1 = tf.layers.dense(
                inputs=tf.expand_dims(self.state, 0),
                units=N_HIDDEN,
                activation=tf.nn.relu)

            self.full_2 = tf.layers.dense(
                inputs=self.full_1,
                units=N_HIDDEN,
                activation=tf.nn.relu)

            # Output the mean of a multi-variate gaussian
            #self.output_layer = tf.layers.dense(inputs=self.full_2, units=2*env.action_space.shape[0])
            self.output_layer = tf.layers.dense(inputs=self.full_2, units=env.action_space.shape[0])

            mean = self.output_layer
            #mean = tf.gather(tf.squeeze(self.output_layer), 0)
            #var = tf.gather(tf.squeeze(self.output_layer), 1)

            self.picked_action_prob = (1 / (var * np.sqrt(2 * np.pi))) \
                            * tf.exp(-tf.square(self.action - mean) / (2 * var**2))

            # Loss and train op
            self.loss = -tf.log(self.picked_action_prob) * self.target

            self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
            self.train_op = self.optimizer.minimize(
                self.loss, global_step=tf.contrib.framework.get_global_step())
    
    def predict(self, state, sess=None):
        sess = sess or tf.get_default_session()
        return sess.run(self.output_layer, { self.state: state })

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
            action_mean_variance = np.ndarray.flatten(estimator_policy.predict(state))
            action = np.random.normal(loc=action_mean_variance[0], scale=var)
            next_state, reward, done, _ = env.step(action)
            
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
    stats = reinforce(env, policy_estimator, None, 10000, discount_factor=1)

