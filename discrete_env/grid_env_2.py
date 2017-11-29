from rllab.envs.base import Env
from rllab.spaces import Box
from rllab.envs.base import Step
import numpy as np


grid_size = 5


class GridEnv():

    def is_top_boarder(self):
        return self._state in list(range(grid_size))

    def is_bottom_boarder(self):
        return self._state in list(range(grid_size**2 - grid_size, grid_size**2))

    def is_left_boarder(self):
        return self._state in list(range(0, grid_size**2 - grid_size + 1, grid_size))

    def is_right_boarder(self):
        return self._state in list(range(grid_size - 1, grid_size**2, grid_size))

    def get_dimensions(self):
        obs_dim = grid_size**2
        act_dim = 4
        return obs_dim, act_dim, grid_size

    def reset(self):
        # initial_states = [198, 199, 200, 219, 220, 221, 240, 241, 242]
        # rand_idx = np.random.randint(len(initial_states))
        # self._state = initial_states[rand_idx]

        self._state = 2

        # self._state = 220 #10
        # self._state = np.array([-3, -5])
        observation = np.copy(self._state)
        return observation

    def step(self, action):

        # Up
        prev = np.copy(self._state)

        if action == 0:
            if not self.is_top_boarder():
                self._state = self._state - grid_size

        # Right
        elif action == 1:
            if not self.is_right_boarder():
                self._state = self._state + 1

        # Down
        elif action == 2:
            if not self.is_bottom_boarder():
                self._state = self._state + grid_size

        # Left
        elif action == 3:
            if not self.is_left_boarder():
                self._state = self._state - 1

        goal = 22
        #trap = [218, 219, 220, 221, 222]
        trap = [11, 12, 13]

        if self._state == goal:
            reward = 100
            done = True

        elif self._state in trap:
            reward = -30
            done = True

        elif self._state == prev:
            #reward = -0.2
            reward = -0.2
            done = False

        else:
            #reward = -0.2
            reward = -0.2
            done = False

        next_observation = np.copy(self._state)

        # return (next_observation, reward, done)
        return next_observation, reward, done

    def render(self):
        print('current state:', self._state)


'''
from rllab.envs.base import Env
from rllab.spaces import Box
from rllab.envs.base import Step
import numpy as np


class PointEnv(Env):
    @property
    def observation_space(self):
        return Box(low=-np.inf, high=np.inf, shape=(2,))

    @property
    def action_space(self):
        return Box(low=-0.1, high=0.1, shape=(2,))

    def reset(self, **kwargs):
        self._state = np.random.uniform(-1, 1, size=(2,))
        observation = np.copy(self._state)
        return observation

    def step(self, action):
        self._state = self._state + action
        x, y = self._state
        reward = - (x ** 2 + y ** 2) ** 0.5
        done = abs(x) < 0.01 and abs(y) < 0.01
        next_observation = np.copy(self._state)
        return Step(observation=next_observation, reward=reward, done=done)

    def render(self):
        print('current state:', self._state)

'''
