
import numpy as np
import random
import math
import tensorflow as tf
import numpy as np

from tf_agents.environments import py_environment
from tf_agents.environments import tf_environment
from tf_agents.environments import tf_py_environment
from tf_agents.environments import utils
from tf_agents.specs import array_spec
from tf_agents.environments import wrappers
from tf_agents.environments import suite_gym
from tf_agents.trajectories import time_step as ts


class SnakeGame(py_environment.PyEnvironment):

    def __init__(self):
        # self._action_spec = array_spec.BoundedArraySpec(shape=(), dtype=np.int32, minimum=0, maximum=3, name='action')
        # self._observation_spec = array_spec.BoundedArraySpec(shape=(1,9), dtype=np.int32, minimum=0, maximum=1, name='observation')
        self._action_spec = array_spec.BoundedArraySpec(shape=(), dtype=np.int32, minimum=0, maximum=3, name='action')
        self._observation_spec = array_spec.BoundedArraySpec(shape=(1,9), dtype=np.int32, minimum=0, maximum=1, name='observation')
        self._state = [0, 0, 0, 0, 0, 0, 0, 0, 0]
        self._episode_ended = False
        self.ENV_SIZE=100
        self.snake_position = [[50, 50], [40, 50], [30, 50]]
        self.apple_position = [random.randrange(1, 10) * 10, random.randrange(1, 10) * 10]
        self._steps_per_game = 1500
        self.score = 0
        self.step2=0
    
    def __is_spot_empty(self, index):
        return self._state[index] == 0

    def __all_spots_occupied(self):
        return all(item == 1 for item in self._state)

    def action_spec(self):
        return self._action_spec

    def observation_spec(self):
        return self._observation_spec   
    
    def _reset(self):
        self._state = [0, 0, 0, 0, 0, 0, 0, 0, 0]
        self.snake_position = [[50, 50], [40, 50], [30, 50]]
        self.apple_position = [random.randrange(1, 10) * 10, random.randrange(1, 10) * 10]
        self._steps_per_game = 1500
        self.score = 0
        self.reward=0
        self.step2=0
        self._episode_ended = False
        return ts.restart(np.array([self._state], dtype=np.int32))

    def _step(self, action):   
        print(len(self.snake_position))
        self.step2 += 1 
        current_direction=[self.snake_position[0][0]-self.snake_position[1][0],self.snake_position[0][1]-self.snake_position[1][1]]
        done = self.my_step(action,current_direction)
        if self._episode_ended==True:
            print(len(self.snake_position))
            return self.reset()
        if done or self.step2 > self._steps_per_game:
            self._episode_ended=True
            return ts.termination(np.array([self._state], dtype=np.int32), self.reward)
        else:
            return ts.termination(np.array([self._state], dtype=np.int32), self.reward)
        
    def starting_positions(self):
        self.snake_position = [[50, 50], [40, 50], [30, 50]]
        self.apple_position = [random.randrange(1, 10) * 10, random.randrange(1, 10) * 10]
        self._episode_ended = False
        self.score = 0
        return self.snake_position[0], self.snake_position, self.apple_position, self.score, self._episode_ended

    def collision_with_apple(self):
        self.apple_position = [random.randrange(1, 10) * 10, random.randrange(1, 10) * 10]
        self.score += 1
        
    def collision_with_boundaries(self,snake_start):
        if snake_start[0] >= 100 or snake_start[0] < 0 or snake_start[1] >= 100 or snake_start[1] < 0:
            return 1
        else:
            return 0

    def collision_with_self(self,snake_start):
        # snake_start = snake_position[0]
        if snake_start in self.snake_position[1:]:
            return 1
        else:
            return 0

    def blocked_directions(self):
        current_direction_vector = np.array(self.snake_position[0]) - np.array(self.snake_position[1])

        left_direction_vector = np.array([current_direction_vector[1], -current_direction_vector[0]])
        right_direction_vector = np.array([-current_direction_vector[1], current_direction_vector[0]])

        is_front_blocked = self.is_direction_blocked(current_direction_vector)
        is_left_blocked = self.is_direction_blocked(left_direction_vector)
        is_right_blocked = self.is_direction_blocked(right_direction_vector)

        return current_direction_vector, is_front_blocked, is_left_blocked, is_right_blocked

    def is_direction_blocked(self,current_direction_vector):
        next_step = self.snake_position[0] + current_direction_vector
        snake_start = self.snake_position[0]
        if self.collision_with_boundaries(next_step) == 1 or self.collision_with_self(next_step.tolist()) == 1:
            return 1
        else:
            return 0
    def angle_with_apple(self):
        apple_direction_vector = np.array(self.apple_position) - np.array(self.snake_position[0])
        snake_direction_vector = np.array(self.snake_position[0]) - np.array(self.snake_position[1])

        norm_of_apple_direction_vector = np.linalg.norm(apple_direction_vector)
        norm_of_snake_direction_vector = np.linalg.norm(snake_direction_vector)
        if norm_of_apple_direction_vector == 0:
            norm_of_apple_direction_vector = 10
        if norm_of_snake_direction_vector == 0:
            norm_of_snake_direction_vector = 10

        apple_direction_vector_normalized = apple_direction_vector / norm_of_apple_direction_vector
        snake_direction_vector_normalized = snake_direction_vector / norm_of_snake_direction_vector
        angle = math.atan2(
            apple_direction_vector_normalized[1] * snake_direction_vector_normalized[0] - apple_direction_vector_normalized[
                0] * snake_direction_vector_normalized[1],
            apple_direction_vector_normalized[1] * snake_direction_vector_normalized[1] + apple_direction_vector_normalized[
                0] * snake_direction_vector_normalized[0]) / math.pi
        return angle, snake_direction_vector, apple_direction_vector_normalized, snake_direction_vector_normalized

    def get_action_direction(self,action,dir):
        if action==0:
            ret_dir=dir
        else:
            #left =2 right =1
            factor=(-1)**(action)
            if dir==[10,0]:
                ret_dir=[0,-1*factor*10]
            elif dir==[-10,0]:
                ret_dir=[0,1*factor*10]
            elif dir==[0,10]:
                ret_dir=[10*factor,0]
            else:
                ret_dir=[-1*10*factor,0]
        return ret_dir
    
    def my_step(self,action,dir):
        # print(self.apple_position)
        is_done=False
        action_direction=self.get_action_direction(action,dir)
        snake_start=self.snake_position[0]
        new_start=[snake_start[0]+action_direction[0],snake_start[1]+action_direction[1]]
        self.snake_position.insert(0,new_start)
        self.reward=0
        if self.snake_position[0]==self.apple_position:
            self.score+=1
            self.collision_with_apple()
            self.reward=1
            return False
        self.snake_position.pop(-1)
        if (self.collision_with_boundaries(self.snake_position[0]) or self.collision_with_self(self.snake_position[0])):
            self.reward=-100
            return True
        else:
            return False
        
    


environment2 = SnakeGame()
# utils.validate_py_environment(environment2, episodes=5)

tf_env = tf_py_environment.TFPyEnvironment(environment2)

time_step = tf_env.reset()
rewards = []
steps = []
num_episodes = 10

for _ in range(num_episodes):
  episode_reward = 0
  episode_steps = 0
  tf_env.reset()
  while not tf_env.current_time_step().is_last():
    action = tf.random.uniform([1], 0, 3, dtype=tf.int32)
    next_time_step = tf_env.step(action)
    episode_steps += 1
    episode_reward += next_time_step.reward.numpy()
  rewards.append(episode_reward)
  steps.append(episode_steps)
  
num_steps = np.sum(steps)
avg_length = np.mean(steps)
avg_reward = np.mean(rewards)
max_reward = np.max(rewards)
max_length = np.max(steps)

print('num_episodes:', num_episodes, 'num_steps:', num_steps)
print('avg_length', avg_length, 'avg_reward:', avg_reward)
print('max_length', max_length, 'max_reward:', max_reward)