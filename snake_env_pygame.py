# import tensorflow as tf
import numpy as np
import random
import math
import pygame

class SnakeGame():
    
    #Need to FIX
    def __init__(self,size=100):
        self.ENV_SIZE=size
        self.snake_position = [[50, 50], [40, 50], [30, 50]]
        self._state = [0,0,0,1,1,1]
        self._episode_ended = False
        self._steps_per_game = 1500
        self.score = 0
    
    # Fixed
    def starting_positions(self):
        self.snake_position = [[50, 50], [40, 50], [30, 50]]
        self.apple_position = [random.randrange(1, 10) * 10, random.randrange(1, 10) * 10]
        self._episode_ended = False
        self.score = 0
        return self.snake_position[0], self.snake_position, self.apple_position, self.score, self._episode_ended

    # Fixed
    def display_snake(self,display):
        for position in self.snake_position:
            pygame.draw.rect(display,red,pygame.Rect(position[0],position[1],10,10))

    def display_apple(self,display, apple):
        display.blit(apple,(self.apple_position[0], self.apple_position[1]))


    # Fixed
    def collision_with_apple(self):
        self.apple_position = [random.randrange(1, 10) * 10, random.randrange(1, 10) * 10]
        self.score += 1
        
    # Fixed
    def collision_with_boundaries(self,snake_start):
        if snake_start[0] >= 100 or snake_start[0] < 0 or snake_start[1] >= 100 or snake_start[1] < 0:
            return 1
        else:
            return 0

    # Fixed
    def collision_with_self(self,snake_start):
        # snake_start = snake_position[0]
        if snake_start in self.snake_position[1:]:
            return 1
        else:
            return 0

    # Fixed
    def blocked_directions(self):
        current_direction_vector = np.array(self.snake_position[0]) - np.array(self.snake_position[1])

        left_direction_vector = np.array([current_direction_vector[1], -current_direction_vector[0]])
        right_direction_vector = np.array([-current_direction_vector[1], current_direction_vector[0]])

        is_front_blocked = self.is_direction_blocked(current_direction_vector)
        is_left_blocked = self.is_direction_blocked(left_direction_vector)
        is_right_blocked = self.is_direction_blocked(right_direction_vector)

        return current_direction_vector, is_front_blocked, is_left_blocked, is_right_blocked

    # Fixed
    def is_direction_blocked(self,current_direction_vector):
        next_step = self.snake_position[0] + current_direction_vector
        snake_start = self.snake_position[0]
        if self.collision_with_boundaries(next_step) == 1 or self.collision_with_self(next_step.tolist()) == 1:
            return 1
        else:
            return 0



    # Fixed
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

    # Fixed 
    
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
    

    def step(self,action,dir):
        print(self.apple_position)
        is_done=False
        action_direction=self.get_action_direction(action,dir)
        snake_start=self.snake_position[0]
        new_start=[snake_start[0]+action_direction[0],snake_start[1]+action_direction[1]]
        self.snake_position.insert(0,new_start)
        # print(self.snake_position[0],self.apple_position)
        if self.snake_position[0]==self.apple_position:
            self.collision_with_apple()
            return False
        self.snake_position.pop(-1)
        if (self.collision_with_boundaries(self.snake_position[0]) or self.collision_with_self(self.snake_position[0])):
            return True
        else:
            return False
        
    
    def play_random(self,episode=1):
        score_list=[]
        while(episode>0):
            pygame.init()
            display = pygame.display.set_mode((display_width,display_height))
            snake_start, snake_position, apple_position, score, done = self.starting_positions()
            my_step=0
            while(not done and my_step<self._steps_per_game):
                
                display.fill(window_color)                
                rand_action=np.random.randint(3)
                current_direction=[self.snake_position[0][0]-self.snake_position[1][0],self.snake_position[0][1]-self.snake_position[1][1]]
                # print(current_direction)
                done = self.step(rand_action,current_direction)
                print(len(self.snake_position))
                self.display_apple(display,apple_image)
                self.display_snake(display)
                pygame.display.update()
                clock.tick(100)
                my_step +=1
            # print(self.score)
            pygame.quit()
            score_list.append(self.score)
            episode -= 1
        print(score_list)
        print(max(score_list))

snake_env=SnakeGame()
display_width = 100
display_height = 100
green = (0,255,0)
red = (255,0,0)
black = (0,0,0)
window_color = (200,200,200)
apple_image = pygame.image.load('apple.jpg')
clock=pygame.time.Clock() 
snake_env.play_random(episode=10)



