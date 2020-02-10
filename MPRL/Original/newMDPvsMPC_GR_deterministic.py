# -*- coding: utf-8 -*-
"""
Created on Sat Apr  6 10:46:49 2019

@author: megha
"""


import gym
import numpy as np
import matplotlib.pyplot as plt

from qlearning import QLearning

def downsample(image):
    """ Take only alternate pixels - basically halves the resolution of the image (which is fine for us) """
    return image[::2, ::2, :]

def remove_color(image):
    """Convert all color (RGB is the third dimension in the image)"""
    return image[:, :, 0]

def remove_background(image):
    image[image == 144] = 0
    image[image == 109] = 0
    return image

def preprocess_observations(observation):

    """ convert the 210x160x3 uint8 frame into a 6400 float vector """
    processed_observation = observation[35:195] # cropping (walls are cropped)
    processed_observation = downsample(processed_observation)
    processed_observation = remove_color(processed_observation)
    processed_observation = remove_background(processed_observation)
    
    return processed_observation

def calculate_Vavg(current_pos, prev_ball_pos_T1, prev_ball_pos_T2):
    ball_dir1 = detect_ball_dir(current_pos, prev_ball_pos_T1)
    
    ball_dir2 = detect_ball_dir(prev_ball_pos_T1, prev_ball_pos_T2)
    Vavg = detect_impact(ball_dir1, ball_dir2)
    #print(Vavg)
    return Vavg

def find_ball(processed_observation, colour):

    found = False
    pos = [0, 0]
    i=0
    
    while not found and i < 80:

        j = 0

        while not found and j < 80:

            if processed_observation[i, j] == colour:

                top_left = np.array([i, j])
                pos = [top_left[1], top_left[0]+1]

                found = True

            j += 1

        i += 1  
        #dims = [0, 2]   
    return pos, found


def find_ballNext(processed_observation, colour, prev_pos, window):

    pos = [0, 0]
    i=0
    found = False
    processed_observationCropped =processed_observation[:, prev_pos[0]-(window):prev_pos[0]+(window)]
    while not found and i < 80:

        j = 0

        while not found and j < 2*window:

            if processed_observationCropped[i, j] == colour:

                top_left = np.array([i, j])
                pos = [top_left[1], top_left[0]+1]
                difference = pos[0]-window
                pos = [prev_pos[0]+difference, pos[1]]
                found = True

            j += 1

        i += 1  
        #dims = [0, 2]
    return pos

def find_paddle(processed_observation, colour, start, end):

    found = False
    pos = [0, 0]
    i=0
    processed_observation = processed_observation[:, start:end]

    while not found and i < 80:

        j = 0

        while not found and j < 10:

            if processed_observation[i, j] == colour:

                top_left = np.array([i, j])
                pos = [top_left[1], top_left[0]+4]

                found = True

            j += 1

        i += 1  
        #dims = [0, 2]   
    return pos
    

def detect_ball_dir(current_pos, prev_pos):
    ball_dir = None
    
    if current_pos is not None:
        diff = [current_pos[0]-prev_pos[0], current_pos[1]-prev_pos[1]]
        ball_dir = diff
    else:
        diff = None #no ball found
        ball_dir=diff
    return ball_dir

def detect_impact(ball_dir1, ball_dir2):
    if np.sign(ball_dir1[0])!=np.sign(ball_dir2[0]) or np.sign(ball_dir1[1])!=np.sign(ball_dir2[1]):
        return ball_dir1

    else:
        Vavg = [(ball_dir1[0] + ball_dir2[0]) / 2.0, (ball_dir1[1] + ball_dir2[1]) / 2.0]
        #matrix_x = np.array([ball_dir1[0], ball_dir2[0]])
        #matrix_y = np.array([ball_dir1[1], ball_dir2[1]])

        #variance_x = np.var(matrix_x)
        #variance_y = np.var(matrix_y)
        #noise_x = np.random.normal(0, variance_x)
        #noise_y = np.random.normal(0, variance_y)
        #Vavg = [Vavg[0] + noise_x, Vavg[1]+noise_y]
        return Vavg



def propagate_model2(pos, vel):
    y_old = pos[1]
    x_old = pos[0]
    y_next=0
    x_next=0
    while x_old<=70:
        if y_old>=80 or y_old<=0:
            y_next = y_old -vel[1]
        else:
            y_next = y_old +vel[1]
        y_old = y_next
        x_next = (x_old) +vel[0]
        x_old = x_next
    pos = [70,y_old]
    return pos

#    m=(vel[1]/vel[0])
#    y=0
#    if vel[1]<=0:
#        y=(m*(70-pos[0]))+pos[1]
#        if y<0:
#            x=((0-pos[1])/m)+pos[0]
#            m=-1*m
#            y=m*(70-x)
#            
#    if vel[1]>0:
#        y=(m*(70-pos[0]))+pos[1]
#        if y>80:
#            x=((80-pos[1])/m)+pos[0]
#            m=-1*m
#            y=(m*(70-x))+80        
#            
#    pos = [70, y]        
#    return pos
    
def choose_action(ai_pos, pred_pos):

    if pred_pos[1]-ai_pos[1]>3: #ai_pos[1] < pred_pos[1]:
        # signifies down in openai gym
        return [3, 1]
    elif ai_pos[1]-pred_pos[1]>3:#ai_pos[1] > pred_pos[1]:
         # signifies up in openai gym
        return [2, 2]
    else:
        return [0, 0]
    
def indexToAction(index):
    if index==0:
        return 0
    elif index==1:
        return 3
    else:
        return 2


def main():
    env = gym.make("PongDeterministic-v4") # skip 4 frames everytime and no random action 
    #env = gym.make("Pong-v0")
    observation = env.reset() # This gets us the image
    episode_number, done, action_mpc, reward_sum, index_mpc = 0, 0, 0, 0, 0
    prev_ball_pos_T2, prev_ball_pos_T1, pred_pos= [0, 0], [0, 0], [0, 0]
    ball_colour = 236
    #opponent_colour = 213
    ai_colour = 92
    episode_rewards, game_states, game_actions = [], [], []
    found, action_not_same = False, False
    
    controller = QLearning(ball_x=82, ball_y=82, ai_pos_y=84, v_x=11, v_y=11, n_action=3)
    state = [0, 0, 0, 0, 0]
    index_mdp = 0
    while episode_number<4500:
        """ Get Image """
        env.render()
        
        """ Process the Image to give 80x80 matrix""" 
        processed_image = preprocess_observations(observation)
        
        """ Process 80x80 image till ball is found, once it is found only process a window around ball to detect its new position"""
        if not found:
            ball_pos, found = find_ball(processed_image, ball_colour) 
        else:
            ball_pos = find_ballNext(processed_image, ball_colour, ball_pos, window=6)
            
        """ In order to prevent program from crashing it is important to reset ball position if it goes out of bounds"""                    
        if ball_pos[0]==0 and ball_pos[1]==0 or ball_pos[0]>70 or ball_pos[0]<10:
            found=False

        """ Calculate ball velocity given current position and previous 2 consecutive positions"""         
        ball_vel = calculate_Vavg(ball_pos, prev_ball_pos_T1, prev_ball_pos_T2)
        
        """ Ball velocity always goes high at start of game this can crash the code, so need to ensure in this case it is reset"""
        if abs(int(ball_vel[0]))>5 or abs(int(ball_vel[1]))>5:
            ball_vel = [0, 0] 
        
        """ Find position of our player's paddle"""
        ai_pos = find_paddle(processed_image, ai_colour, 65, 75)
        
        """ Only try to predict ball's position if it is coming towards our player"""
        if prev_ball_pos_T2[0]>10 and ball_vel[0]>0 and ball_pos[0]<71:
            pred_pos = propagate_model2(ball_pos, ball_vel)
            [action_mpc, index_mpc] = choose_action(ai_pos, pred_pos)
        
        last_s = state
        state = [round(ball_pos[0]), round(ball_pos[1]), round(ai_pos[1]), int(round(4+ball_vel[0])), int(round(4+ball_vel[1]))]
        if action_not_same:
            controller.update(last_s, index_mdp, state, 0)
        else:
            controller.update(last_s, index_mdp, state, 0.1)
        
        index_mdp = controller.action(state)
        action_mdp = indexToAction(index_mdp)
        
        """ If our player is very close to the predicted ball position then it chooses action given by MDP else it uses MPC"""
        if abs(ai_pos[1]-pred_pos[1])<5: 
            action = action_mdp
            index = index_mdp
            game_states.append(state)
            game_actions.append(index)
        else:
            if (action_mdp!=action_mpc): # when ai far from predicted pos use mpc to train ai
                action = action_mpc
                index = index_mpc
                action_not_same = True
            else:
                action = action_mdp
                index = index_mdp
                
        # game_states.append(state)
        # game_actions.append(index)
              
        observation, reward, done, info = env.step(action) 
   
        if reward<0:
            
            for ind, val in enumerate(game_states[len(game_states)-4:-1]):
                controller.update(val, game_actions[len(game_states)-4+ind], game_states[len(game_states)-4+ind+1], -1, 0.7, 0.1)
                
            game_states=[]
            game_actions=[]

        elif reward>0:
            for ind, val in enumerate(game_states[:-1]):
                controller.update(val, game_actions[ind], game_states[ind+1], 1, 0.7, 0.1)
            game_states =[]
            game_actions=[]
            
            #if reward is 1 then all past  actions before get positive reward
        
        reward_sum += reward
        
        if done:
            observation = env.reset() # reset env
            episode_rewards.append(reward_sum)
            print ('episode:',episode_number, ' reward total was %f' %(reward_sum))   #. running mean: %f' % (reward_sum, running_reward))
            reward_sum = 0
        
            episode_number += 1
            
        prev_ball_pos_T2 = prev_ball_pos_T1
        prev_ball_pos_T1 = ball_pos
        
    return episode_rewards

if __name__ == '__main__':
    reward = main()
    plt.plot(reward)
    plt.ylabel('reward')
    plt.xlabel('episode')
    plt.show()
