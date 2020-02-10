# import the system and os
import sys
from os import getcwd
import gym
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as image
import random
import time
import copy
import cv2
import datetime
from pongStates import pongStates
from timeit import default_timer as timer
# need to include the other folders in the project
sys.path.append(getcwd() + "\\Original")
sys.path.append(getcwd() + "\\Algorithm")
sys.path.append(getcwd() + "\\Main")
from mpc import mpc
from newMDPvsMPC_GR_deterministic import indexToAction
from qlearning import QLearning
# import important packages
from collections import deque

zfactor = 4
downsample_factor = np.array( [2,4] )
grayscale = 2

# main body of program
def Main():
    # parameters for paddle dynamics
    alpha = 2
    beta = 4

    # modes
    visualMode = False

    grayscale = 2
    agentCol = [92]
    ballCol = [236]
    paddleWidth = 4
    # area of interest before and after downsampling
    #aoi_orig = np.array( [[15,34], [144,196]])
    #ballaoiFull = np.array( [[ ( (aoi_orig[0,0] / downsample_factor[0]) + 1), \
    #( (aoi_orig[0,1] / downsample_factor[1]) + 1) ], [ (aoi_orig[1,0] / downsample_factor[0]),\
    #(aoi_orig[1,1] / downsample_factor[1]) ] ] ).astype(int)
    #print("Ball AOI Full: ",ballaoiFull)

    ## the width of the paddle after downsampling
    #paddleWidth_ds = (paddleWidth / downsample_factor[0]).astype(int)

    #agentaoiFull = np.array( [[ (ballaoiFull[1,0]-paddleWidth_ds), ballaoiFull[0,1] ],\
    #[ ballaoiFull[1,0], ballaoiFull[1,1] ]] ).astype(int)
    #justLeftOfPaddle = ballaoiFull[1,0] - paddleWidth_ds - 2
    justLeftOfPaddle = 68
    #print("agent AOI Full: ",agentaoiFull)
    learningMode = True
    mpc_h = mpc(alpha,beta,25,0,justLeftOfPaddle, learningMode)


    pStates = pongStates(ballCol,agentCol)
    #env = gym.make("PongDeterministic-v4") # skip 4 frames everytime and no random action
    env = gym.make("Pong-v0")
    observation = env.reset()
    observation, reward, done, info = env.step(0)

    i = 0

    #temp
    next_move = 0
    next_index = 0

    # qlearning
    # should change this to only make the bits it needs
    rl = QLearning(ball_x=82, ball_y=82, ai_pos_y=84, v_x=11, v_y=11, n_action=3)
    episode_rewards,partial, game_states, game_actions = [], [], [], []

    episode_number=1
    reward_sum = 0

    opponent_score = 0
    agent_score = 0


    start = timer()
    action_not_same = False
    # wait until the ball and other player is in the game
    while episode_number<=50:
        #####################################
        # hopefully loop will look like this

        # dsImg = downsample(observation)
        # pongStates.updateStates(dsImg,i)
        # mpcAction = mpc.getMPCPred(pongStates)
        # rlAction = rl.getQPred(pongStates)
        # next_move = chooseAction(pongStates,rlAction,mpcAction)
        # observation, reward, done, info = env.step(next_move)
        # show (ifvisual mode)

        #####################################

        downsampled,ds_nc_u = ds(observation,visualMode)
        
        pStates.updateStates(downsampled,i)
        #pStates.printStates()
        mpcAction,predpos = mpc_h.getMPCPred(pStates)
        #mpcAction = 0
        #phyAction = rl.getPhyAct(pStates)
        rlAction = getQPred(pStates,rl,next_index,action_not_same)
        #print("RL action: ",rlAction)
        if (rlAction != mpcAction):
            action_not_same = True
        #else:
        #    action_not_same = False


        #pStates.printStates()
        #next_move = mpcAction
        next_move,next_index = chooseAction(pStates,rlAction,mpcAction,predpos)

        #print("ball pos: ",pStates.ball[0,1])
        #print("Pred:",predpos)
        #pStates.printStates()

        if abs(pStates.agent-predpos)<5:
            game_states.append(pStates.qs)
            game_actions.append(next_index)

        # get the observation (image), reward, done flag and info (unused)
        observation, reward, done, info = env.step(next_move)

        if reward<0:
            #print("Game States:",*game_states)
            #print("Game actions: ",*game_actions)
            #print("Reward:",reward)
            for ind, val in enumerate(game_states[len(game_states)-4:-1]):
                rl.update(val, game_actions[len(game_states)-4+ind], game_states[len(game_states)-4+ind+1], -1, 0.7, 0.1)
            game_states=[]
            game_actions=[]
            opponent_score-=(reward)
        elif reward>0:
            #print("Game States:",*game_states)
            #print("Game actions: ",*game_actions)
            #print("Reward:",reward)
            for ind, val in enumerate(game_states[:-1]):
                rl.update(val, game_actions[ind], game_states[ind+1], 1, 0.7, 0.1)
            game_states =[]
            game_actions=[]
            agent_score+=reward
        
        if (visualMode):
            bre = showImg(ds_nc_u,downsampled)
            if bre:
                break
        i = i+1

        reward_sum += reward

        if ((i % 500) == 0):
            print("Episode: ",episode_number, "Frame: ",i)
            print("Opponent: ",opponent_score,"Agent: ", agent_score)

        if done or i>=30000:
            observation = env.reset() # reset env
            episode_rewards.append(reward_sum)
            end = timer()
            print ('episode:',episode_number, ' reward total was ',reward_sum,"Time taken:",str(datetime.timedelta(seconds=end-start)))   #. running mean: %f' % (reward_sum, running_reward))
            print("Number of frames: ",i)
            print("Opponent: ",opponent_score,"Agent: ", agent_score)
            print("Episode rewards:")
            print(*episode_rewards)
            if ((opponent_score<21) and (agent_score<21)):
                print("GOT STUCK, DISREGARD EPISODE ",episode_number)
                partial.append(reward_sum)
            else:
                partial.append(None)
            opponent_score=0
            agent_score=0
            start = timer()
            i=0
            reward_sum = 0
            episode_number += 1

    plt.plot(episode_rewards,"bs")
    plt.plot(partial, "ro")
    plt.ylabel('reward')
    plt.xlabel('episode')
    plt.show()
        
    
def showImg(ds_nc_u,downsampled):
    # resize the downsampled version and show it on the screen
    r_dsNoColor = cv2.resize(ds_nc_u,(downsampled.shape[1]*(zfactor*downsample_factor[0]),downsampled.shape[0]*(zfactor*downsample_factor[1])), interpolation=cv2.INTER_AREA)
    cv2.imshow('resized downsampled', r_dsNoColor)
    #model_results.append(r_dsNoColor)
    #cv2.imshow("ds for agent",downsampled[paddlePos[0,1]:paddlePos[0,0],paddlePos[1,1]:paddlePos[1,0]])

    if cv2.waitKey(25) & 0xFF == ord('d'):
        bre = False
        if cv2.waitKey() & 0xFF == ord('q'):
            cv2.imwrite( "./Illustrations/dataModel.jpg", r_dsNoColor )
            #out = cv2.VideoWriter('./Illustrations/mpcFull3.avi',cv2.VideoWriter_fourcc(*'DIVX'),10,(r_dsNoColor.shape[1],r_dsNoColor.shape[0]))

            #for i in range(len(model_results)):
            #    out.write(model_results[i])
            #out.release()
            bre = True
        return bre

def ds(observation,visualMode):
    # remove color
    dsNoColour = copy.deepcopy(observation[:,:,grayscale])
    # downsample
    downsampled = copy.deepcopy(dsNoColour[::downsample_factor[1],::downsample_factor[0]])
    if (visualMode):
        # turn grayscale back into rgb for data visualisation purposes
        ds_nc_u = cv2.cvtColor(downsampled, cv2.COLOR_GRAY2RGB)
        return downsampled,ds_nc_u
    return downsampled, None

def getQPred(pStates,rl,next_move,action_not_same):
    #print("Q state:",pStates.qs, "\nLast Q:", pStates.last_qs)
    #print("\nNext move: ",next_move)
    if action_not_same:
        rl.update(pStates.last_qs, next_move, pStates.qs, 0)
    else:
        rl.update(pStates.last_qs, next_move, pStates.qs, 0.1)
    return indexToAction(rl.action(pStates.qs))

def chooseAction(pongStates, rl, mpc,predpos):
    if(abs(predpos-pongStates.agent)<3):
        return rl, getIndex(rl)
    else:
        return mpc, getIndex(mpc)

def getIndex(action):
    if action==0:
        return 0
    elif action==3:
        return 1
    else:
        return 2
    

if __name__ == '__main__':
    Main()