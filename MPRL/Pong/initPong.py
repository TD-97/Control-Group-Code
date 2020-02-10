# -*- coding: utf-8 -*-

# import the system and os
import sys
from os import getcwd
# need to include the other folders in the project
sys.path.append(getcwd() + "\\Original")
sys.path.append(getcwd() + "\\Algorithm")
sys.path.append(getcwd() + "\\Main")
# import important packages
from collections import deque
import gym
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as image
import random
import time
import cv2
from newMDPvsMPC_GR_deterministic import downsample,remove_background,remove_color,indexToAction
import copy
from probabilisticTrajectory import init_PT_db,addState,create_connection,getTrajectory,getTrajectoryAll
import sqlite3
from qlearning import QLearning
from gekko import GEKKO

model_results = []

##########################################################################################################
################################## Mode to run ###########################################################
##########################################################################################################
visualMode = True
learningMode = False
##########################################################################################################
################################## Pong Specific Parameters ##############################################
##########################################################################################################
# colours in RGB values
oppCol = [74,130,213]
agentCol = [92,186,92]
ballCol = [236,236,236]
bgCol = [17,72,144]
maxIter = 30
# grayscale colours
gs_ball = 236
gs_agent = 92
gs_opp = 74
gs_bg = 17

# paddle width in pixels
paddleWidth = 4

# the area of interest for the original image 
# in the form [top left corner] [bottom right corner]
# (x,y),(x,y)
aoi_orig = np.array( [[15,34], [144,196]])

# list of possible actions which our agent can take
actions = [0,2,3]

# parameters for paddle dynamics
alpha = 2
beta = 4

##########################################################################################################
################################## Design Parameters #####################################################
##########################################################################################################
# the rate we downsample in the x and y directions
downsample_factor = np.array( [2,4] )

# max number of times we propogate model
maxIter = 30
zfactor = 4 # the factor we zoom in by when displaying the game

db_file_name = "third_attempt_PT.db" # name of database to use/create

# determines when we start trying to predict where the ball is going
# example if it is 50%, we try to predict where the ball is going when it is in our agent half of the 
# field
areaPredictPercent = 90

# a queue of last positions of the ball, we store the last three of them
last_pos = deque(maxlen=3)

# number of episodes we want to run
numOfEps = 25

# the amount of tolerance to the paddle position being off
paddleTol = 1

##########################################################################################################
################################## Derived Parameters ####################################################
##########################################################################################################

# the downsampled version of the area of interest
aoi_ds = np.array( [[ ( (aoi_orig[0,0] / downsample_factor[0]) + 1), \
    ( (aoi_orig[0,1] / downsample_factor[1]) + 1) ], [ (aoi_orig[1,0] / downsample_factor[0]),\
    (aoi_orig[1,1] / downsample_factor[1]) ] ] ).astype(int)

# the width of the paddle after downsampling
paddleWidth_ds = (paddleWidth / downsample_factor[0]).astype(int)

# area our agents paddle can be
paddle_aoi_full = np.array( [[ (aoi_ds[1,0]-paddleWidth_ds), aoi_ds[0,1] ],\
    [ aoi_ds[1,0], aoi_ds[1,1] ]] ).astype(int)

# x-coordinate just before our agents paddle
# TODO THIS SHOULDN'T BE MINUS 4 NEED SOMETHING ELSE
justLeftOfPaddle = aoi_ds[1,0] - paddleWidth_ds - 2

# x coordinate where we start trying to predict the path of the ball
startPredictPoint=( ((aoi_ds[1,0]-aoi_ds[0,0])*( (100-areaPredictPercent)/100)) + aoi_ds[0,0] ).astype(int)

##########################################################################################################
##########################################################################################################
##########################################################################################################

# main body of program
def Main():
    m,y,u = initGekko(alpha,beta,25,0)
    env = gym.make("PongDeterministic-v4") # skip 4 frames everytime and no random action
    observation = env.reset()
    found = False
    sqlite3.register_adapter(np.int32, lambda val: int(val))
    init_PT_db(db_file_name)
    d = getcwd() + "\\Database\\" + db_file_name # get path to db
    c = create_connection(d)

    observation, reward, done, info = env.step(random.choice(actions))
    # TODO remove this -> aoi_ds = np.array( [[8,12], [72,48]]) # this is in for (x,y) (x,y)

    # at 4 (in y direction) it never seems to not find it and it is always 1 pixel
    # at 3, it is sometimes 2 pixels in y direction
    # at 5 sometimes it disapears
    
    # Debug statements
    print("Area of Interest:\n",aoi_orig)
    print("Downsampled aoi:\n",aoi_ds)
    print("Paddle area:\n",paddle_aoi_full)
    
    # initialise some variables
    i=0                     # the time step of the episode we are currently on
    found3 = False          # boolean to see if we have found the last three position of the ball
    next_move = 0           # the next move our agent will take
    episode_rewards = []    # a list of rewards we have got in prev episodes
    reward_sum = 0          # the sum of rewards from the current episode
    episode_number=1        # the episode number we are on
    last_action = deque(maxlen=3)
    last_action.append(0)
    last_action.append(0)
    last_action.append(0)
    if visualMode:
        grayscale = 0
    else:
        grayscale = 2
    paddle_aoi = copy.deepcopy(paddle_aoi_full)
    foundAgent=False
    foundBall=False

    # qlearning stuff
    index_q = 0
    controller = QLearning(ball_x=82, ball_y=82, ai_pos_y=84, v_x=11, v_y=11, n_action=3)
    qstate = [0,0,0,0,0]
    game_actions = []
    game_states = []

    # wait until the ball and other player is in the game
    while episode_number<=numOfEps:
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

        #next_move=random.choice(actions)
        next_move=0
        if (visualMode):
            # allow opencv to interperet the image
            observation = cv2.cvtColor(observation, cv2.COLOR_BGR2RGB)
        
        # remove color
        dsNoColour = copy.deepcopy(observation[:,:,grayscale])
        # downsample
        downsampled = copy.deepcopy(dsNoColour[::downsample_factor[1],::downsample_factor[0]])

        # turn grayscale back into rgb for data visualisation purposes
        ds_nc_u = cv2.cvtColor(downsampled, cv2.COLOR_GRAY2RGB)
        
        # if we found the ball in the last frame, we only check a small area around the ball, rather than the whole frame
        if foundBall:
            foundBall,pos = findBall(downsampled,gs_ball,aoi)
        # otherwise check the whole downsampled frame
        else:
            foundBall,pos = findBall(downsampled,gs_ball,aoi_ds)
        #print(downsampled[paddlePos[0,1]:paddlePos[1,1],paddlePos[0,0]:paddlePos[1,0]])
        if foundAgent:
            paddlePos = findAgent(downsampled,gs_agent,paddle_aoi)
        else:
            paddlePos = findAgent(downsampled,gs_agent,paddle_aoi_full)
        
        # if we have found the ball in this time step
        if foundBall:

            # append the ball position to queue
            last_pos.append([pos[0,0],pos[0,1],i])
            
            # check if we have found the last three positions consecutively
            if len(last_pos)>2:
                if (last_pos[2][2]-last_pos[1][2] == 1) and (last_pos[1][2]-last_pos[0][2] == 1):
                    found3 = True
                else:
                    found3= False
            else:
                found3 = False
            #print("Position:\n",pos)

            aoi = findaoi(pos)
        else:
            #print("Could not find at timestep:", i)
            #aoi = np.empty([2,2])
            last_pos.append([-1,-1,-1])

        if paddlePos!=-1:
            #print("found agent")
            foundAgent=True
            # the area to search for the ball in the next time step
            paddle_aoi[0,1] = paddlePos-agentBuffer
            paddle_aoi[1,1] = paddlePos+agentBuffer

            #print("aoi before rejig:\n",aoi)

            # if this is outside the bounds of the aoi of the whole downsampled version set it equal to it 
            # since we do not want to search outside this area
            if (paddle_aoi[0,1]<paddle_aoi_full[0,1]):
                paddle_aoi[0,1] = paddle_aoi_full[0,1]
            if (paddle_aoi[1,1]>paddle_aoi_full[1,1]):
                paddle_aoi[1,1] = paddle_aoi_full[1,1]
            #print("Next paddle aoi: ",paddle_aoi," paddle position: ",paddlePos," paddle aoi full: ",paddle_aoi_full)
            
        else:
            foundAgent=False

        if (found3):
            #print("found last three. Last position: ",last_pos)
            
            # we want to predict the trajectory if 1. the ball is coming towards us. 2. it is past a certain point
            # as specified by startPredictPoint 3. We detected the ball position during this timestep
            if ( ((last_pos[1][0] - last_pos[0][0]) > 0) and last_pos[1][0]>startPredictPoint and i==last_pos[2][2]):
                #print("Ball is coming towards us, and in our half")
                
                # find the velocity at the last few timesteps
                velxk1 = last_pos[2][0] - last_pos[1][0]
                velyk1 = last_pos[2][1] - last_pos[1][1]
                velxk = last_pos[1][0] - last_pos[0][0]
                velyk = last_pos[1][1] - last_pos[0][1]
                
                # debug statement
                #if abs(velxk1)>6:
                #    print("last Position: ",last_pos)
                
                # the last two states according to our data
                statek = (last_pos[1][0].astype(int),last_pos[1][1].astype(int),velxk.astype(float),velyk.astype(float))
                statek1 = (last_pos[2][0].astype(int),last_pos[2][1].astype(int),velxk1.astype(float),velyk1.astype(float))

                #print("adding states\nstatek: ",statek,"\nstatek1: ",statek1)

                if learningMode:
                    # add this data to the database
                    addState(statek,statek1,c)

                # get the prediction of the ball trajectory
                ball_y,T = getTrajectory(statek1,justLeftOfPaddle,maxIter,c)

                # get all the points the ball will hit on its trajectory
                # for visualisation purposes only
                if (visualMode):
                    prediciton = getTrajectoryAll(statek1,justLeftOfPaddle,maxIter,c)

                    # this is for visualisation, plot the trajectory the agent thinks the ball will take
                    for n in prediciton:
                        ds_nc_u[n[1],n[0]] = (40,166,255)
                    #print("Prediction: ",ball_y)

                # if no prediction, do nothing
                if (ball_y == None):
                    #print("No Prediction at time: ",i)
                    next_move=0
                    last_action.append(next_move)
                
                # if the agent isn't found make random choice TODO justify this
                elif(paddlePos==-1):
                    next_move=random.choice(actions)
                    # debug stuff
                    #print("Can't see agent at time: ",i)
                    #cv2.imshow("Test",downsampled[paddlePos[0,1]:paddlePos[0,0],paddlePos[1,1]:paddlePos[1,0]])
                    #if cv2.waitKey() & 0xFF == ord('q'):
                    #    break

                # this is where I need to put MPC
                else:
                    nextU = nxtMPC(ball_y,paddlePos,m,y,u)
                    nextInd = int(round(nextU))
                    next_move = indexToAction(nextInd)                    
                    print(nextU)

                if visualMode:
                    if (ball_y!=None):
                    # if we have a trajectory guess, paint the pixel red 
                    # visualisation only  
                        ds_nc_u[ball_y,justLeftOfPaddle] = (0,0,255)

        # get the observation (image), reward, done flag and info (unused)
        observation, reward, done, info = env.step(next_move)

        if (visualMode):
            bre = showImg(ds_nc_u)
            if bre:
                break

        i = i+1
        reward_sum += reward

        # stop the game from getting stuck in infinite loop, or reset when done an episode
        if done or i>10000:
            observation = env.reset() # reset env
            episode_rewards.append(reward_sum) # add episode reward to reward list
            print ('episode:',episode_number, ' reward total was %f' %(reward_sum))   #. running mean: %f' % (reward_sum, running_reward))
            reward_sum = 0
            i=0
        
            episode_number += 1

        if not visualMode:
            if i%100==0:
                print("Time: ",i,". Reward sum: ",reward_sum)

    print("commiting and closing")
    c.commit()
    c.close()

# don't need this anymore
def findBallRGB(obs,col,aoi):
    found = False
    pos = np.array([ [0, 0], [0,0] ])
    if(obs.shape[2] != 3):
        print("Error, obs is not in RGB format")
    for i in range(aoi[0,1],aoi[1,1]):
        for j in range(aoi[0,0],aoi[1,0]):
            if (np.array_equal(col,obs[i,j]) ):
                #if ( (i>=aoi[0,1] and i<=aoi[1,1]) and (j>=aoi[0,0] and j<=aoi[1,0]) ):
                if found == False:
                    pos[0,0]=j
                    pos[1,0]=j
                    pos[0,1]=i
                    pos[1,1]=i
                    found=True
                if j < pos[0,0]:
                    pos[0,0] = j
                elif i < pos[0,1]:
                    pos[0,1] = i
                elif j> pos[1,0]:
                    pos[1,0] = j
                elif i> pos[1,1]:
                    pos[1,1] = i

                    #print("Position: ")
                    #print(i,j)
                    #print("Color: ")
                    #print(obs[i,j])
            #if (np.array_equal(col,obs[i,j])):
            #    pos = (i,j)
            #    found = True
            #    break
    if found:
        return pos
    elif not found:
        return pos

# function for finding the ball
def findBall(obs,col,aoi):
    found = False
    pos = np.array([ [-1, -1], [-1,-1] ])
    for i in range(aoi[0,1],aoi[1,1]):
        for j in range(aoi[0,0],aoi[1,0]):
            if (col == obs[i,j] ):
                if ( (i>=aoi[0,1] and i<=aoi[1,1]) and (j>=aoi[0,0] and j<=aoi[1,0]) ):
                    if found == False:
                        pos[0,0]=j
                        pos[1,0]=j
                        pos[0,1]=i
                        pos[1,1]=i
                        found=True
                    if j < pos[0,0]:
                        pos[0,0] = j
                    elif i < pos[0,1]:
                        pos[0,1] = i
                    elif j> pos[1,0]:
                        pos[1,0] = j
                    elif i> pos[1,1]:
                        pos[1,1] = i
    return found, pos

def findAgent(obs,col,aoi):
    pos = np.array([ [-1, -1], [-1,-1] ])
    found=False
    for i in range(aoi[0,1],aoi[1,1]):
        for j in range(aoi[0,0],aoi[1,0]):
            if (col == obs[i,j] ):
                if ( (i>=aoi[0,1] and i<=aoi[1,1]) and (j>=aoi[0,0] and j<=aoi[1,0]) ):
                    if found == False:
                        pos[0,0]=j
                        pos[1,0]=j
                        pos[0,1]=i
                        pos[1,1]=i
                        found=True
                    if j < pos[0,0]:
                        pos[0,0] = j
                    elif i < pos[0,1]:
                        pos[0,1] = i
                    elif j> pos[1,0]:
                        pos[1,0] = j
                    elif i> pos[1,1]:
                        pos[1,1] = i
    return ((pos[0,1]+pos[1,1])/2).astype(int)

def nxtMPC(ball_y,paddle,m,y,u):

    y.MEAS = paddle
    y.sphi = ball_y+1
    y.splo = ball_y-1
    m.solve(False)

    #print("Paddle: "+str(paddle))
    #print("Ball: "+str(ball_y))
    #print("Control Plan: "+str(u.VALUE))
    #print("Control: "+str(u.NEWVAL))

    return u.NEWVAL

def findaoi(pos):
    # the area to search for the ball in the next time step
    aoi = np.array( [ [pos[0,0]-ballBuffer, pos[0,1]-ballBuffer] , [pos[1,0]+ballBuffer, pos[0,1]+ballBuffer] ] )

    #print("aoi before rejig:\n",aoi)

    # if this is outside the bounds of the aoi of the whole downsampled version set it equal to it 
    # since we do not want to search outside this area
    if (aoi[0,0]<aoi_ds[0,0]):
        aoi[0,0] = aoi_ds[0,0]
    if (aoi[0,1]<aoi_ds[0,1]):
        aoi[0,1] = aoi_ds[0,1]
    if (aoi[1,0]>aoi_ds[1,0]):
        aoi[1,0] = aoi_ds[1,0]
    if (aoi[1,1]>aoi_ds[1,1]):
        aoi[1,1] = aoi_ds[1,1]
    return aoi

def initGekko(alpha,beta,paddle,last_move):
    m = GEKKO()
    m.WEB = 0
    m.options.SOLVER=1 # APOPT is an MINLP solver

    # optional solver settings with APOPT
    m.solver_options = ['minlp_maximum_iterations 500', \
                        # minlp iterations with integer solution
                        'minlp_max_iter_with_int_sol 10', \
                        # treat minlp as nlp
                        'minlp_as_nlp 0', \
                        # nlp sub-problem max iterations
                        'nlp_maximum_iterations 50', \
                        # 1 = depth first, 2 = breadth first
                        'minlp_branch_method 1', \
                        # maximum deviation from whole number
                        'minlp_integer_tol 0.1', \
                       # covergence tolerance
                        'minlp_gap_tol 0.01']
    m.time = [0,1,2,3]
    # control variable
    u = m.MV(lb=-1,ub=1,integer=True)
    u.DCOST = 0.1
    # parameters
    alpha = m.Param(value=alpha)
    beta = m.Param(value=beta)
    # need need the last control vector
    ulast = m.Var()
    m.delay(u,ulast,1)

    #variable
    y = m.CV(paddle)

    #equation
    m.Equation(y.dt() == (alpha*u) + (beta*ulast))

    #options
    m.options.IMODE = 6
    m.options.NODES = 2
    m.options.CV_TYPE = 1

    y.STATUS = 1
    y.FSTATUS = 1

    # to do get this from input
    y.SPHI = 25
    y.SPLO = 25
    y.TAU = 0
    y.TR_INIT = 2

    u.STATUS = 1
    u.FSTATUS = 0
    
    return m,y,u

def showImg(ds_nc_u):
    # resize the downsampled version and show it on the screen
    r_dsNoColor = cv2.resize(ds_nc_u,(downsampled.shape[1]*(zfactor*downsample_factor[0]),downsampled.shape[0]*(zfactor*downsample_factor[1])), interpolation=cv2.INTER_AREA)
    cv2.imshow('resized downsampled', r_dsNoColor)
    model_results.append(r_dsNoColor)
    #cv2.imshow("ds for agent",downsampled[paddlePos[0,1]:paddlePos[0,0],paddlePos[1,1]:paddlePos[1,0]])

    if cv2.waitKey(25) & 0xFF == ord('d'):
        bre = False
        if cv2.waitKey() & 0xFF == ord('q'):
            cv2.imwrite( "./Illustrations/dataModel.jpg", r_dsNoColor )
            out = cv2.VideoWriter('./Illustrations/mpcFull3.avi',cv2.VideoWriter_fourcc(*'DIVX'),10,(r_dsNoColor.shape[1],r_dsNoColor.shape[0]))

            for i in range(len(model_results)):
                out.write(model_results[i])
            out.release()
            bre = True
        return bre

if __name__ == '__main__':
    Main()