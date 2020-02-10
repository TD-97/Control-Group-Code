#!/usr/bin/python
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
from initPong import findAgent
from gekko import GEKKO
#from mipcl_py.models.CVP import CVP
#from CVP import CVP
import numpy as np
paddle_aoi_full = np.array( [[ 70, 9],[ 72, 49 ]] ).astype(int)
grayscale = 0
#controlSeq = [0,0,-1,1,0,0,0,1,-1,0,1,0,-1,-1,-1,0,0,0,0,1,1,-1,0,0,1,1,1,0,1,0,1,1,0,0,0,1,1,1,1,-1,-1,-1,0,0,-1,0,0,0,0,0,\
#    1,1,0,0,0,0,0,0,-1,0,0,0,0,1,0,0,0,1,1,0,0,-1,-1,-1,0,0,0,1,0,0,0,-1,0,0,0,1,0,-1,0,0,0,0,0,1,0,0,0,0,0]
controlSeq = [0,0,0,-1,-1,0,0,0,1,1,1,1,1,1,0,0,0]
#controlSeq = [0,0,1,0,0,0]
paddle = np.zeros([1,len(controlSeq)])
paddle[0][0]=25
paddle[0][1]=25
len(paddle)

env = gym.make("PongDeterministic-v4") # skip 4 frames everytime and no random action
observation = env.reset()
gs_agent = 92
downsample_factor = np.array( [2,4] )
zfactor = 4

steps = len(controlSeq)-1
i=1
while i < steps:
    next_move = indexToAction(controlSeq[i])

    observation, reward, done, info = env.step(next_move)
    i=i+1
    # allow opencv to interperet the image
    observation_rgb = cv2.cvtColor(observation, cv2.COLOR_BGR2RGB)

    # remove color
    dsNoColour = copy.deepcopy(observation_rgb[:,:,grayscale])
    # downsample
    downsampled = copy.deepcopy(dsNoColour[::downsample_factor[1],::downsample_factor[0]])

    paddle[0][i] = findAgent(downsampled,gs_agent,paddle_aoi_full)

    # turn grayscale back into rgb for data visualisation p
    ds_nc_u = cv2.cvtColor(downsampled, cv2.COLOR_GRAY2RGB)

    r_dsNoColor = cv2.resize(ds_nc_u,(downsampled.shape[1]*(zfactor*downsample_factor[0]),downsampled.shape[0]*(zfactor*downsample_factor[1])), interpolation=cv2.INTER_AREA)
    cv2.imshow('resized downsampled', r_dsNoColor)

    if cv2.waitKey(25) & 0xFF == ord('d'):
        if cv2.waitKey() & 0xFF == ord('q'):
            break
        else:
            continue
    time.sleep(0.1)
cv2.waitKey()
cv2.destroyAllWindows()
#print(paddle)

#print(sum(paddle[0][2:len(controlSeq)]))
#print(sum(paddle[0][1:len(controlSeq)-1]))
#print(sum(controlSeq[1:len(controlSeq)-1]))
#print(sum(controlSeq[0:len(controlSeq)-2]))
#print(paddle[0][2:len(controlSeq)]-paddle[0][1:len(controlSeq)-1])


ua = controlSeq[1:len(controlSeq)-2]
print(ua)
ub = controlSeq[0:len(controlSeq)-3]
print(ub)
pk = paddle[0][2:len(controlSeq)-1]
p_1 = paddle[0][1]

#print(ua)
#print(ub)
#print(pk)
#print(pk_1)
#uam = [x*3 for x in ua]
#ubm = [x*1 for x in ub]
#print("vector of diffs:")
#print(pk - pk_1 - uam - ubm)
#print(sum(abs(pk - pk_1 - uam - ubm)))

f = np.ones([1,len(controlSeq)-1])
f[0][0] = 0
f[0][1] = 0
#print(f)
#print(ua)
#print(ub)

#A = np.array([[sum(ua),sum(ub),-1],[-sum(ua),-sum(ub),-1]])
#b = np.array([[ -paddle[0][len(controlSeq)-1]+paddle[0][2] ] , [ paddle[0][len(controlSeq)-1]-paddle[0][2] ] ])
#f = np.array([0,0,1])

A = np.zeros([((len(controlSeq)-3)*2), len(controlSeq)-1 ])
b = np.zeros([(len(controlSeq)-3)*2,1])
n=0
x=0
#print(pk)
while (x<len(controlSeq)-3):
    A[n][0] = -sum(ua[0:x+1])
    A[n][1] = -sum(ub[0:x+1])
    A[n+1][0] = sum(ua[0:x+1])
    A[n+1][1] = sum(ub[0:x+1])

    A[n][x+2] = -1
    A[n+1][x+2] = -1

    b[n][0] = -pk[x] + p_1
    b[n+1][0] = -p_1 + pk[x]

    n+=2
    x+=1
print("b")
print(b)
print("A")
print(A)
print("f")
print(f)
print("Paddle")
print(paddle)
#print(pk)
#print(pk_1)

import matlab.engine
eng = matlab.engine.start_matlab()

mat_A = matlab.double(A.tolist())
mat_b = matlab.double(b.tolist())
mat_f = matlab.double(f.tolist())
mat_intcon = matlab.double([1,2])

#tf = eng.isprime(37)
#print(tf)
x = eng.intlinprog(mat_f,mat_intcon,mat_A,mat_b)
print("Alpha:",x[0])
print("Beta:",x[1])
#print(x)
print("Error at each timestep:",x[2:])

x = np.asarray(x)

pkEst = np.zeros([1,len(pk)+1],dtype=int)
pkEst[0][0]=25
for j in range(0,len(pk)):
    pkEst[0][j+1] = pkEst[0][j] + x[0]*ua[j] + x[1]*ub[j]
np.insert(pk,[0],[25])
print("Actual paddle position:")
pkReal = paddle[0][1:(len(controlSeq)-1)]
print(pkReal)

time = np.arange(1,len(controlSeq)-1,1)
print(time)
print("Estimated paddle postion:")
print(pkEst)


plt.figure(1)
#plt.subplot(211)
plt.step(time,pkReal)
plt.plot(time,pkReal,'C0o',alpha=0.5)
plt.step(time,pkEst[0])
plt.plot(time,pkEst[0],'C1o',alpha=0.5)
plt.legend(['Actual Paddle','','Estimated Paddle',''])
plt.ylabel("Paddle Position (pixels)")
plt.xlabel("Time steps")
plt.title("Least Absolute Parameter Estimation")

#plt.subplot(212)
#plt.plot(time,controlSeq[1:len(controlSeq)-1])

plt.show()

#
#print(np.transpose(A))
#
#print(b)

#from gekko import GEKKO
#m = GEKKO() # Initialize gekko
#m.options.SOLVER=1  # APOPT is an MINLP solver
#
## optional solver settings with APOPT
#m.solver_options = ['minlp_maximum_iterations 500', \
#                    # minlp iterations with integer solution
#                    'minlp_max_iter_with_int_sol 10', \
#                    # treat minlp as nlp
#                    'minlp_as_nlp 0', \
#                    # nlp sub-problem max iterations
#                    'nlp_maximum_iterations 50', \
#                    # 1 = depth first, 2 = breadth first
#                    'minlp_branch_method 1', \
#                    # maximum deviation from whole number
#                    'minlp_integer_tol 0.05', \
#                    # covergence tolerance
#                    'minlp_gap_tol 0.01']
#
#x = m.axb(A,b,etype='<=',sparse=False)
#m.Obj(sum(x[2: (len(controlSeq)-2) ]))
#
#m.solve(disp=False) # Solve
#
#print(x)