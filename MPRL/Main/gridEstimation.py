# -*- coding: utf-8 -*-

# some_file.py
import sys
# insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(1,r"C:\Users\thoma\git\College\ME_proj\mpcarl\Original")

import gym
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as image
import random
import time
import cv2
#from qlearning import QLearning
from newMDPvsMPC_GR_deterministic import preprocess_observations

def main() :
    env = gym.make("PongDeterministic-v4") # skip 4 frames everytime and no random action
    observation = env.reset()

    i = 0
    print("Here")
    print(env.observation_space)
    while i<500:
        #env.render()
        list = [0,2,3]
        observation, reward, done, info = env.step(random.choice(list))
        print(done)
        print(info)
        cv2.imshow('Image',observation)
        # Press Q on keyboard to  exit
        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.imshow('Image',observation)
            cv2.waitKey()
            print(observation.shape)

            observation = preprocess_observations(observation)

            cv2.imshow('Image',observation)
            cv2.waitKey()
            print(observation.shape)

            observation = cv2.resize(observation,(observation.shape[1]*4,observation.shape[0]*4),interpolation=cv2.INTER_AREA)

            cv2.imshow('Image',observation)
            cv2.waitKey()
            print(observation.shape)

            observation[100:200,100:200] = 255

            cv2.imshow('Image',observation)
            cv2.waitKey()
            print(observation.shape)
        i = i+1
            
main()