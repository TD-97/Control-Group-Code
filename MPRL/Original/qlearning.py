# -*- coding: utf-8 -*-
"""
Created on Sat Mar 30 18:27:45 2019

@author: megha
"""

import random
from math import pi
import numpy as np

class QLearning:
    def __init__(self, ball_x=80, ball_y=80, ai_pos_y=80, v_x=80, v_y=80, n_action=3):

        self.init_constants()
        self.ball_values_x = ball_x
        self.ball_values_y = ball_y
        self.v_x=v_x
        self.v_y=v_y
        self.ai_pos_y = ai_pos_y
        self.n_action = n_action
        
        shape = (ball_x, ball_y, ai_pos_y, v_x, v_y, n_action )
                
        self.Q = self.initial_Q*np.ones(shape, dtype=float)
        
#        for a in range( self.Q.shape[0] ):
#            for b in range(self.Q.shape[1]):             
#                for c in range(self.Q.shape[2]):
#                    for d in  range(self.Q.shape[3]):
#                        for e in range(self.Q.shape[4]):
#                            for f in range(self.Q.shape[5]):
#                                for g in range(self.Q.shape[6]):
#                                    for h in range(self.Q.shape[7]):
#                                        for i in range(self.Q.shape[8]):
#                                            self.Q[(a, b, c, d, e, f, g, h, i)]
#
#                        #for c in range(self.Q.shape[4]):
#                        self.Q[(a, b, i, j, 0)] = 0 # smoother with 2 instead of 0 also 1 works but cart moves more

    def init_constants(self):
        self.initial_Q = 0

    def rand_action(self, state, probability):
        # in order to get action the state passed in as input
        # actions is obtained in form of tuple from which the maximum is selected
        random_value = np.random.uniform()
        if random_value < probability:
            actions = self.Q[tuple(list(state))]

            actions = [ (i, actions[i]) for i in range(len(actions)) ]        
            max_action = max(actions, key=lambda x: x[1])
            
            return max_action[0] # give index of the action selected
            

        else:                
            action = random.randint(0,2)
            return action # give index of the action selected

    def action(self, state):
        # in order to get action the state passed in as input
        # actions is obtained in form of tuple from which the maximum is selected
        
        actions = self.Q[tuple(list(state))]
        #print("\n\naction1: ", actions)
        actions = [ (i, actions[i]) for i in range(len(actions)) ]
        #print("actions: ", actions)
        max_action = max(actions, key=lambda x: x[1])
        #if (max_action[0]!=0):
        #    print("Max action: ", max_action[0])
        return max_action[0] # give index of the action selected

    def update(self, s, a, next_s, r, gamma = 0.9, alpha=0.9):             
        #stores value for that state and action reward + discounted*max action
        #sets r +gamma*max_action value in that state so if the state is revisited then that max value is used for unvisited states the initial value is set at 2 this gets discounted over time=
        max_action = max( list(self.Q[ tuple(next_s) ]) ) # from the list of actions in self.Q[input] select max
        #print(tuple(list(s)+[a]))
        self.Q[ tuple( list(s) + [a] ) ] = (1-alpha)*self.Q[ tuple( list(s) + [a] ) ]  + alpha*( r + gamma * max_action)
	

        
        
