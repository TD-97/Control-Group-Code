import numpy as np
from collections import deque
import sqlite3

class pongStates():
    ballaoiFull = np.array([[8,9],[72,49]])
    agentaoiFull = np.array([[70,9],[72,49]])
    activeArea = 40
    def __init__(self,ballCol,agentCol):
        self.ballaoi = self.ballaoiFull
        self.agentaoi = self.agentaoiFull
        self.ballCol = ballCol
        self.agentCol = agentCol

        self.foundBall = False
        self.foundAgent = False
        self.found3 = False
        self.ball = np.array([ [-1, -1], [-1,-1] ])
        self.agent = -1
        self.ballBuffer = 10
        self.agentBuffer = 8
        self.ballPred = -1
        # a queue of last positions of the ball, we store the last three of them
        self.last_pos = deque(maxlen=3)
        self.last_pos.append([0,0,0])
        self.last_pos.append([0,0,0])
        self.last_pos.append([0,0,0])
        self.velxk = 0
        self.velxk1 = 0
        self.velyk = 0
        self.velyk1 = 0
        self.statek1 = (0,0,0,0)
        self.statek = (0,0,0,0)

        ## need to add qlearning states here ##
        self.qs = (self.ball[0,0], self.ball[0,1], self.agent, self.velxk1, self.velyk1)
        self.last_qs = self.qs
        #######################################

    def findObj(self,obs,col,aoi):
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
        return pos,found

    def updateStates(self,img,i):
        # find the ball
        self.ball,self.foundBall = self.findObj(img,self.ballCol,self.ballaoi)
        # find the agent
        self.agent,self.foundAgent = self.findObj(img,self.agentCol,self.agentaoi)
        self.agent = ((self.agent[0,1]+self.agent[1,1])/2).astype(int)

        self.update_aoi()

        self.updateLastPos(i)

        self.updateVel()

        self.updateF3()

        self.updateStatek()

        self.updateQL()

    def updateQL(self):
        self.last_qs = self.qs
        if self.found3:
            self.qs = (self.ball[0,0], self.ball[0,1], self.agent, self.velxk1+4, self.velyk1+4)
        else:
            self.qs = (self.ball[0,0], self.ball[0,1], self.agent, 0, 0)

    def update_aoi(self):
        if (self.foundBall):
            # the area to search for the ball in the next time step
            self.ballaoi = np.array( [ [self.ball[0,0]-self.ballBuffer, self.ball[0,1]-self.ballBuffer] , [self.ball[1,0]+self.ballBuffer, self.ball[0,1]+self.ballBuffer] ] )

            # if this is outside the bounds of the aoi of the whole downsampled version set it equal to it 
            # since we do not want to search outside this area
            if (self.ballaoi[0,0]<self.ballaoiFull[0,0]):
                self.ballaoi[0,0] = self.ballaoiFull[0,0]
            if (self.ballaoi[0,1]<self.ballaoiFull[0,1]):
                self.ballaoi[0,1] = self.ballaoiFull[0,1]
            if (self.ballaoi[1,0]>self.ballaoiFull[1,0]):
                self.ballaoi[1,0] = self.ballaoiFull[1,0]
            if (self.ballaoi[1,1]>self.ballaoiFull[1,1]):
                self.ballaoi[1,1] = self.ballaoiFull[1,1]
        else:
            self.ballaoi=self.ballaoiFull

        if (self.foundAgent):
            # the area to search for the ball in the next time step
            #self.agentaoi[0,1] = self.agent-self.agentBuffer
            #self.agentaoi[1,1] = self.agent+self.agentBuffer

            self.agentaoi = np.array([[self.agentaoi[0,0] , self.agentaoi[0,1]-self.agentBuffer] , [self.agentaoi[1,0], self.agent+self.agentBuffer ]])

            # if this is outside the bounds of the aoi of the whole downsampled version set it equal to it 
            # since we do not want to search outside this area
            if (self.agentaoi[0,1]<self.agentaoiFull[0,1]):
                self.agentaoi[0,1] = self.agentaoiFull[0,1]
            if (self.agentaoi[1,1]>self.agentaoiFull[1,1]):
                self.agentaoi[1,1] = self.agentaoiFull[1,1]
        else:
            self.agentaoi=self.agentaoiFull

    def updateLastPos(self,i):
        if (self.foundBall):
            self.last_pos.append([self.ball[0][0],self.ball[0][1],i])
        else:
            self.last_pos.append([0,0,0])

    def updateVel(self):
        self.velxk1 = self.last_pos[2][0] - self.last_pos[1][0]
        self.velyk1 = self.last_pos[2][1] - self.last_pos[1][1]
        self.velxk = self.last_pos[1][0] - self.last_pos[0][0]
        self.velyk = self.last_pos[1][1] - self.last_pos[0][1]

    def updateF3(self):
        if len(self.last_pos)>2:
            #print(self.last_pos)
            if (self.last_pos[2][2]-self.last_pos[1][2] == 1) and (self.last_pos[1][2]-self.last_pos[0][2] == 1):
                self.found3 = True
            else:
                self.found3= False
        else:
            self.found3 = False
    
    def updateStatek(self):
        
        self.statek = (self.last_pos[1][0],\
            self.last_pos[1][1],self.velxk,\
                self.velyk)
        self.statek1 = (self.last_pos[2][0],\
            self.last_pos[2][1],self.velxk1,\
                self.velyk1)


    def printStates(self):
        print("Current States:\nball:",self.ball)
        print("Velxk: ",self.velxk,"velxk1:",self.velxk)
        print("Found: ",self.found3)
        #self.ballaoi = ballaoiFull
        #self.agentaoi = agentaoiFull
        #self.ballCol = ballCol
        #self.agentCol = agentCol

        #self.foundBall = False
        #self.foundAgent = False
        #self.found3 = False
        #self.ball = np.array([ [-1, -1], [-1,-1] ])
        #self.agent = -1
        #self.ballBuffer = 10
        #self.agentBuffer = 8
        #self.ballPred = -1
        ## a queue of last positions of the ball, we store the last three of them
        #self.last_pos = deque(maxlen=3)
        #self.last_pos.append([0,0,0])
        #self.last_pos.append([0,0,0])
        #self.last_pos.append([0,0,0])
        #self.velxk = 0
        #self.velxk1 = 0
        #self.velyk = 0
        #self.velyk1 = 0")

         


