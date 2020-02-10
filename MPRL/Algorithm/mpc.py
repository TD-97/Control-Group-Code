from gekko import GEKKO
from probabilisticTrajectory import init_PT_db,addState,create_connection,getTrajectory,getTrajectoryAll
from newMDPvsMPC_GR_deterministic import indexToAction
import sqlite3
import numpy as np
# import the system and os
import sys
from os import getcwd
from timeit import default_timer as timer
# need to include the other folders in the project
#sys.path.append(getcwd() + "\\Original")
sys.path.append(getcwd() + "\\Algorithm")
sys.path.append(getcwd() + "\\Main")


class mpc():

    def __init__(self,alpha,beta,paddle,last_move,justLeftOfPaddle,lmode):
        #db_file_name = "third_attempt_PT.db" # name of database to use/create
        db_file_name = "pong_v0.db"
        self.justLeftOfPaddle = justLeftOfPaddle
        self.m = GEKKO(remote=False)
        self.learnMode = lmode
        self.m.WEB = 0
        self.m.options.SOLVER=1 # APOPT is an MINLP solver
        #self.m.options.LINEAR=1
        self.maxIter = 30 # this is a parameter

        #optional solver settings with APOPT
        self.m.solver_options = ['minlp_maximum_iterations 500', \
                            # minlp iterations with integer solution
                            'minlp_max_iter_with_int_sol 10', \
                            # treat minlp as nlp
                            'minlp_as_nlp 0', \
                            # nlp sub-problem max iterations
                            'nlp_maximum_iterations 50', \
                            # 1 = depth first, 2 = breadth first
                            'minlp_branch_method 1', \
                            # maximum deviation from whole number
                            'minlp_integer_tol 0.01', \
                        # covergence tolerance
                            'minlp_gap_tol 0.1']
        self.m.time = [0,1,2,3]
        # control variable
        self.u = self.m.MV(lb=-1,ub=1,integer=True)
        self.u.DCOST = 0.1
        # parameters
        alpha = self.m.Param(value=alpha)
        beta = self.m.Param(value=beta)
        # need need the last control vector
        ulast = self.m.Var()
        self.m.delay(self.u,ulast,1)

        #variable
        self.y = self.m.CV(paddle)

        #equation
        self.m.Equation(self.y.dt() == (alpha*self.u) + (beta*ulast))

        #options
        self.m.options.IMODE = 6
        self.m.options.NODES = 2
        self.m.options.CV_TYPE = 1

        self.y.STATUS = 1
        self.y.FSTATUS = 1

        # to do get this from input
        self.y.SPHI = 25
        self.y.SPLO = 25
        self.y.TAU = 0
        self.y.TR_INIT = 2

        self.u.STATUS = 1
        self.u.FSTATUS = 0
        #self.m.options.MAX_TIME = 0.1

        sqlite3.register_adapter(np.int32, lambda val: int(val))
        init_PT_db(db_file_name)
        d = getcwd() + "\\Database\\" + db_file_name # get path to db
        self.c = create_connection(d)
        print(self.m._path)

    def nxtMPC(self,ball_y,paddle,m,y,u):
        error=0
        self.y.MEAS = paddle
        self.y.sphi = ball_y+1
        self.y.splo = ball_y-1
        start = timer()
        try:
            self.m.solve(False)
        except:
            error+=1
        end = timer()

        return self.u.NEWVAL

    def getMPCPred(self,pongStates):
        # if we the states are right, use nxtMPC
        if not self.checkState(pongStates):
            return 0,-1

        predY,criticalT = getTrajectory(pongStates.statek1,self.justLeftOfPaddle,self.maxIter,self.c)
        #print("\n\nPrediciton: ", predY)
        if (self.learnMode):
            addState(pongStates.statek,pongStates.statek1,self.c)
        if (predY!=None):
            nextU = self.nxtMPC(predY,pongStates.agent,self.m,self.y,self.u)
            #nextU = 0
            nextInd = int(round(nextU))
            #print("Action: ",nextInd)
            next_move = indexToAction(nextInd)                    
            return next_move,predY
        # otherwise, return zero
        else:
            return 0,-1

    def checkState(self,pongStates):
        # want to check the ball is heading towards us
        # and that it is a certain distance frome us
        if (pongStates.velxk <= 0 or pongStates.velxk1 <= 0):
            return False
        elif (pongStates.ball[0,0] < pongStates.activeArea):
            return False
        elif(pongStates.found3 is False):
            return False
        else:
            return True