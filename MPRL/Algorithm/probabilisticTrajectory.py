# -*- coding: utf-8 -*-

# some_file.py
import sys
from os import getcwd
# insert at 1, 0 is the script path (or '' in REPL)
sys.path.append(getcwd() + "\\Original")
sys.path.append(getcwd() + "\\Pong")
sys.path.append(getcwd() + "\\Main")

import gym
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as image
import random
import time
import cv2
#from qlearning import QLearning
from newMDPvsMPC_GR_deterministic import downsample,remove_background,remove_color
import copy
import sqlite3
from sqlite3 import Error
import pandas as pd

#def initialise_PT():
   # if not (gridShape.shape ==  (2,)):
   #     raise Exception("Error, probabilistic trajectory must be initialised with an numpy array with two entries")
   # if (gridShape[0] > 100) or (gridShape[1] > 100):
   #     raise Exception("Error, too big a grid")

   # print("Making grid ", gridShape[0], " x ", gridShape[1])

def adapt_array(arr):
    out = io.BytesIO()
    np.save(out, arr)
    out.seek(0)
    return sqlite3.Binary(out.read())

def convert_array(text):
    out = io.BytesIO(text)
    out.seek(0)
    return np.load(out)


# making a database to store values
def create_connection(db_file):
    """ create a database connection to the SQLite database
        specified by db_file
    :param db_file: database file
    :return: Connection object or None
    """ 
    conn = None
    try:
        conn = sqlite3.connect(db_file)
        return conn
    except Error as e:
        print(e)
    return conn

def create_table(conn, create_table_sql):
    """ create a table from the create_table_sql statement
    :param conn: Connection object
    :param create_table_sql: a CREATE TABLE statement
    :return:
    """
    try:
        c = conn.cursor()
        c.execute(create_table_sql)
    except Error as e:
        print(e)

def create_state_k(conn, state_k):
    """
    Create a new state_k in the state_k table
    :param conn:
    :param state_k:
    """
    sql = """INSERT OR IGNORE INTO state_k(pos_x,pos_y,vel_x,vel_y) VALUES(?,?,?,?);"""
    try:
        cur = conn.cursor()
        cur.execute(sql, state_k)
    except Error as e:
        print(e)

def create_state_k1(conn, state_k1):
    """
    Create a new project into the projects table
    :param conn:
    :param state_k1:
    """
    sql = """INSERT OR IGNORE INTO state_k1(pos_x,pos_y,vel_x,vel_y,pos_x_k1,pos_y_k1,vel_x_k1,vel_y_k1,nTimesRecorded) VALUES(?,?,?,?,?,?,?,?,?);"""
    sql1 = """UPDATE state_k1 SET nTimesRecorded = nTimesRecorded+1 WHERE pos_x=? AND pos_y=? AND vel_x=? AND vel_y=? AND pos_x_k1=? AND pos_y_k1=? AND vel_x_k1=? AND vel_y_k1=?;"""
    try:
        cur = conn.cursor()
        cur.execute(sql, state_k1)
        cur.execute(sql1, state_k1[0:8])
    except Error as e:
        print(e)

def init_PT_db(database):
    database = getcwd() + "\\Database\\"+ database
    print("Path to DB: ",database)
    """
    Create a new probabilistic trajectory database
    : param database:   the relative path from where the code is being run 
                        to the database that we want to create
    """
 
    sql_create_state_k_table = """ CREATE TABLE IF NOT EXISTS state_k(
                                        pos_x integer,
                                        pos_y integer,
                                        vel_x real,
                                        vel_y real,
                                        PRIMARY KEY(pos_x,pos_y,vel_x,vel_y)
                                    ); """
 
    sql_create_state_k1_table = """CREATE TABLE IF NOT EXISTS state_k1(
                                    pos_x integer,
                                    pos_y integer,
                                    vel_x real,
                                    vel_y real,
                                    pos_x_k1 integer,
                                    pos_y_k1 integer,
                                    vel_x_k1 real,
                                    vel_y_k1 real,
                                    nTimesRecorded integer,
                                    PRIMARY KEY(pos_x,pos_y,vel_x,vel_y,pos_x_k1,pos_y_k1,vel_x_k1,vel_y_k1) FOREIGN KEY (pos_x,pos_y,vel_x,vel_y) REFERENCES state_k (pos_x,pos_y,vel_x,vel_y)
                                );"""
 
    # create a database connection
    conn = create_connection(database)
 
    # create tables
    if conn is not None:
        # create state_k table
        create_table(conn, sql_create_state_k_table)
 
        # create tasks table
        create_table(conn, sql_create_state_k1_table)
        conn.close()
    else:
        print("Error! cannot create the database connection.")
    return

def addState(state_k,state_k1,conn):
    initN = (0,)
    state_k1 = state_k + state_k1 + initN

    #print("state_k: ",state_k," state_k1: ",state_k1[0:8])

    #conn = create_connection(getcwd() + "\\Database\\ProbTraj.db")
    create_state_k(conn,state_k)
    create_state_k1(conn,state_k1)
    conn.commit()
    return

def getTrajectory(state,x,maxIter,conn):
    i=0
    while state[0]<x and i < maxIter:
        state = getNextState(state,conn)
        #print("State: ",state)
        #print("State[0]: ", state[0]," x: ", x)
        if (state[0] == None):
            break
        i=i+1
        
    if (state[1]==0):
        return None,i
    return state[1],i

def getTrajectoryAll(state,x,maxIter,conn):
    i=0
    allStates = []
    while state[0]<x and i < maxIter:
        state = getNextState(state,conn)
        #print("State: ",state)
        #print("State[0]: ", state[0]," x: ", x)
        if (state[0] == None):
            break
        i=+1
        allStates.append([state[0],state[1]])
    return allStates

def getNextState(state,conn):
    sql = """SELECT pos_x_k1,
       pos_y_k1,
       vel_x_k1,
       vel_y_k1
FROM (
   SELECT pos_x_k1,
          pos_y_k1,
          vel_x_k1,
          vel_y_k1,
		  MAX(nTimesRecorded) nTimesRecorded
   FROM state_k1
   WHERE  pos_x=? AND pos_y=? AND vel_x=? AND vel_y = ?);"""
    row = (None,None,None,None)
    try:
        cur = conn.cursor()
        cur.execute(sql, state)
        row = cur.fetchone()
        return row
    except Error as e:
        print(e)
    return row


#if __name__ == '__main__':
    #db_file_name = "ProbTraj.db"
    #print("Initialising db")
    #init_PT_db(db_file_name)
    # create a database connection
    #print("adding states")
    #state_k = (5,12,13.7,14.1)
    #state_k1 = (15,16,17.2,10)
    #d = getcwd() + "\\Database\\" + db_file_name
    #c = create_connection(d)
    #print("Connection: ",c)

    #addState(state_k,state_k1)