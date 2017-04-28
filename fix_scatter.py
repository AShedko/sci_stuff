import os
import re
import csv
import glob
import pandas as pd
import 
from enum import Enum

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from sklearn.linear_model import BayesianRidge

column_names = ['CURRENT_SAC_END_TIME', 'CURRENT_SAC_AVG_VELOCITY', 'CURRENT_SAC_PEAK_VELOCITY', 'CURRENT_SAC_DURATION']
column_names_fix = ['CURRENT_FIX_END', 'CURRENT_FIX_DURATION']
column_dict = {}
skip_val = ['.']

step = 30000

Health = ["Без заикания", "До лечения", "Во время лечения"]

def dists(v):
    mean = np.mean(v)
    return [np.linalg.norm(x - mean) for x in v]

class EyeDat:

    """ The class that handles the dataset """
    def __init__(self, health, exp, trial):
        self.health = health
        self.exp = exp
        patients = os.listdir("./Данные айтрекера/{}/".format(health))
        fnames = [(glob.glob("./*/{}/{}/Fix*_{}*".format(health, pat, exp))[0], glob.glob("./*/{}/{}/Simple*_{}*".format(health, pat, exp))[0]) for pat in patients]
        df_s = [(pd.read_csv(n[0], delim_whitespace =1, decimal=',',usecols=["CURRENT_FIX_START","CURRENT_FIX_END","CURRENT_FIX_X","CURRENT_FIX_Y","CURRENT_FIX_DURATION"])[df[CURRENT_FIX_END]//step % 3 == trial],
                 pd.read_csv(n[1], delimiter = '\t', decimal=',',usecols=["LEFT_GAZE_X","LEFT_GAZE_Y","RIGHT_GAZE_X","RIGHT_GAZE_Y"])) for n in fnames]        
        disps = self.get_disps(df_s)
        self.df = pd.concat([ f[0] for f in df_s ])
        self.disps = disps


    def get_disps(self,df_s):
        disps = []
        for fixs,simp in df_s:
            simp = simp.loc[:,(simp != '.').any(axis=0)]
            simp = simp.stack().str.replace(',','.').unstack()
            simp = simp[simp[simp.columns[0]]!='.'].astype(float)
            stepsize = len(simp)/int(fixs.tail(1)["CURRENT_FIX_END"])            
            for rec in fixs[["CURRENT_FIX_START","CURRENT_FIX_END"]].itertuples():
                fixation = np.asarray(simp[int(stepsize*rec[1]):int(stepsize*rec[2])])                
                disps.append(dists(fixation).var())

        return disps
    
    def plot(self,axis):    
        X = self.disps
        y = self.df["CURRENT_FIX_DURATION"]
        axes.scatter(X,y, s = 4)
        axes.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
        axes.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
        if self.exp == 1 : axes.set_ylabel(self.health)
        if self.health == "Во время лечения" : axes.set_xlabel("Эксперимент "+ str(self.exp))
        # regression
        # TODO !!!
        
        return axes

def __main__(argc, argv):
    f, axarr = plt.subplots(3, 4)
    i = 0
    for cond in Health:
        for exp in range(1,5):
            dat = EyeDat(cond,exp,0)
            dat.plot(axarr[i,exp-1])
            i+=1

    f.set_label("Длительность фиксации")
    f.show()
