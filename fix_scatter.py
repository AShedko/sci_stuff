import os
import re
import csv
import glob
import pandas as pd
from enum import Enum

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

column_names = ['CURRENT_SAC_END_TIME', 'CURRENT_SAC_AVG_VELOCITY', 'CURRENT_SAC_PEAK_VELOCITY', 'CURRENT_SAC_DURATION']
column_names_fix = ['CURRENT_FIX_END', 'CURRENT_FIX_DURATION']
column_dict = {}
skip_val = ['.']

step = 30000

Health = ["Без заикания", "До лечения", "Во время лечения"]

class EyeDat:
    """ The class that handles the dataset """
    def __init__(self,health,exp,trial):
        self.health = health
        self.exp = exp
        patiets = os.listdir("./Eyetrack/{}".format(health))
        fnames = [(glob.glob("./Eyetrack/{}/{}/Fix*_{}*".format(health,pat,exp)),glob.glob("./Eyetrack/{}/{}/Simple*_{}*".format(health,pat,exp))) for pat in patients]
        fix_df_s = [pd.read_csv(n[0],delimiter='\t',decimal=',') for n in fnames]
        simpl_df_s =  [pd.read_csv(n[1],delimiter='\t',decimal=',') for n in fnames]
        lst = []
        for df in df_s:
            lst.append(df[(df["CURRENT_FIX_END"] // step) % 3 == trial])
        df = pd.concat(lst)
        self.df = df

    def hist_calc(self, axes):
#         mean = np.array((self.df["CURRENT_FIX_X"].mean(), self.df["CURRENT_FIX_Y"].mean()))
#         pairs = ([np.linalg.norm(x - mean) for x in np.array((self.df["CURRENT_FIX_X"],self.df["CURRENT_FIX_Y"])).transpose()],self.df["CURRENT_FIX_DURATION"])
        pairs = (self.disps, self.df["CURRENT_FIX_DURATION"])
        axes.scatter(pairs[0],pairs[1], s = 4)
        axes.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
        if self.exp == 1 : axes.set_ylabel(self.health)
        if self.health == "Во время лечения" : axes.set_xlabel("Эксперимент "+ str(self.exp))
        return pairs

def __main__(argc, argv):
    f, axarr = plt.subplots(3, 4)
    i = 0
    for cond in Health:
        for exp in range(1,5):
            dat = EyeDat(cond,exp,0)
            dat.hist_calc(axarr[i,exp-1])
            i+=1

    f.set_label("Длительность фиксации")
    f.show()
