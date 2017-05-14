import os
import re
import csv
import glob
import pandas as pd
import locale
import pickle
locale.setlocale(locale.LC_ALL, '')

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from sklearn.linear_model import BayesianRidge


column_names = ["CURRENT_FIX_START","CURRENT_FIX_END","CURRENT_FIX_X","CURRENT_FIX_Y","CURRENT_FIX_DURATION"]
# column_names_gaze = ["LEFT_GAZE_X","LEFT_GAZE_Y","RIGHT_GAZE_X","RIGHT_GAZE_Y"]
column_names_gaze = ["LEFT_GAZE_X", "LEFT_GAZE_Y"]
column_dict = {}
skip_val = ['.']

step = 30000
stepsize = 0.5


Health = ["Без заикания",  "Во время лечения","До лечения"]

class EyeDat:

    """ The class that handles the dataset """

    def __init__(self):
        def f(x):
            if x == '.':
                return None
            elif isinstance(x, int):
                return x
            else:
                return locale.atof(x)

        self.data = dict()
        self.pat_l = dict()

        for cond in Health:
            for exp in range(1,5): # 1,2,3,4 Эксперимент
                patients = os.listdir("./Данные айтрекера/{}/".format(cond))
                fnames = [(glob.glob("./*/{}/{}/Fix*_{}*".format(cond, pat, exp))[0],
                           glob.glob("./*/{}/{}/Simple*_{}*".format(cond, pat, exp))[0]) for pat in patients]

                for i in range(len(patients)):
                    # Читаем фиксации
                    Fix = []
                    with open(fnames[i][0]) as csvfile:
                        reader = csv.DictReader(csvfile, delimiter='\t')
                        for tup in reader:
                            Fix.append([tup[fld] for fld in column_names])
                    Simp = []
                    # Читаем Simp`ы
                    with open(fnames[i][1]) as csvfile:
                        local_cols = column_names_gaze
                        reader = csv.DictReader(csvfile, delimiter='\t')
                        for tup in reader:
                            Simp.append([tup[fld] for fld in local_cols] + [reader.line_num - 2])
                    Fix = pd.DataFrame(Fix,columns=column_names)
                    Simp = pd.DataFrame(Simp, columns=column_names_gaze + ["IDX"])
                    Fix = Fix.applymap(locale.atof)
                    Simp = Simp.applymap(f) # Переводим запятые в точкио
                    self.data["{}_{}_{}_Simp".format(cond,exp,i)] = Simp                    
                    for k in range(3):
                        self.data["{}_{}_{}_Fix_{}".format(cond,exp,i,k)] = Fix[Fix["CURRENT_FIX_END"]//step %3 ==k]
                self.pat_l[cond] = len(patients)

    def get_df(self, cond, exp, pat, type, trial=0):
        """
        Get selected DataFrame
        :param exp: 
        :param cond: 
        :param pat: 
        :param type: 
        :param trial: 
        :return: 
        """
        if type == 'Fix':
            return self.data["{}_{}_{}_Fix_{}".format(cond,exp,pat,trial)]
        elif type == 'Simp':
            return self.data["{}_{}_{}_Simp".format(cond, exp, pat)]
        else:
            raise ("No such type!!!")

    def disps(self, cond, exp, trial):
        disps = []
        for pat in range(self.pat_l[cond]):
            fix = self.get_df(cond,exp,pat,'Fix',trial)
            scatter = self.get_df(cond,exp,pat,'Simp')
            disps += self.__get_disps(fix,scatter)
        return disps
        
    def __get_disps(self, fixs, simp):
        disps = []
        for rec in fixs[["CURRENT_FIX_START", "CURRENT_FIX_END"]].itertuples():
            fixation = np.asarray(simp[int(rec[1] / stepsize):int(rec[2] / stepsize)])
            disps.append(dists(fixation).std())

        return disps

    def durations(self, cond, exp, trial):
        durations = []
        for pat in range(self.pat_l[cond]):
            fix = self.get_df(cond, exp, pat, 'Fix', trial)
            durations += fix["CURRENT_FIX_DURATION"]
        return durations
    

def dists(v):
    length = v.shape[0]
    sum_x = np.sum(v[:, 0])
    sum_y = np.sum(v[:, 1])
    mean = np.array([sum_x/length,sum_y/length])
    deltas = v - mean
    return np.apply_along_axis(lambda x:np.linalg.norm(x), 0, deltas)


#
def plot(disps,durations,axis):

    X = disps
    y = durations
    D = np.column_stack((X,y))
    D = D[~(D[:,0]==0)]

    axis.scatter(D[:,0],D[:,1], s = 4)
    axis.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    axis.ticklabel_format(style='sci', axis='x', scilimits=(0,0))

    clf = BayesianRidge()
    clf.fit(np.matrix(D[:,0]).transpose(), D[:,1])
    L=np.linspace(1,max(D[:,0]))
    axis.plot(L,[clf.predict(x)for x in L],'r')

    return D,clf

def fname(cond,exp,trial):
    return "./cache/dat_{}_{}_{}".format(cond,exp,trial)

def main():
    plt.tight_layout
    cache = False
    f=[0,0,0]
    dat = EyeDat()
    pickle.dump(dat)
    for trial in range(3):
        f[trial], axarr = plt.subplots(3, 4,  figsize=(16, 12), dpi=160)
        i=0
        for cond in Health:
            for exp in range(1,5):
                fn = fname(cond,exp,trial)
                if cache and os.path.isfile(fn + ".npy"):
                    print("cache\n")
                    D = np.load(fn + ".npy")
                    disps, durations = D[:, 0], D[:, 1]
                else:
                    disps, durations = dat.disps(cond,exp,trial),dat.durations(cond,exp,trial)
                D,clf = plot(disps, durations, axarr[i, exp-1])
                np.save(fn, D)
            i+=1
        
        plt.savefig("Trial" + str(trial+1))

if __name__ == "__main__":
    main()
