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
column_names_gaze = ["LEFT_GAZE_X","LEFT_GAZE_Y","RIGHT_GAZE_X","RIGHT_GAZE_Y"]
column_dict = {}
skip_val = ['.']

step = 30000
# stepsize = 0.5

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
        self.stepsize = dict()
        self.empty_slices = 0

        for cond in Health:
            for exp in range(1,5): # 1,2,3,4 Эксперимент
                patients = sorted(os.listdir("./Данные айтрекера/{}/".format(cond)))
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
                        reader = csv.DictReader(csvfile, delimiter='\t')
                        for tup in reader:
                            Simp.append([tup[fld] for fld in column_names_gaze] + [str(reader.line_num - 2)])

                    Fix = pd.DataFrame(Fix,columns=column_names)
                    Simp = pd.DataFrame(Simp, columns=column_names_gaze + ["IDX"])
                    Fix = Fix.applymap(locale.atof)
                    Simp = Simp.loc[:, (Simp != '.').any(axis=0)]
                    Simp = Simp.applymap(f) # Переводим запятые в точки
                    self.data["{}_{}_{}_Simp".format(cond,exp,i)] = Simp                    
                    for k in range(3):
                        self.data["{}_{}_{}_Fix_{}".format(cond,exp,i,k)] = Fix[Fix["CURRENT_FIX_END"]//step %3 ==k]
                    self.stepsize["{}_{}_{}".format(cond,exp,i)] = len(Simp)/Fix.tail(1)["CURRENT_FIX_END"]
                self.pat_l[cond] = len(patients)

    def get_stepsize(self,cond,exp,i):
        return self.stepsize["{}_{}_{}".format(cond,exp,i)]

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
            return self.data["{}_{}_{}_Fix_{}".format(cond, exp, pat, trial)]
        elif type == 'Simp':
            return self.data["{}_{}_{}_Simp".format(cond, exp, pat)]
        else:
            raise ("No such type!!!")

    def disps(self, cond, exp, trial):
        disps = []
        for pat in range(self.pat_l[cond]):
            fix = self.get_df(cond,exp,pat,'Fix',trial)
            simp = self.get_df(cond,exp,pat,'Simp')
            disps += self.__get_disps(fix,simp, self.get_stepsize(cond,exp, pat))
        return disps
        
    def __get_disps(self, fixs, simp, stepsize):
        disps = []

        for rec in fixs[["CURRENT_FIX_START", "CURRENT_FIX_END"]].itertuples():
            i1,i2 = int(rec[1] * stepsize), int(rec[2] * stepsize)
            fixation = np.asarray(simp[i1:i2])
            fixation = fixation[:,:2]
            res = dists(fixation).std()
            # if len(fixation)==0 :
            #     self.empty_slices += 1
                # print(res,fixation)
            disps.append(res)

        return disps

    def durations(self, cond, exp, trial):
        durations = []
        for pat in range(self.pat_l[cond]):
            fix = self.get_df(cond, exp, pat, 'Fix', trial)
            durations += [fix["CURRENT_FIX_DURATION"]]
        return pd.concat(durations)
    

def dists(v):
    length = v.shape[0]
    if np.isnan(v).any():
        return np.asarray(np.nan)
    sum_x = np.sum(v[:, 0])
    sum_y = np.sum(v[:, 1])
    mean = np.array([sum_x/length,sum_y/length])
    # mean = np.nanmean(v)
    deltas = v - mean
    return np.apply_along_axis(lambda x:np.linalg.norm(x), 0, deltas)


#
def plot(disps,durations,axis):

    X = disps
    y = durations
    D = np.column_stack((X,y))
    D = D[~np.isnan(D).any(axis=1)]
    D = D[~(D==0).any(axis=1)]

    axis.scatter(D[:,0],D[:,1], s = 4)
    axis.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    axis.ticklabel_format(style='sci', axis='x', scilimits=(0,0))

    clf = BayesianRidge()
    clf.fit(np.matrix(D[:,0]).transpose(), D[:,1])
    L = np.linspace(1,max(D[:,0]))
    axis.plot(L,[clf.predict(x)for x in L],'r')

    return D, clf

def hist(disps,axis):

    X = np.array(disps)
    D = X[~np.isnan(X)]
    D = D[~(D==0)]

    R = axis.hist(D,bins =70 )
    axis.ticklabel_format(style='sci', axis='y', scilimits=(0,0))

    return R

def fname(cond,exp,trial):
    return "./cache/dat_{}_{}_{}".format(cond,exp,trial)


def save_object(obj, filename):
    with open(filename, 'wb') as output:
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)


def load_object(filename):
    with open(filename, 'rb') as inp:
        return pickle.load(inp)


def main():
    plt.tight_layout
    # act = 0
    f=[0,0,0]

    if os.path.isfile("EyeDump.pcl"):
        dat = load_object("EyeDump.pcl") # From cache
        dat.empty_slices = 0
    else:
        dat = EyeDat()
        save_object(dat,'EyeDump.pcl')
    for trial in range(3):
        f[trial], axarr = plt.subplots(3, 4,  figsize=(16, 12), dpi=160)
        i=0
        for cond in Health:
            for exp in range(1,5):
                # axarr[i, exp-1].set_ylim(0,2000)
                axarr[i, exp - 1].set_xlim(0, 2000)

                disps = dat.disps(cond,exp,trial)
                hist(disps,axarr[i, exp - 1])

                # disps, durations = dat.disps(cond,exp,trial), dat.durations(cond,exp,trial)
                # D, clf = plot(disps, durations, axarr[i, exp-1])
            i+=1
        
        plt.savefig("Trial" + str(trial+1))
    # print(dat.empty_slices)

if __name__ == "__main__":
    main()