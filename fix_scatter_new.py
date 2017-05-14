import os
import re
import csv
import glob
import pandas as pd
import locale
from enum import Enum

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from sklearn.linear_model import BayesianRidge


column_names = ["CURRENT_FIX_START","CURRENT_FIX_END","CURRENT_FIX_X","CURRENT_FIX_Y","CURRENT_FIX_DURATION"]
column_names_gaze = ["LEFT_GAZE_X","LEFT_GAZE_Y","RIGHT_GAZE_X","RIGHT_GAZE_Y"]
column_dict = {}
skip_val = ['.']

step = 30000
stepsize = 0.5


Health = ["Без заикания",  "Во время лечения","До лечения"]

class EyeDat:

    """ The class that handles the dataset """

    def __init__(self):
        self.data = dict()

        def f(x): #Преобразуем точки в числа
            if x == '.': return None
            elif isinstance(x, int): return x
            else: return locale.atof(x)

        for health in Health:
            for exp in range(1,5): # 1,2,3,4 Эксперимент
                patients = os.listdir("./Данные айтрекера/{}/".format(health))
                fnames = [(glob.glob("./*/{}/{}/Fix*_{}*".format(health, pat, exp))[0],
                           glob.glob("./*/{}/{}/Simple*_{}*".format(health, pat, exp))[0]) for pat in patients]

                for i in range(len(patients)):
                    Fix = []
                    # Читаем фиксации
                    with open(fnames[i][0]) as csvfile:
                        reader = csv.DictReader(csvfile, delimiter='\t')
                        for tup in reader:
                            Fix.append([tup[fld] for fld in column_names])

                    Simp = []
                    # Читаем Simp`ы
                    with open(fnames[i][1]) as csvfile:
                        reader = csv.DictReader(csvfile, delimiter='\t')
                        for tup in reader:
                            Simp.append([tup[fld] for fld in column_names_gaze] + [reader.line_num - 2])
                    Fix = pd.DataFrame(Fix, columns=column_names)
                    Simp = pd.DataFrame(Simp, columns=column_names_gaze + ["IDX"])
                    Fix = Fix.applymap(locale.atof)
                    Simp = Simp.applymap(f)
                    self.data["{}_{}_{}_Simp".format(health,exp,i)] = Simp
                    for k in range(3):
                        self.data["{}_{}_{}_Fix_{}".format(health,exp,i,k)] = Fix[Fix["CURRENT_FIX_END"]//step %3 ==k]

    def get_df(self, exp, health, pat, type, trial):
        """
        Get selected DataFrame
        :param exp: 
        :param health: 
        :param pat: 
        :param type: 
        :param trial: 
        :return: 
        """
        if type == 'Fix':
            return self.data["{}_{}_{}_Fix_{}".format(health,exp,pat,trial)]
        elif type == 'Simp':
            return self.data["{}_{}_{}_Simp".format(health, exp, pat, trial)]
        else:
            raise ("No such type!!!")

def dists(v):
    length = v.shape[0]
    sum_x = np.sum(v[:, 0])
    sum_y = np.sum(v[:, 1])
    mean = np.array([sum_x/length,sum_y/length])
    deltas = v - mean
    return np.apply_along_axis(lambda x:np.linalg.norm(x), 0, deltas)

def get_disps(df_s):
    disps = []
    for fixs,simp in df_s:
        simp = simp.loc[:,(simp != '.').any(axis=0)]
        simp = simp.stack().str.replace(',','.').unstack()
        simp = simp[simp[simp.columns[0]]!='.'].astype(float)
        for rec in fixs[["CURRENT_FIX_START","CURRENT_FIX_END"]].itertuples():
            fixation = np.asarray(simp[int(rec[1]/stepsize):int(rec[2]/stepsize)])
            disps.append(dists(fixation).std())

    return disps
#
# def plot(disps,durations,axis):
#     X = disps
#     y = durations
#     D = np.column_stack((X,y))
#     D = D[~(D[:,0]==0)]
#
#     axis.scatter(D[:,0],D[:,1], s = 4)
#     axis.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
#     axis.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
#
#     clf = BayesianRidge()
#     clf.fit(np.matrix(D[:,0]).transpose(), D[:,1])
#     L=np.linspace(1,max(D[:,0]))
#     axis.plot(L,[clf.predict(x)for x in L],'r')
#     np.save(self.fn, D)
#
#     return D,clf
#
#
#         if cache and os.path.isfile(self.fn+".npy"):
#             print ("cache\n")
#             D = np.load(self.fn+".npy")
#             self.disps, self.durations = D[:,0],D[:,1]
#
#         else:


def main():
    plt.tight_layout
    f=[0,0,0]
    for trial in range(3):
        f[trial], axarr = plt.subplots(3, 4,  figsize=(16, 12), dpi=160)
        i=0
        for cond in Health:
            for exp in range(1,5):
                dat = EyeDat(cond, exp, trial, True) # Caching on!!!
                dat.plot(axarr[i, exp-1])
            i+=1
        plt.savefig("Trial" + str(trial+1))

if __name__ == "__main__":
    main()
