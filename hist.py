# -*- coding: utf-8 -*-
import os
import re
import csv
from enum import Enum

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from PIL import Image


column_names = ['CURRENT_SAC_END_TIME', 'CURRENT_SAC_AVG_VELOCITY', 'CURRENT_SAC_PEAK_VELOCITY', 'CURRENT_SAC_DURATION']
column_names_fix = ['CURRENT_FIX_END', 'CURRENT_FIX_DURATION']
column_dict = {}
skip_val = ['.']

step = 30000

cur_dir = os.path.dirname(os.path.abspath(__file__))
csv_dir = os.path.join(cur_dir, 'csv_sac')

csv_dir_fix = os.path.join(cur_dir, 'csv')
res_dir = os.path.join(cur_dir, 'results')


class PType(Enum):
    Healthy = 'Healthy'
    Treat = 'Treat'
    Slut = 'Stut'

class Experiment(Enum):
    Read = 'Read'
    Talk = 'Talk'

class ColType(Enum):
    Number = 1
    NotNumber = 2

class FolderName(Enum):
    Healthy_Read = "Без заикания, чтение"
    Healthy_Talk = "Без заикания, пересказ"
    Stut_Read = "С заиканием, чтение"
    Stut_Talk = "С заиканием, пересказ"
    Treat_Read = "Во время лечения, чтение"
    Treat_Talk = "Во время лечения, пересказ"

def get_init_arr(lst):
    data = {}
    for name in lst:
        data[name] = []
    return data

def get_data_from_dir(dir, solo = False):
    data = []
    dataSolo = get_init_arr(os.listdir(dir))
    i = 0
    for f in os.listdir(dir):
        fullpath = os.path.join(dir, f)
        with open(fullpath) as csv_file:        
            reader = csv.DictReader(csv_file, delimiter='\t')
            for row in reader:
                new_el = {}
                for col_name in column_names_fix:
                    new_el[col_name] = row[col_name]
                    data.append(new_el)
                    dataSolo[f].append(new_el)
            i += 1
            if solo:
                return dataSolo
            else:
                return data

def is_int(val):
    return re.match(r"[-+]?\d+$", val) is not None

def is_float(val):
    return re.match(r"^[\-]?[1-9][0-9]*\.?[0-9]+$", val) is not None

def check_col_type(col_type, old_col_type):
    if old_col_type is None:
        old_col_type = col_type
        return
    if col_type != old_col_type:
        raise Exception('We don\'t expect column type changing. \
                         You should rewrite code')


def get_from_column_dic(col_name, val):
    if col_name in column_dict:
        if val in column_dict[col_name]:
            return column_dict[col_name][val]
        else:
            max_val = max(column_dict[col_name].values())
            column_dict[col_name][val] = max_val + 1
            return max_val + 1
    else:
        column_dict[col_name] = {val: 0}
        return 0

def preprocess_data(data):
    result = [get_init_arr(column_names_fix) for i in range(3)]
    for row in data:
        end = int(row['CURRENT_FIX_END'])
        res_index = (end // step) % 3
        col_type = None
        for col_name in column_names_fix:
            val = row[col_name]
            if val in skip_val:
                continue
            elif is_int(val):
                check_col_type(ColType.Number, col_type)
                val = int(val)
            elif is_float(val):
                check_col_type(ColType.Number, col_type)
                val = float(val)
            else:
                check_col_type(ColType.NotNumber, col_type)
                val = get_from_column_dic(col_name, val)
            result[res_index][col_name].append(val)
    return result

def hist_from_dir(dir, solo = False):
    dir_data = get_data_from_dir(dir)
    if solo:
        solo_dir_data = get_data_from_dir(dir, True)
        solo_data = get_init_arr(os.listdir(dir))
        for f in os.listdir(dir):
            fullpath = os.path.join(dir, f)

            solo_data[f] = preprocess_data(solo_dir_data[f])
            return solo_data
    else:
        data = preprocess_data(dir_data)
        return data

def makeTest(data):
    file = open('Log.txt', 'w')

    pairs = [(PType('Healthy').value,PType('Treat').value),
            (PType('Healthy').value,PType('Stut').value),
            (PType('Treat').value,PType('Stut').value)]
    # Сравниваются люди из разных групп в одном этапе и эксперименте.
    for exp in Experiment:
        for i in range(3):
            for j in range(3):
                file.write(pairs[i][0] + "_" + exp.value + "(" + str(j) + ")"
                    + " - " + pairs[i][1] + "_" + exp.value + "(" + str(j) + ") " + 
                    "mean a = " + str(stats.tmean(data[pairs[i][0] + "_" + exp.value][j]['CURRENT_SAC_AVG_VELOCITY'])) +
                    ", mean b = " + str(stats.tmean(data[pairs[i][1] + "_" + exp.value][j]['CURRENT_SAC_AVG_VELOCITY'])) +
                    ", std dev a = " + str(stats.tstd(data[pairs[i][0] + "_" + exp.value][j]['CURRENT_SAC_AVG_VELOCITY'])) +
                    ", std dev b = " + str(stats.tstd(data[pairs[i][1] + "_" + exp.value][j]['CURRENT_SAC_AVG_VELOCITY'])) + '\n')
                file.write(str(stats.ttest_ind(a= data[pairs[i][0] + "_" + exp.value][j]['CURRENT_SAC_AVG_VELOCITY'],
                        b= data[pairs[i][1] + "_" + exp.value][j]['CURRENT_SAC_AVG_VELOCITY'],
                        equal_var=False))+ '\n')
                file.write(str(stats.mannwhitneyu(data[pairs[i][0] + "_" + exp.value][j]['CURRENT_SAC_AVG_VELOCITY'],
                      data[pairs[i][1] + "_" + exp.value][j]['CURRENT_SAC_AVG_VELOCITY']))+ '\n'+ '\n')
                # Сравниваются люди из одной и той же группы в том же эксперементе в этапах 0 и 1.
        for group in PType: 
            file.write(group.value + "_" + exp.value + "(" + str(0) + ")"
                    + " - " + group.value + "_" + exp.value + "(" + str(1) + ")" + 
                    "mean a = " + str(stats.tmean(data[group.value + "_" + exp.value][0]['CURRENT_SAC_AVG_VELOCITY'])) +
                    ", mean b = " + str(stats.tmean(data[group.value + "_" + exp.value][1]['CURRENT_SAC_AVG_VELOCITY'])) +
                    ", std dev a = " + str(stats.tstd(data[group.value + "_" + exp.value][0]['CURRENT_SAC_AVG_VELOCITY'])) +
                    ", std dev b = " + str(stats.tstd(data[group.value + "_" + exp.value][1]['CURRENT_SAC_AVG_VELOCITY'])) + '\n')
            file.write(group.value + "_" + exp.value + "(" + str(0) + ")"
                + " - " + group.value + "_" + exp.value + "(" + str(1) + ")" + '\n')
            file.write(str(stats.ttest_ind(a= data[group.value + "_" + exp.value][0]['CURRENT_SAC_AVG_VELOCITY'],
                    b= data[group.value + "_" + exp.value][1]['CURRENT_SAC_AVG_VELOCITY'],
                    equal_var=False))+ '\n')
            file.write(str(stats.mannwhitneyu(data[group.value + "_" + exp.value][0]['CURRENT_SAC_AVG_VELOCITY'],
                  data[group.value + "_" + exp.value][1]['CURRENT_SAC_AVG_VELOCITY']))+ '\n'+ '\n')
    file.close()
    pass

def main(solo = False):
    all_data = {}
    all_data_solo = {}
    for ptype in PType:
        for expr in Experiment:
            #makeTestInGroup(fullpath)
            if solo:
                all_data[dir_name] = hist_from_dir(fullpath)
                all_data_solo[dir_name] = hist_from_dir(fullpath, True)
                for f in os.listdir(fullpath):
                    for i in range(2):
                        makeHist(all_data_solo[dir_name][f][i], True)
            else:
                dir_name = ptype.value + '_' + expr.value
                fullpath = os.path.join(csv_dir_fix, dir_name)
                for f in os.listdir(fullpath):
                    for i in range(2):
                        makeHist(all_data[dir_name][f][i])


            
    #makeHist(all_data)

    
    # makeTest(all_data)
    return 0

def makeHist(data, solo = False):
    index = 0
    for col in column_names_fix:
        index=0
        y = makeYMax(data)
        y = makeYMax(data, True) if solo else makeYMax(data)
        if (col == 'CURRENT_SAC_END_TIME') or (col == 'CURRENT_FIX_END'):
            continue
        for folder in data:
            for i, el in enumerate(data[folder]):
                if col == 'CURRENT_SAC_AVG_VELOCITY':
                    plt.hist(avg_vel,  weights=(np.zeros_like(avg_vel)+100./ avg_vel.size), bins=50, range=[1, 9000])
                    plt.xlabel("Скорость, град/с")
                if col == 'CURRENT_SAC_PEAK_VELOCITY':
                    peak_vel = np.array(el[col])
                    plt.hist(peak_vel, weights=(np.zeros_like(peak_vel)+100./ peak_vel.size), bins=50, range=[1, 9000], normed=True)
                    plt.xlabel("Скорость, град/с")
                if col == 'CURRENT_SAC_DURATION':
                    sac_dur = np.array(el[col])
                    plt.hist(sac_dur, weights=(np.zeros_like(sac_dur)+100./ sac_dur.size), bins=50, range=[1, 400], normed=True)
                    plt.xlabel("Длительность в мс")
                if col == 'CURRENT_FIX_DURATION':
                    print(el)
                    fix_dur = np.array(el[col])
                    plt.hist(fix_dur, weights=(np.zeros_like(fix_dur)+100./ fix_dur.size), bins=10, range=[1, 600], normed=True)
                    plt.xlabel("Длительность в мс")

                for f in FolderName:
                    if f.name == folder:
                        plt.title(f.value)

                
                plt.ylabel("Частота, %/100")
                plt.ylim([0,y])
                fig = plt.gcf()
                plt.savefig('hist/'+col+ str(i) + '_' + str(folder) +'.png')
                plt.close()
                #opens an image:
            list_im = ['hist/'+col+ str(0) + '_' + str(folder) +'.png', 
                'hist/'+col+ str(2) + '_' + str(folder) +'.png']
            # Горизонтальное объединение
            imgs    = [ Image.open(i) for i in list_im ]
            imgs_comb = np.hstack( (np.asarray(i) for i in imgs ) )
            imgs_comb = Image.fromarray( imgs_comb)
            imgs_comb.save( 'hist/Tri'+ '_' + str(index) +'.jpg' )  
            index+=1
        # Вертикальное объединение
        list_im = ['hist/Tri'+ '_' + str(0) +'.jpg', 
                'hist/Tri'+ '_' + str(2) +'.jpg', 
                'hist/Tri'+ '_' + str(4) +'.jpg',
                'hist/Tri'+ '_' + str(1) +'.jpg',
                'hist/Tri'+ '_' + str(3) +'.jpg',
                'hist/Tri'+ '_' + str(5) +'.jpg']
            
        imgs    = [ Image.open(i) for i in list_im ]
        imgs_comb = np.vstack( (np.asarray(i) for i in imgs ) )
        imgs_comb = Image.fromarray( imgs_comb)
        imgs_comb.save( 'hist/Ready_'+ str(col)+'.jpg'  )    
            
    pass

def makeYMax(data, solo=False):
    index = 0
    y = 0 
    yMax = 0
    for col in column_names_fix:
        #print(col)
        index=0
        if (col == 'CURRENT_SAC_END_TIME') or (col == 'CURRENT_FIX_END'):
            continue
        for folder in data:
            for i, el in enumerate(data[folder]):
                if col == 'CURRENT_SAC_AVG_VELOCITY':
                    avg_vel = np.array(el[col])
                    y, x, _ = plt.hist(avg_vel,  weights=(np.zeros_like(avg_vel)+1./ avg_vel.size), bins=50, range=[1, 9000])
                if col == 'CURRENT_SAC_PEAK_VELOCITY':
                    peak_vel = np.array(el[col])
                    y, x, _ = plt.hist(peak_vel, weights=(np.zeros_like(peak_vel)+1./ peak_vel.size), bins=50, range=[1, 9000], normed=True)

                if col == 'CURRENT_SAC_DURATION':
                    sac_dur = np.array(el[col])
                    y, x, _ = plt.hist(sac_dur, weights=(np.zeros_like(sac_dur)+1./ sac_dur.size), bins=50, range=[1, 400], normed=True)
                if col == 'CURRENT_FIX_DURATION':
                    print(el)
                    fix_dur = np.array(el[col])
                    y, x, _ = plt.hist(fix_dur, weights=(np.zeros_like(fix_dur)+1./ fix_dur.size), bins=10, range=[1, 600], normed=True)


                if y.max() > yMax:
                    yMax = y.max()
                plt.ylabel("Частота")
                fig = plt.gcf()
                plt.close()

        return yMax

main(True)

