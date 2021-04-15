"""
Usage: Coalitional Game, Approximating the contribution of the i-th feature’s value
Python version: 3.9.X
Author: rauls19
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import pickle
import random
from sklearn.linear_model import LinearRegression
import xgboost as xgb
import progressbar
import os
import datetime
import matplotlib.pyplot as plt
import multiprocessing
from sklearn.neural_network import MLPRegressor

class CoalitionalGame:

    def __init__(self, model, features, data_x, data_y, test_data_x, test_data_y, m = 50):
        self.__m = m
        self.__N = features
        self.__model = model
        self.x_train = data_x
        self.y_train = data_y
        self.x_test = test_data_x
        self.y_test = test_data_y
        self.fi_i_contributions = []
        self.__features_N = []

    def __checkModelAvailability(self):
        x = pd.DataFrame(np.random.choice(range(20), size=(50, 4)))
        y = pd.DataFrame(pd.DataFrame(np.random.choice(range(10), size=(50, 1))))
        try:
            self.__model.fit(x, np.ravel(y))
            return True            
        except Exception as e:
            print(e)
            return False

    def __getPre_i_o(self, O, target):
        prei = []
        for v in O:
            prei.append(v)
            if v == target:
                break
        return prei

    def __calculusContribution(self, target):
        fi_i = 0
        counter_No_Preivalues = 0
        showprogressbar = progressbar.ProgressBar(max_value = self.__m, widgets = ['Contribution of '+target+' (', progressbar.Counter('%(value)d'), '/'+str(self.__m)+')', progressbar.Bar('#', ' [', ']'), ' ', progressbar.Percentage()])
        showprogressbar.start()
        for j in range(1, (self.__m+1)):
            try:
                O = np.random.permutation(self.__N)
                y = self.x_test.sample()
                pre_i_O = self.__getPre_i_o(O, target)
                if pre_i_O[0] == target:
                    counter_No_Preivalues = counter_No_Preivalues + 1
                    showprogressbar.update(j)
                    continue
                # INI - PREPARE DATA WITH SELECTED COLUMNS
                x = self.x_train[self.x_train.columns.intersection(pre_i_O)]
                y = y[y.columns.intersection(pre_i_O)]
                # END
                self.__model.fit(x, self.y_train)
                v1 = self.__model.predict(y)
                pre_i_O.remove(target)
                # INI - PREPARE DATA WITHOUT THE TARGET
                x = self.x_train[self.x_train.columns.intersection(pre_i_O)]
                y = y[y.columns.intersection(pre_i_O)]
                # END
                self.__model.fit(x, self.y_train)
                v2 = self.__model.predict(y)
                fi_i = fi_i + (v1 - v2)
                showprogressbar.update(j)
            except Exception as e:
                logf = open('Error.log', 'a')
                er = {'exception:': e, 'O': O, 'Pre^i(O)': pre_i_O, 'fi_i': fi_i, 'target': target, 'x_shape' : x.shape, 'y_shape': y.shape} 
                logf.write('\n ###################################'+datetime.datetime.now().strftime('%a, %d %b %Y %H:%M:%S')+'################################### \n')
                logf.write(str(er))
                logf.close()
                raise('\n Unexpected error, please check the file called Error.log for further information')
        showprogressbar.finish()
        if (self.__m - counter_No_Preivalues) == 0:
            final_fi_i = 0
        else:
            final_fi_i = fi_i / (self.__m - counter_No_Preivalues)
        self.fi_i_contributions.append({target: self.__extractValue(final_fi_i)})

    def __extractValue(self, val):
        value = val
        if isinstance(value, np.ndarray):
            value = self.__extractValue(value[0])
        return value

    def plot(self):
        vals = []
        i = 0
        max = 0
        for i in range(len(self.fi_i_contributions)):
            value = self.fi_i_contributions[i][self.__features_N[i]]
            if max < abs(value):
                max = abs(value)
            vals.append(value)
        plt.barh(range(len(self.__features_N)), vals)
        plt.yticks(range(len(self.__features_N)), self.__features_N, fontsize = 7)
        plt.xlim(-(max+0.25), (max+0.25))
        plt.title('Contribution of the i-th features')
        plt.show()

    def explainerContribution(self, spec_feat = '*'):
        if self.__checkModelAvailability() == False:
            print('\n This model is not currently available, sorry for the inconvinience')
            return

        if spec_feat == '*':
            spec_feat = self.__N
        
        self.__features_N = spec_feat
        
        for player in spec_feat:
            self.__calculusContribution(player)