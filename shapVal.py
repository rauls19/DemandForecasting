"""
Usage: shapley values
Python version: 3.9.X
Author: rauls19
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import pickle
import itertools
import threading

""" def groupData(df, col_groups, ref_col):
    testing =  df.groupby(col_groups)[ref_col].sum()
    grouped_testing = []
    for k,v in zip(testing.index, testing.values):
        val = [k[0],k[1],k[2],v,k[3], k[4],k[5],k[6],k[7],k[8],k[9],k[10],k[11],k[12],k[13],k[14],k[15],k[16],k[17],k[18],k[19],k[20],k[21],k[22],k[23],k[24], k[25]]
        grouped_testing.append(val)
    grouped_testing = pd.DataFrame(grouped_testing, columns = df.columns)
    #grouped_testing = grouped_testing.reset_index(drop=True)
    return grouped_testing

def getISOCountry(iso_country):
    indexes = data_info_original[data_info_original['NUMERO_DEUDOR_PAIS_ID'] == iso_country].index
    iso_code = data_info.iloc[[indexes[0]]]['NUMERO_DEUDOR_PAIS_ID']
    return iso_code.values[0]

def getEncode(dto, dt, val, col):
    indexes = dto[-dto[col] == val].index
    code = dt.iloc[[indexes[0]]][col]
    return code.values[0]

def getValuesFilter(ds, value, columns, target):
    if value == '*': #no filter 
        return ds
    indexes = getIndexFilter(ds, value, target)
    datat = getValues(ds, indexes, columns)
    return datat

def getIndexFilter(dt, value, target):
    indexes = dt[dt[target] == value].index
    return indexes

def getValues(dt, indexes, columns):
    datat = []
    for k, v in zip(dt.index, dt.values):
        if k in indexes:
            datat.append(v)
    df = pd.DataFrame(datat, columns = columns)
    return df

def managementDataFile(dat, name_file, r_w):      
    # Its important to use binary mode
    if r_w == 'r':
        dbfile = open(name_file, 'ab')
        pickle.dump(dat, dbfile)
    elif r_w == 'w':
        dbfile = open(name_file, 'rb')
        db = pickle.load(dbfile)
    dbfile.close()

columns = ['ANO_FACTURA', 'MES_FACTURA', 'FECHA_FACTURA', 'TEMPORADA_COMERCIAL_ID', 'PRODUCTO_ID', 'TALLA', 'ESFUERZO_VENTA_ID', 'NUMERO_DEUDOR_PAIS_ID', 'JERARQUIA_PROD_ID', 'GRUPO_ARTICULO_PRODUCTO_ID', 'GENERO_PRODUCTO', 'CATEGORIA', 'TIPOLOGIA', 'CONSUMER_COLOR', 'CREMALLERA', 'CORDONES', 'OUTSOLE_SUELA_TIPO', 'OUTSOLE_SUELA_SUBTIPO', 'PLANTILLA_EXTRAIBLE', 'CONTACTO_SN', 'EDAD_SN', 'GENERO_CONTACTO', 'EDAD_COMPRA', 'EDAD_RANGO_COMPRA', 'CIUDAD_CONTACTO', 'IDIOMA_CONTACTO']

data_info_original = pd.read_csv('info_data.csv')
data_info_original = data_info_original.drop(columns = ['DAY_OF_WEEK'])
data_info_original = data_info_original.sort_values(['ANO_FACTURA', 'MES_FACTURA', 'FECHA_FACTURA'])
data_info = pd.read_csv('info_datav2.csv')
data_info = data_info.drop(columns = ['DAY_OF_WEEK'])
data_info = data_info.sort_values(['ANO_FACTURA', 'MES_FACTURA', 'FECHA_FACTURA'])

print(data_info.shape)
print(data_info.columns)

data_info_grouped = groupData(data_info, columns, 'IMP_VENTA_NETO_EUR')
data_info_grouped = data_info_grouped.sort_values(['ANO_FACTURA', 'MES_FACTURA', 'FECHA_FACTURA'])
data_info_original_grouped = groupData(data_info_original, columns, 'IMP_VENTA_NETO_EUR')
data_info_original_grouped = data_info_original_grouped.sort_values(['ANO_FACTURA', 'MES_FACTURA', 'FECHA_FACTURA'])

data_info_filtered_grouped = getValuesFilter(data_info_grouped, '*', data_info_grouped.columns, 'NUMERO_DEUDOR_PAIS_ID') # iso = * -> no filter by country
data_info_filtered_grouped = data_info_filtered_grouped.sort_values(['ANO_FACTURA', 'MES_FACTURA', 'FECHA_FACTURA'])
data_info_original_filtered_grouped = getValuesFilter(data_info_original_grouped, '*', data_info_original_grouped.columns, 'NUMERO_DEUDOR_PAIS_ID') # iso = * -> no filter by country
data_info_original_filtered_grouped = data_info_original_filtered_grouped.sort_values(['ANO_FACTURA', 'MES_FACTURA', 'FECHA_FACTURA'])

traindataset_original, testdataset_original = train_test_split(data_info_original_filtered_grouped, test_size=0.4, shuffle= False) # To use all the data, change to -> data_info
traindataset, testdataset = train_test_split(data_info_filtered_grouped, test_size=0.4, shuffle= False) # To use all the data, change to -> data_info
x_train = traindataset.loc[:, traindataset.columns != 'IMP_VENTA_NETO_EUR']
y_train = traindataset.loc[:, traindataset.columns == 'IMP_VENTA_NETO_EUR']
x_train = x_train.drop(columns=['EDAD_RANGO_COMPRA'])
x_test = testdataset.loc[:, testdataset.columns != 'IMP_VENTA_NETO_EUR']
y_test = testdataset.loc[:, testdataset.columns == 'IMP_VENTA_NETO_EUR']
x_test = x_test.drop(columns = 'EDAD_RANGO_COMPRA')

normalizer = MinMaxScaler(feature_range = (-1, 1))
x_train = pd.DataFrame(normalizer.fit_transform(x_train), columns= x_train.columns, index = traindataset.index)
x_test = pd.DataFrame(normalizer.fit_transform(x_test), columns= x_test.columns, index = testdataset.index)
 """
import queue
import random
from sklearn.linear_model import LinearRegression
import xgboost as xgb

q = queue.Queue()

def perms(N):
    pi_N = list(itertools.permutations(N))
    q.put(pi_N)
    return pi_N

#print(np.random.permutation(columns))
fi_i = 0 # contribution of i-th features's value
m = 3 # number of samples

#print(list(itertools.permutations([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])))
#t1 = threading.Thread(target= perms, args=(N))
#t1.start()
#pi_N = q.get()
#print(pi_N[0])
# TEST DATA
data = pd.DataFrame([[1, 2, 50, 5], [4, 5, 60, 7], [7, 8, 60, 5]], columns = ['A', 'B', 'C', 'D'])
y_data = pd.DataFrame([57, 70, 85])
t_data = pd.DataFrame([[5, 2, 100, 4], [3, 4, 80, 7], [7, 9, 30, 0]], columns = ['A', 'B', 'C', 'D'])
ty_data = pd.DataFrame([108, 95, 65])
# FI TEST DATA
N = ['A', 'B', 'C', 'D'] # Features (players)
pi_N = list(itertools.permutations(N)) # the set of all ordered permutations of N
# Pre^i(O) the set of players which are predecessors of player i  in the order O
i = 'C' # observing feature
columns = ['A', 'B', 'C', 'D']
aux = 0
model = LinearRegression()
for j in range(1, m):
    print('j = ', j)
    O = random.sample(pi_N, 1)
    print('Permutation: ', O, '\n')
    y = data.sample()
    pre_i_O = []
    for v in O[0]:
        pre_i_O.append(v)
        if v == i:
            break
    if pre_i_O[0] == i:
        print('No')
        aux = aux + 1
        continue
    x = data[data.columns.intersection(pre_i_O)]
    y = y[y.columns.intersection(pre_i_O)]
    print('x data v1: ', x)
    model.fit(x, y_data)
    v1 = model.predict(y)
    v1 = v1[0]
    print('\n v1 = ', v1)
    pre_i_O.remove(i)
    x = data[data.columns.intersection(pre_i_O)]
    y = y[y.columns.intersection(pre_i_O)]
    print('x data v2: ', x)
    model.fit(x, y_data)
    v2 = model.predict(y)
    v2 = v2[0]
    print('\n v2 = ',v2)
    fi_i = fi_i + (v1 - v2)
    print('\n fi_i = ',fi_i)

print(list(pre_i_O))
fi_i = fi_i/(m - 1 - aux)

print('fi_i_total = ', fi_i)