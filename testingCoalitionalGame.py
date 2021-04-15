"""
Usage: Testing Coalitional Game py
Python version: 3.9.X
Author: rauls19
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import random
from sklearn.linear_model import LinearRegression
import xgboost as xgb
import os
import CoalitionalGame as cg
from sklearn.feature_selection import RFE
from sklearn.neural_network import MLPRegressor

data_info_original = pd.read_csv('info_data.csv')
data_info_original = data_info_original.drop(columns = ['DAY_OF_WEEK'])
data_info_original = data_info_original.sort_values(['ANO_FACTURA', 'MES_FACTURA', 'FECHA_FACTURA'])
data_info = pd.read_csv('info_datav2.csv')
data_info = data_info.drop(columns = ['DAY_OF_WEEK'])
data_info = data_info.sort_values(['ANO_FACTURA', 'MES_FACTURA', 'FECHA_FACTURA'])

def groupData(df, col_groups, ref_col):
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

columns = ['ANO_FACTURA', 'MES_FACTURA', 'FECHA_FACTURA', 'TEMPORADA_COMERCIAL_ID', 'PRODUCTO_ID', 'TALLA', 'ESFUERZO_VENTA_ID', 'NUMERO_DEUDOR_PAIS_ID', 'JERARQUIA_PROD_ID', 'GRUPO_ARTICULO_PRODUCTO_ID', 'GENERO_PRODUCTO', 'CATEGORIA', 'TIPOLOGIA', 'CONSUMER_COLOR', 'CREMALLERA', 'CORDONES', 'OUTSOLE_SUELA_TIPO', 'OUTSOLE_SUELA_SUBTIPO', 'PLANTILLA_EXTRAIBLE', 'CONTACTO_SN', 'EDAD_SN', 'GENERO_CONTACTO', 'EDAD_COMPRA', 'EDAD_RANGO_COMPRA', 'CIUDAD_CONTACTO', 'IDIOMA_CONTACTO']

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

columns = ['ANO_FACTURA', 'MES_FACTURA', 'FECHA_FACTURA', 'TEMPORADA_COMERCIAL_ID', 'PRODUCTO_ID', 'TALLA', 'ESFUERZO_VENTA_ID', 'NUMERO_DEUDOR_PAIS_ID', 'JERARQUIA_PROD_ID', 'GRUPO_ARTICULO_PRODUCTO_ID', 'GENERO_PRODUCTO', 'CATEGORIA', 'TIPOLOGIA', 'CONSUMER_COLOR', 'CREMALLERA', 'CORDONES', 'OUTSOLE_SUELA_TIPO', 'OUTSOLE_SUELA_SUBTIPO', 'PLANTILLA_EXTRAIBLE', 'CONTACTO_SN', 'EDAD_SN', 'GENERO_CONTACTO', 'EDAD_COMPRA', 'CIUDAD_CONTACTO', 'IDIOMA_CONTACTO']

# INI - TESTING LINEAR MODEL
# lm_model = LinearRegression()
# coalgame_lm = cg.CoalitionalGame(lm_model, columns, x_train, y_train, x_test, y_test, m = 10)
# coalgame_lm.explainerContribution('*')
# print(coalgame_lm.fi_i_contributions)
# coalgame_lm.plot()
# END

# INI - TESTING XGBOOST
# def removeFeatures(dt_train, dt_test, rfe_model):
#     chosen = pd.DataFrame(rfe_model.support_, index=dt_train.columns, columns=['Rank'])
#     featuresnotselected = []
#     for k, v in zip(chosen.index, chosen.values):
#         if v == False:
#             featuresnotselected.append(k)
#     dt_train_final = dt_train.drop(columns= featuresnotselected)
#     dt_test_final = dt_test.drop(columns= featuresnotselected)
#     return dt_train_final, dt_test_final

# n_features_optimal = 21 # model_cv.best_params_['n_features_to_select']
# xgbm = xgb.XGBRegressor(learning_rate =0.01, n_estimators=215, max_depth=10, min_child_weight=0.8, subsample=1, nthread=4)
# rfe = RFE(xgbm, n_features_to_select= n_features_optimal)             
# rfe = rfe.fit(x_train, y_train)
# x_train_XGBM, x_test_XGBM = removeFeatures(x_train, x_test, rfe)
# xgbm = xgb.XGBRegressor(learning_rate =0.01, n_estimators=215, max_depth=10, min_child_weight=0.8, subsample=1, nthread=4)
# coalgame_xgb = cg.CoalitionalGame(xgbm, x_train_XGBM.columns, x_train_XGBM, y_train, x_test_XGBM, y_test, m = 5)
# coalgame_xgb.explainerContribution('*')
# print(coalgame_xgb.fi_i_contributions)
# coalgame_xgb.plot()
# END

# INI - TESTING MLP
nn_reg = MLPRegressor(hidden_layer_sizes=(60, 20),  activation='logistic', solver='adam', alpha=0.01, batch_size='auto', learning_rate='constant', learning_rate_init=0.01, max_iter=1000, shuffle=False, tol=0.0001, verbose=False, early_stopping= True, validation_fraction=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
coalgame_nn_reg = cg.CoalitionalGame(nn_reg, x_train.columns, x_train, np.ravel(y_train), x_test, np.ravel(y_test), m = 5)
coalgame_nn_reg.explainerContribution('*')
print(coalgame_nn_reg.fi_i_contributions)
coalgame_nn_reg.plot()
# END

# INI - TESTING LSTM

# END