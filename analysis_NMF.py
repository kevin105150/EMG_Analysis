# -*- coding: utf-8 -*-
"""
Created on Sat Mar  6 15:19:18 2021

@author: csyu
"""

import os
import numpy as np
import csv
from sklearn.decomposition import NMF
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import tkinter as tk
from tkinter import filedialog

root = tk.Tk()
root.withdraw()

#group A
file_path_A = filedialog.askdirectory()
print(file_path_A)
file_list_A = os.listdir(file_path_A)
print(file_list_A)

all_data_A = []

for filename in file_list_A:
    filename = file_path_A + '/' + filename
    with open(filename, newline='') as csvfile:
      X = csv.reader(csvfile)
      X = list(X)
      
    X = np.array(X).astype(float).T
    all_data_A.append(X)

all_data_A = np.array(all_data_A)

data_shape_A = all_data_A.shape
print('group A ' + str(data_shape_A))

all_data_A = np.sqrt(np.square(all_data_A))

#group B
file_path_B = filedialog.askdirectory()
print(file_path_B)
file_list_B = os.listdir(file_path_B)
print(file_list_B)

all_data_B = []

for filename in file_list_B:
    filename = file_path_B + '/' + filename
    with open(filename, newline='') as csvfile:
      X = csv.reader(csvfile)
      X = list(X)
      
    X = np.array(X).astype(float).T
    all_data_B.append(X)

all_data_B = np.array(all_data_B)

data_shape_B = all_data_B.shape
print('group B ' + str(data_shape_B))

all_data_B = np.sqrt(np.square(all_data_B))

RMS_data = []

plt.rcParams['font.sans-serif']=['Microsoft JhengHei']
plt.rcParams['axes.unicode_minus']=False

#main
for i in range(0,data_shape_A[1]):
    data_emg = []
    data_origin = []
    for j in range(0,data_shape_A[0]):
        data_origin.append(all_data_A[j][i])
        data_fft = np.fft.fft(all_data_A[j][i])/len(all_data_A[j][i])
        data_emg.append(data_fft)
        #data_emg.append(all_data_A[j][i])
    
    for j in range(0,data_shape_B[0]):
        data_origin.append(all_data_B[j][i])
        data_fft = np.fft.fft(all_data_B[j][i])/len(all_data_B[j][i])
        data_emg.append(data_fft)
        #data_emg.append(all_data_B[j][i])
    
    data_emg = np.absolute(data_emg)
    data_emg = np.array(data_emg).T
    print(data_emg.shape)
    
    nmf = NMF(n_components=None,  # k value,???????????????????????????
          init=None,  # W H ???????????????????????????'random' | 'nndsvd'(??????) |  'nndsvda' | 'nndsvdar' | 'custom'.
          solver='cd',  # 'cd' | 'mu'
          beta_loss='frobenius',  # {'frobenius', 'kullback-leibler', 'itakura-saito'}?????????????????????
          tol=1e-4,  # ???????????????????????????
          max_iter=10000,  # ??????????????????
          random_state=None,
          alpha=0.,  # ???????????????
          l1_ratio=0.,  # ???????????????
          verbose=0,  # ????????????
          shuffle=False  # ??????"cd solver"
          )
    
    nmf.fit(data_emg)
    #X_transformed = dict_learner.fit_transform(data_emg)
    print('start learning:' + str(i+1))
    W = nmf.fit_transform(data_emg)
    #W = nmf.transform(data_emg)
    #nmf.inverse_transform(W)
    # -----------------??????------------------------
    sparse_codes = nmf.components_  # H??????
    print('reconstruction_err_', nmf.reconstruction_err_)  # ???????????????
    print('n_iter_', nmf.n_iter_)  # ??????????????????
    print('end learning:' + str(i+1))
    
    #plt.figure()
    #plt.title('Learn_Error_' + str(i+1)) # title
    #plt.plot(X_error)
    #plt.savefig('Learn_Error_' + str(i+1) + '.png')
    #plt.show()
    plt.clf()
    plt.cla()
    plt.close()
    
    i_code_A = []
    
    for j in range(0,data_shape_A[0]):
        i_code_A.append(sparse_codes.T[j])
        
    i_code_A_avg = np.mean(np.array(i_code_A), axis=0)
    i_code_A_up = np.ndarray.max(np.array(i_code_A), axis=0)
    i_code_A_down = np.ndarray.min(np.array(i_code_A), axis=0)
    
    i_code_A_up_RMS = np.sqrt(np.mean(np.square(i_code_A_up)))
    i_code_A_down_RMS = np.sqrt(np.mean(np.square(i_code_A_down)))
    
    plt.figure()
    #plt.subplot(1, 2, 1)
    plt.title('EMG_' + str(i+1) + '_NMF') # title
    plt.ylabel("EMG Code") # y label
    plt.xlabel("n_components") # x label
    plt.plot(np.array(i_code_A_avg).T, label = "?????????", color='blue')
    plt.fill_between(range(0,32), i_code_A_up.T, i_code_A_down.T, label = "?????????????????????", facecolor='blue', alpha=0.3)
    #np.savetxt('EMG_A_' + str(i+1) + '_Code_' + str(n_nonzero) + '_' + str(n_components) + '.csv', np.array(i_code_A), delimiter=",")
    #plt.savefig('EMG_A_' + str(i+1) + '_Code_' + str(n_nonzero) + '_' + str(n_components) + '.png')
    #plt.clf()
    #plt.cla()
    #plt.close()
    
    i_code_B = []
    
    for j in range(data_shape_A[0],data_shape_A[0]+data_shape_B[0]):
        i_code_B.append(sparse_codes.T[j])
        
    i_code_B_avg = np.mean(np.array(i_code_B), axis=0)
    i_code_B_up = np.ndarray.max(np.array(i_code_B), axis=0)
    i_code_B_down = np.ndarray.min(np.array(i_code_B), axis=0)
    
    i_code_B_up_RMS = np.sqrt(np.mean(np.square(i_code_B_up)))
    i_code_B_down_RMS = np.sqrt(np.mean(np.square(i_code_B_down)))
    
    #plt.figure()
    #plt.subplot(1, 2, 2)
    #plt.title('EMG_B_' + str(i+1) + '_Code_' + str(n_nonzero) + '_' + str(n_components)) # title
    #plt.ylabel("EMG Code") # y label
    #plt.xlabel("n_components") # x label
    plt.plot(np.array(i_code_B_avg).T, label = "?????????", color='green') 
    plt.fill_between(range(0,32), i_code_B_up.T, i_code_B_down.T, label = "?????????????????????", facecolor='green', alpha=0.3)
    plt.legend()
    #np.savetxt('EMG_B_' + str(i+1) + '_Code_' + str(n_nonzero) + '_' + str(n_components) + '.csv', np.array(i_code_B), delimiter=",")
    #plt.savefig('EMG_B_' + str(i+1) + '_Code_' + str(n_nonzero) + '_' + str(n_components) + '.png')
    plt.savefig('EMG_' + str(i+1) + '_NMF.png')
    plt.clf()
    plt.cla()
    plt.close()
    
    RMS_title = ['?????????_MAX', '?????????_MAX', '?????????_MIN', '?????????_MIN']
    RMS_val = [i_code_A_up_RMS, i_code_B_up_RMS, i_code_A_down_RMS, i_code_B_down_RMS]
    RMS_data.append(RMS_val)
    x = np.arange(len(RMS_title))
    plt.bar(x, RMS_val, color=['red', 'green', 'blue', 'yellow'])
    plt.xticks(x, RMS_title)
    plt.xlabel('Title')
    plt.ylabel('RMS')
    plt.title('2 Norm_Value_'+ str(i+1))
    plt.savefig('RMS_' + str(i+1) + '_NMF.png')
    plt.clf()
    plt.cla()
    plt.close()
    
    #break
    
np.savetxt('RMS_data.csv', np.array(RMS_data), delimiter=",")