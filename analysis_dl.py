# -*- coding: utf-8 -*-
"""
Created on Sat Mar  6 15:19:18 2021

@author: csyu
"""

import os
import numpy as np
import csv
import dictlearn as dl
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

plt.rcParams['font.sans-serif']=['Microsoft JhengHei']
plt.rcParams['axes.unicode_minus']=False

#main
for i in range(0,data_shape_A[1]):
    n_components = 32
    n_nonzero = 32
    data_emg = []
    data_origin = []
    
    for j in range(0,data_shape_A[0]):
        data = np.absolute(all_data_A[j][i])
        data_origin.append(all_data_A[j][i])
        data_fft = np.fft.fft(data)/len(data)
        data_emg.append(data_fft)
    
    for j in range(0,data_shape_B[0]):
        data = np.absolute(all_data_B[j][i])
        data_origin.append(all_data_B[j][i])
        data_fft = np.fft.fft(data)/len(data)
        data_emg.append(data_fft)
    
    data_emg = np.array(data_emg).T
    data_emg = np.absolute(data_emg)
    print(data_emg.shape)
    
    plt.figure()
    #plt.subplot(1, 2, 1)
    plt.title('EMG_' + str(i+1) + '_FFT') # title
    plt.ylabel("電位") # y label
    plt.xlabel("時間") # x label
    plt.plot(np.array(data_origin[0]).T, color='red')
    #plt.savefig('test.png')
    plt.clf()
    plt.cla()
    plt.close()
    
    
    i_fft_A = []
    
    for j in range(0,data_shape_A[0]):
        i_fft_A.append(data_emg.T[j])
        
    i_fft_A_avg = np.mean(np.array(i_fft_A), axis=0)
    i_fft_A_var = np.var(np.array(i_fft_A), axis=0)
    i_fft_A_up = i_fft_A_avg + i_fft_A_var
    i_fft_A_down = i_fft_A_avg - i_fft_A_var
    
    plt.figure()
    #plt.subplot(1, 2, 1)
    plt.title('EMG_' + str(i+1) + '_FFT') # title
    plt.ylabel("EMG Code") # y label
    plt.xlabel("n_components") # x label
    plt.plot(np.array(i_fft_A_avg).T, label = "運動員", color='blue')
    plt.fill_between(range(0,5000), i_fft_A_up.T, i_fft_A_down.T, label = "運動員變異區間", facecolor='blue', alpha=0.3)
    #np.savetxt('EMG_A_' + str(i+1) + '_Code_' + str(n_nonzero) + '_' + str(n_components) + '.csv', np.array(i_code_A), delimiter=",")
    #plt.savefig('EMG_A_' + str(i+1) + '_Code_' + str(n_nonzero) + '_' + str(n_components) + '.png')
    #plt.clf()
    #plt.cla()
    #plt.close()
    
    i_fft_B = []
    
    for j in range(data_shape_A[0],data_shape_A[0]+data_shape_B[0]):
        i_fft_B.append(data_emg.T[j])
        
    i_fft_B_avg = np.mean(np.array(i_fft_B), axis=0)
    i_fft_B_var = np.var(np.array(i_fft_B), axis=0)
    i_fft_B_up = i_fft_B_avg + i_fft_B_var
    i_fft_B_down = i_fft_B_avg - i_fft_B_var
    
    #plt.figure()
    #plt.subplot(1, 2, 2)
    #plt.title('EMG_B_' + str(i+1) + '_Code_' + str(n_nonzero) + '_' + str(n_components)) # title
    #plt.ylabel("EMG Code") # y label
    #plt.xlabel("n_components") # x label
    plt.plot(np.array(i_fft_B_avg).T, label = "一般人", color='green') 
    plt.fill_between(range(0,5000), i_fft_B_up.T, i_fft_B_down.T, label = "一般人變異區間", facecolor='green', alpha=0.3)
    plt.legend()
    #np.savetxt('EMG_B_' + str(i+1) + '_Code_' + str(n_nonzero) + '_' + str(n_components) + '.csv', np.array(i_code_B), delimiter=",")
    #plt.savefig('EMG_B_' + str(i+1) + '_Code_' + str(n_nonzero) + '_' + str(n_components) + '.png')
    #plt.savefig('EMG_' + str(i+1) + '_FFT.png')
    plt.clf()
    plt.cla()
    plt.close()
    
    
    dictionary = dl.random_dictionary(5000, n_components)  # dl.random_dictionary(rows, columns)
    dictionary = dl.optimize.mod(data_emg, dictionary, n_nonzero=n_nonzero, iters=1000, n_threads=10)
    
    print('start learning:' + str(i+1))
    
    sparse_codes = dl.omp_cholesky(data_emg, dictionary, n_nonzero=n_nonzero)
    sparse_approx = dictionary.dot(sparse_codes)
    
    dist = np.linalg.norm(data_emg - sparse_approx)
    print('dist = ' + str(dist))
    
    print('end learning:' + str(i+1))
    
    i_code_A = []
    
    for j in range(0,data_shape_A[0]):
        i_code_A.append(sparse_codes.T[j])
        
    i_code_A_avg = np.mean(np.array(i_code_A), axis=0)
    #i_code_A_var = np.sqrt(np.var(np.array(i_code_A), axis=0))
    #i_code_A_up = i_code_A_avg + i_code_A_var
    #i_code_A_down = i_code_A_avg - i_code_A_var
    i_code_A_up = np.ndarray.max(np.array(i_code_A), axis=0)
    i_code_A_down = np.ndarray.min(np.array(i_code_A), axis=0)
    
    plt.figure()
    #plt.subplot(1, 2, 1)
    plt.title('EMG_' + str(i+1) + '_Code_' + str(n_nonzero) + '_' + str(n_components)) # title
    plt.ylabel("EMG Code") # y label
    plt.xlabel("n_components") # x label
    plt.plot(np.array(i_code_A_avg).T, label = "運動員", color='blue')
    plt.fill_between(range(0,32), i_code_A_up.T, i_code_A_down.T, label = "運動員變異區間", facecolor='blue', alpha=0.3)
    #np.savetxt('EMG_A_' + str(i+1) + '_Code_' + str(n_nonzero) + '_' + str(n_components) + '.csv', np.array(i_code_A), delimiter=",")
    #plt.savefig('EMG_A_' + str(i+1) + '_Code_' + str(n_nonzero) + '_' + str(n_components) + '.png')
    #plt.clf()
    #plt.cla()
    #plt.close()
    
    i_code_B = []
    
    for j in range(data_shape_A[0],data_shape_A[0]+data_shape_B[0]):
        i_code_B.append(sparse_codes.T[j])
        
    i_code_B_avg = np.mean(np.array(i_code_B), axis=0)
    #i_code_B_var = np.sqrt(np.var(np.array(i_code_B), axis=0))
    #i_code_B_up = i_code_B_avg + i_code_B_var
    #i_code_B_down = i_code_B_avg - i_code_B_var
    i_code_B_up = np.ndarray.max(np.array(i_code_B), axis=0)
    i_code_B_down = np.ndarray.min(np.array(i_code_B), axis=0)
    
    #plt.figure()
    #plt.subplot(1, 2, 2)
    #plt.title('EMG_B_' + str(i+1) + '_Code_' + str(n_nonzero) + '_' + str(n_components)) # title
    #plt.ylabel("EMG Code") # y label
    #plt.xlabel("n_components") # x label
    plt.plot(np.array(i_code_B_avg).T, label = "一般人", color='green') 
    plt.fill_between(range(0,32), i_code_B_up.T, i_code_B_down.T, label = "一般人變異區間", facecolor='green', alpha=0.3)
    plt.legend()
    #np.savetxt('EMG_B_' + str(i+1) + '_Code_' + str(n_nonzero) + '_' + str(n_components) + '.csv', np.array(i_code_B), delimiter=",")
    #plt.savefig('EMG_B_' + str(i+1) + '_Code_' + str(n_nonzero) + '_' + str(n_components) + '.png')
    plt.savefig('EMG_' + str(i+1) + '_Code_' + str(n_nonzero) + '_' + str(n_components) + '.png')
    plt.clf()
    plt.cla()
    plt.close()
    
    #break