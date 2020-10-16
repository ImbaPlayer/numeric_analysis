# -*- coding: utf-8 -*-
"""
Created on Fri Oct  9 09:58:12 2020

@author: Administrator
"""

import scipy.linalg as li
import numpy as np
import itertools as it
import math

def G(A, b, xTure):
    I = np.identity(n)
    L = np.zeros((n,n))
    for i in range(n):
        L[i, np.arange(0, i)] = - A[i, np.arange(0, i)]
    D = np.zeros((n,n))
    for i in range(n):
        D[i, i] = A[i, i]
    B = I - np.dot(li.inv(D), A)
    w, v = li.eig(B)
    w_max = np.max(np.abs(w))
    print("J_B", w_max)
    f = np.inner(li.inv(D), b)
    x0 = np.zeros(n)
        
    # for i in range(500):
    #     x = np.inner(B, x0) + f
    #     x0 = x
    q = li.norm(B, ord=2)
    times = 0
    # while(True):
    #     x = np.inner(B, x0) + f
        
    #     if np.max(np.abs(x - x0)) < 0.001:
    #         break
    #     x0 = x
    #     times += 1
    # print("x", x)
    # print("times", times)
    print("cal times", getK(B, 6))


def GS(A, b):
    I = np.identity(n)
    L = np.zeros((n,n))
    for i in range(n):
        L[i, np.arange(0, i)] = - A[i, np.arange(0, i)]
    D = np.zeros((n,n))
    for i in range(n):
        D[i, i] = A[i, i]
    B = I - np.dot(li.inv(D - L), A)
    f = np.inner(li.inv(D - L), b)
    
    x0 = np.zeros(n)
        
    # for i in range(500):
    #     x = np.inner(B, x0) + f
    #     x0 = x
    w, v = li.eig(B)
    w_max = np.max(np.abs(w))
    print("GS_B", w_max)
    times = 0
    # while(True):
    #     x = np.inner(B, x0) + f
        
    #     if np.max(np.abs(x - x0)) < 0.001:
    #         break
    #     x0 = x
    #     times += 1
    # print("x", x)
    # print("times", times)
    print("cal times", getK(B, 6))

def getK(B, s):
    w, v = li.eig(B)
    w_max = np.max(np.abs(w))
    k = s * math.log(10) / - math.log(w_max)
    return k

def Hilbert(n):
    A = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            A[i,j] = np.float64(1.0 / (i + j + 1))
    return A

# 谱半径
def calP(B):
    w, v = li.eig(B)
    w_max = np.max(np.abs(w))
    return w_max

# 最大特征值和最小特征值之比
def testH(B):
    w, v = li.eig(B)
    w_max = np.max(np.abs(w))
    w_min = np.min(np.abs(w))
    return w_max / w_min
if __name__ == "__main__":
    # for n in range(2, 9):
    #     print(n)
    #     # A = np.float64(np.random.randint(0,10,(n,n)))
    #     A = Hilbert(n)
    #     # idx0 = np.arange(0, n)
    #     # A[idx0, idx0] += 20
    #     detTure = np.float64(li.det(A))
    #     xTure = np.random.randn(n)
    #     b = np.inner(A,xTure)
        
    #     G(A, b, xTure)
    #     GS(A,b)
        # print("xTrue", xTure)
    for n in range(2, 9):
        A = Hilbert(n)
        print(calP(A))
        print(testH(A))
