import scipy.linalg as li
import numpy as np
import itertools as it
import math


def getK(B, s):
    W, v = li.eig(B)
    w_max = np.max(np.abs(W))
    k = s * math.log(10) / - math.log(w_max)
    return k

def SOR(A, b, w):
    
    I = np.identity(n)
    L = np.zeros((n,n))
    U = np.zeros((n,n))
    for i in range(n):
        L[i, np.arange(0, i)] = - A[i, np.arange(0, i)]
        U[i ,np.arange(i + 1, n)] = - A[i ,np.arange(i + 1, n)]
    D = np.zeros((n,n))
    # print(L)
    # print(U)
    for i in range(n):
        D[i, i] = A[i, i]


    B1 = li.inv(D - w*L)
    B2 = w*U + (1-w)*D
    B = np.dot(B1, B2)
    f = w*np.inner(B1, b)
    
    x0 = np.array([0.2 for i in range(n)])
        
    # for i in range(500):
    #     x = np.inner(B, x0) + f
    #     x0 = x
    w, v = li.eig(B)
    w_max = np.max(np.abs(w))
    print("SOR_B", w_max)
    times = 0
    while(True):
        x = np.inner(B, x0) + f
        
        if np.max(np.abs(x - x0)) < 0.000001:
            break
        x0 = x
        times += 1
    print("x", x)
    print("times", times)
    print("cal times", getK(B, 6))

def Hilbert(n):
    A = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            A[i,j] = np.float64(1.0 / (i + j + 1))
    return A

def test():
    H = Hilbert(2)
    print(2*H)

if __name__ == "__main__":
    n = 8
    H = Hilbert(n)
    xTure = np.array([1 for i in range(n)])
    b = np.inner(H,xTure)
    w = 1.6
    # while(w < 2):
    #     print(w)
    #     SOR(H, b, w)
    #     w += 0.1
    SOR(H, b, w)