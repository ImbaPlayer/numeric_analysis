import scipy.linalg as li
import numpy as np
import itertools as it

def Hilbert(n):
    A = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            A[i,j] = np.float64(1.0 / (i + j + 1))
    return A

# 索引对角元素
def indexA(A):
    idx0 = np.arange(0, n)
    a = A[idx0, idx0]
    return a

def J(A):
    D = np.diag(A)
    print(D)
    B = np.identity(n) - np.dot(D, A)
    print(B)

def GS(A):
    D = np.diag(A)
    L = np.zeros((n, n))
    for i in range(n):
        for j in range(0, i + 1):
            L[i, j] = - A[i, j]
    print(L)

if __name__ == "__main__":
    n = 3
    H = Hilbert(n)
    print(H)
    # J(H)
    GS(H)