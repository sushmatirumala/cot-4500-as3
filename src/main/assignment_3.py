import numpy as np


# Question 1

def function(t: float, w: float):
    return t - w**2

def eulers():
    original_w = 1
    start_of_t, end_of_t = (0, 2)
    num_of_iterations = 10
    h = (end_of_t - start_of_t) / num_of_iterations

    for cur_iteration in range(0, num_of_iterations):
        t = start_of_t
        w = original_w
        h = h
        next_w = w + h * function(t,w)
        start_of_t = t + h
        original_w = next_w
        
    return next_w

print("%.5f" % eulers())
print()


# Question 2

def runge_kutta():
    original_w = 1
    start_of_t, end_of_t = (0, 2)
    num_of_iterations = 10
    h = (end_of_t - start_of_t) / num_of_iterations

    for cur_iteration in range(0, num_of_iterations):
        t = start_of_t
        w = original_w
        h = h
        start_of_t = t + h
        k1 = h * function(t,w)
        k2 = h * function(t+(h/2), w + (1/2) * k1)
        k3 = h * function(t + (h/2), w + (1/2) * k2)
        k4 = h * function(t + h, w + k3)
        next_w = w + (1/6)*(k1 + 2*k2 + 2*k3 + k4)
        original_w = next_w
        
    return next_w

print("%.5f" % runge_kutta(), "\n")


# Question 3

def gaussian_elimination(A, b):
    n = len(b)
    Ab = np.concatenate((A, b.reshape(n,1)), axis=1)

    for i in range(n):
        max_row = i
        for j in range(i+1, n):
            if abs(Ab[j,i]) > abs(Ab[max_row,i]):
                max_row = j
        Ab[[i,max_row], :] = Ab[[max_row,i], :]
        pivot = Ab[i,i]
        Ab[i,:] = Ab[i,:] / pivot

        for j in range(i+1, n):
            factor = Ab[j,i]
            Ab[j,:] -= factor * Ab[i,:]

    for i in range(n-1, -1, -1):
        for j in range(i-1, -1, -1):
            factor = Ab[j,i]
            Ab[j,:] -= factor * Ab[i,:]

    x = Ab[:,n]
    return x

A = np.array([[2,-1,1],[1,3,1],[-1,5,4]], dtype=float)
b = np.array([6,0,-3])
print(gaussian_elimination(A, b))
print()


# Question 4

def LU_factorization(A):
    n = len(A)
    U = np.copy(A)
    L = np.identity(n)

    for i in range(0, n):
        for j in range(i+1, n):
            factor = U[j][i]/U[i][i]
            U[j,:] -= factor * U[i,:]
            L[j][i] = factor
            
    print(L)
    print()
    print(U)
    print()
    return None

A = np.array([[1,1,0,3],[2,1,-1,1],[3,-1,-1,2], [-1,2,3,-1]], dtype =np.double)
determinant = np.linalg.det(A)
print("%.5f" % determinant)
print()
LU_factorization(A)


# Question 5

def diagonally_dominant(A):
    n = len(A)
    for i in range(0, n) :        
        sum = 0
        for j in range(0, n) :
            sum = sum + abs(A[i][j])  

        sum -= abs(A[i][i])
        if (abs(A[i][i]) < sum) :
            return False
    return True

A = np.array([[9, 0, 5, 2, 1], [3, 9, 1, 2, 1], [0, 1, 7, 2, 3], [4, 2, 3, 12, 2], [3, 2, 4, 0, 8]])
print(diagonally_dominant(A))
print()


# Question 6

import numpy as np

def is_positive_definite(A):
    if np.all(np.linalg.eigvals(A) > 0):
        return True
    else:
        return False
        
A = np.array([[2,2,1], [2,3,0], [1,0,2]])
print(is_positive_definite(A))
print()
