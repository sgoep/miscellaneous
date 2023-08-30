'''
This is a python function for solving the optimization problem
    min_x 1/2*||Ax - y||_2^2 + alpha*||Tx||_1,      (1)
using the FISTA algorithm. Here
    A     - is the forward operator,
    y     - is the available (noisy) data,
    T     - is an analysis operator for example for the wavelet transform.
    alpha - is the regularization parameter.
The formula given in (1) is also called the analysis form of the sparse ell1 regularization approach.

Note that the code may not run for generic A and T and depends on their implementation.
'''

import numpy as np

def S(x, alpha):
    return np.sign(x)*np.maximum(0, np.abs(x)-alpha)

def soft(y, Fgrad, la, L):
    alpha = la/L
    cnew = [(y[0] - Fgrad[0])/L]
    for i in range(len(y)-1):
        tmp = (S((y[i+1][0] - Fgrad[i+1][0])/L,alpha), S((y[i+1][1] - Fgrad[i+1][1])/L,alpha), S((y[i+1][2] - Fgrad[i+1][2])/L, alpha))
        cnew.append(tmp)
    return cnew

def fista(x0, A, AT, L, G, la, niter):
    xout   = x0
    y      = x0
    t_step = 1
    
    for i in range(niter):
        print(i,'/',niter)
        tprev = t_step
        xprev = xout
        
        Giter = A(y)
        Fgrad = AT(Giter - G)
        if la > 0.0:
            xout = S(y-Fgrad/L, la/L)
        else:
            xout = y - Fgrad/L
        t_step   = (1+np.sqrt(1+4*tprev**2))/2
        y = xout + (tprev-1)/t_step*(xout - xprev)
        
    return xout
