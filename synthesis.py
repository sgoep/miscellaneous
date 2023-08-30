'''
This is a python function for solving the optimization problem
    min_x 1/2*||AT*c - y||_2^2 + alpha*||c||_1,      (1)
using the Chambolle-Pock algorithm. Here
    A     - is the forward operator,
    y     - is the available (noisy) data,
    T*    - is an synthesis operator for example for the wavelet transform.
    alpha - is the regularization parameter.
The function returns a sequence of coefficients c, such that 
    x = T*c.

The formula given in (1) is also called the synthesis form of the sparse ell1 regularization approach.

Note that the code may not run for generic A and T and depends on their implementation.
'''

import numpy as np

def soft(c, al):
    cnew = [np.sign(c[0])*np.maximum(0, np.abs(c[0]) - al)]
    for i in range(1, len(c)):
        ctmp = list(c[i])
        for j in range(3):
            ctmp[j] = np.sign(ctmp[j])*np.maximum(0, np.abs(ctmp[j]) - al)    
        cnew.append(ctmp)
    return cnew

def syn(x0, A, AT, T, TT, g, alpha, L, Niter, f):
    tau = 1/L
    sigma = 1/L
    theta = 1
    
    p    = np.zeros_like(g)
    q    = np.zeros_like(g)
    
    u    = x0
    ubar = x0
    
    er = np.zeros(Niter)
    for k in range(Niter):
        p = (p + sigma*(A(TT(ubar)[:-1,:-1]) - g))/(1 + sigma)
        
        c = T(AT(p + q))
            
        v = [ u[0] - tau * c[0] ]
        for i in range(1, len(c)):
            ctmp = c[i]
            utmp = u[i]
            vtmp = []
            for j in range(len(ctmp)):
                vtmp.append(utmp[j] - tau * ctmp[j])
            v.append(tuple(vtmp))
   
        uiter = soft(v, alpha)
                   
        ubar = [uiter[0] + theta * (uiter[0] - u[0])]
        for i in range(1, len(u)):
            uitertmp = uiter[i]
            utmp     = u[i]
            ubartmp  = []
            for j in range(len(utmp)):
                ubartmp.append(uitertmp[j] + theta * (uitertmp[j] - utmp[j]) )
            ubar.append(tuple(ubartmp))
        
        u = uiter
        
        frec = TT(ubar)[:-1,:-1]
        er[k] = np.sum(np.abs(frec - f)**2)/np.sum(np.abs(f)**2)
        print('Synthesis Iteration: ' + str(k+1) + '/' + str(Niter) + ', Error:' + str(er[k]))
        
    return ubar

