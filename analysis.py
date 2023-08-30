'''
This is a python function for solving the optimization problem
    min_x 1/2*||Ax - y||_2^2 + alpha*||Tx||_1,      (1)
using the Chambolle-Pock algorithm. Here
    A     - is the forward operator,
    y     - is the available (noisy) data,
    T     - is an analysis operator for example for the wavelet transform.
    alpha - is the regularization parameter.
The formula given in (1) is also called the analysis form of the sparse ell1 regularization approach.

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

def analysis(x0, A, AT, T, TT, g, alpha, L, Niter, f):
    tau = 1/L
    sigma = 1/L
    theta = 1
    
    p    = np.zeros_like(g)
    q    = T(np.zeros_like(x0))
    
    u    = x0
    ubar = x0
    
    er = np.zeros(Niter)
    for k in range(Niter):
        p = (p + sigma*(A(ubar) - g))/(1 + sigma)
        
        Tu = T(u)
        qtmp = [alpha*(q[0] + sigma*Tu[0])/np.maximum(alpha, np.abs(q[0] + sigma*Tu[0]))]
        for i in range(1, len(Tu)):
            qq = list(Tu[i])
            for j in range(3):
                qq[j] = alpha*(q[i][j] + sigma*Tu[i][j])/np.maximum(alpha, np.abs(q[i][j] + sigma*Tu[i][j]))
            qtmp.append(qq)
        q = qtmp    
        
        uiter = np.maximum(0, u - tau*(AT(p) + TT(q)))
        ubar = uiter + theta*(uiter - u)        
        u = uiter
        
        frec = ubar
        er[k] = np.sum(np.abs(frec - f)**2)/np.sum(np.abs(f)**2)
        # print('Analysis Iteration: ' + str(k+1) + '/' + str(Niter) + ', Error:' + str(er[k]))
        
    return ubar
