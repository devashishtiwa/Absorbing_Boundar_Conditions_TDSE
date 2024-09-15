# -*- coding: utf-8 -*-
"""
Created on Sat Apr  8 01:30:57 2023

@author: Devashish
"""

import numpy as np
import scipy
import matplotlib.pyplot as plt
from scipy import sparse
from scipy.sparse import diags, linalg
import scipy.sparse.linalg as splin



def P3_ZETA(q1,q2,q3,V,dt,dx,m):       
    """
    Here, we take input of q1, q2, and q3, which are variables in terms of k0,
    potential array, 
    dt, 
    dx
    and mass which is fixed as 0.5. 
    
    
    The code computes the zeta matrix, for the ABC for p = 3. This is derived in our report. 
    """
    
    h1 = m*(q1 + q2 + q3)
    h2 = m**2*q1*q2*q3*(1/q1 + 1/q2 + 1/q3)
    h3 = m**3*q1*q2*q3
    a_0  = 1j*(h2/(2*m) - V[0])
    a_J  = 1j*(h2/(2*m) - V[-1])
    b  = 1
    c  = 1j*h1
    d_0  = (h3/(2*m) - h1*V[0])
    d_J  = (h3/(2*m) - h1*V[-1])
    
    P3_zeta1_0 = ( -a_0/(2*dx) + b/(dt*dx) - c/(2*dt) - d_0/4 )
    P3_zeta1_J = ( -a_J/(2*dx) + b/(dt*dx) - c/(2*dt) - d_J/4 )
    P3_zeta2_0 = (  a_0/(2*dx) - b/(dt*dx) - c/(2*dt) - d_0/4 )
    P3_zeta2_J = (  a_J/(2*dx) - b/(dt*dx) - c/(2*dt) - d_J/4 )
    P3_zeta3_0 = (  a_0/(2*dx) + b/(dt*dx) - c/(2*dt) + d_0/4 )
    P3_zeta3_J = (  a_J/(2*dx) + b/(dt*dx) - c/(2*dt) + d_J/4 )
    P3_zeta4_0 = ( -a_0/(2*dx) - b/(dt*dx) - c/(2*dt) + d_0/4 )
    P3_zeta4_J = ( -a_J/(2*dx) - b/(dt*dx) - c/(2*dt) + d_J/4 )
    P3_ZETA = np.array([[0, P3_zeta1_0, P3_zeta2_0, P3_zeta3_0, P3_zeta4_0],
                       [0, P3_zeta1_J, P3_zeta2_J,P3_zeta3_J, P3_zeta4_J]])
    
    P3_ZETA = np.transpose(P3_ZETA)
    return P3_ZETA



##################################################

def CN_P3(J,V,p3_zeta,dx,dt,m):
    
    
    """
    The function takes the input as, 
    J - length of array
    V - Potential array
    P2_Zeta - ABC which is defined for P = 2 Absorbing boundary Conditions
    dx - grid step
    dt - time step
    m - mass which is fixed in our case as 0.5
    
    """
    
    one     = np.ones((J),complex)
    alpha   = (1j)*dt/(4*m*dx**2)                
    xi    = one + 1j*dt/2*(1/(m*dx**2)*one + V)    
    gamma   = one - 1j*dt/2*(1/(m*dx**2)*one + V)   
    xi[0]  = p3_zeta[1,0]; xi[J-1]   = p3_zeta[1,-1]
    gamma[0] = p3_zeta[3,0]; gamma[J-1] = p3_zeta[3,-1]
    up1     = (-alpha)*np.copy(one) ; up2     = alpha*np.copy(one); 
    up1[1]  = p3_zeta[2,0]           ; up2[1]  = p3_zeta[4,0]
    dn1     = (-alpha)*np.copy(one) ; dn2     = alpha*np.copy(one);
    dn1[J-2]= p3_zeta[2,-1]           ; dn2[J-2]= p3_zeta[4,-1]
    vecs1 = np.array([dn1,xi,up1]) ; vecs2 = np.array([dn2,gamma,up2])
    diags = np.array([-1,0,+1])
    U1    = sparse.spdiags(vecs1,diags,J,J)
    U1    = U1.tocsc()
    U2    = sparse.spdiags(vecs2,diags,J,J)
    U2    = U2.tocsc()
    
    return U1, U2


################################################


def p3_Solve_CN(x_Range,dx,t,dt,psi0,V, m, p3_zeta):
    
    """
    This fucntion solves the Crank Niksolson metod matrix, and obtains the direct result of wave fucntion, 
    provided we properly defines the initial wave fucntion.
    
    Input taken, are x_Range,
    grid step, time step
    and initial wave fucntion, at t = 0
    Potential array
    m- which is fixed in our case as 0.5
    p3_zeta is required, the matrix which is obtained by P3ZETA fucntion
    
    """
    
    J = len(x_Range)
    N = len(t)

    P3_PSI      = np.zeros((J,N),complex)
    P3_PSI[:,0] = psi0
    
    P3_U1, P3_U2 = CN_P3(J,V,p3_zeta, dx,dt,m)
    P3_LU  = splin.splu(P3_U1)
    
    for n in range(0,N - 1):          
        b = P3_U2.dot(P3_PSI[:,n])          
        P3_PSI[:,n + 1] = P3_LU.solve(b)
        
    return P3_PSI



#############################################################################################