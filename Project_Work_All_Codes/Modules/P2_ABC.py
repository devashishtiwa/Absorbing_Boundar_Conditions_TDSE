# -*- coding: utf-8 -*-
"""
Created on Sat Apr  8 01:29:26 2023

@author: Devashish
"""

import numpy as np
import scipy
import matplotlib.pyplot as plt
from scipy import sparse
from scipy.sparse import diags, linalg
import scipy.sparse.linalg as splin


def P2_ABC(q1,q2,V,dt,dx,m = 0.5):
    
    """
    ABC with p=2. This function computes the matrix zeta, for substituting in Crank Nikolson method. 
    """
    c1 = 2./(q1 + q2)
    c2 = (m*q1*q2)/(q1 + q2)
    zeta1_0 = ( -1j/(2*dx) - 1j*c1/(2*dt) + (c1*V[0] - c2)/4 )
    zeta1_J = ( -1j/(2*dx) - 1j*c1/(2*dt) + (c1*V[-1] - c2)/4 )
    zeta2_0 = (  1j/(2*dx) - 1j*c1/(2*dt) + (c1*V[0] - c2)/4 )
    zeta2_J = (  1j/(2*dx) - 1j*c1/(2*dt) + (c1*V[-1] - c2)/4 )
    zeta3_0 = (  1j/(2*dx) - 1j*c1/(2*dt) - (c1*V[0] - c2)/4 )
    zeta3_J = (  1j/(2*dx) - 1j*c1/(2*dt) - (c1*V[-1] - c2)/4 )
    zeta4_0 = ( -1j/(2*dx) - 1j*c1/(2*dt) - (c1*V[0] - c2)/4 )
    zeta4_J = ( -1j/(2*dx) - 1j*c1/(2*dt) - (c1*V[-1] - c2)/4 )
    P2zeta = np.array([[0,zeta1_0,zeta2_0,zeta3_0,zeta4_0],
                       [0,zeta1_J,zeta2_J,zeta3_J,zeta4_J]])
    P2zeta = np.transpose(P2zeta)
    return P2zeta


#########################################################

def CN_P2(J,V,zeta,dx,dt,m):
    
    
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
    xi[0]  = zeta[1,0]; xi[J-1]   = zeta[1,-1]
    gamma[0] = zeta[3,0]; gamma[J-1] = zeta[3,-1]
    up1     = (-alpha)*np.copy(one) ; up2     = alpha*np.copy(one); 
    up1[1]  = zeta[2,0]           ; up2[1]  = zeta[4,0]
    dn1     = (-alpha)*np.copy(one) ; dn2     = alpha*np.copy(one);
    dn1[J-2]= zeta[2,-1]           ; dn2[J-2]= zeta[4,-1]
    vecs1 = np.array([dn1,xi,up1]) ; vecs2 = np.array([dn2,gamma,up2])
    diags = np.array([-1,0,+1])
    U1    = sparse.spdiags(vecs1,diags,J,J)
    U1    = U1.tocsc()
    U2    = sparse.spdiags(vecs2,diags,J,J)
    U2    = U2.tocsc()
    
    return U1, U2


######################################################

def p2_Solve_CN(x_Range,dx,t,dt,psi0,V, m, p2_zeta):
    
    """
    This fucntion solves the Crank Niksolson metod matrix, and obtains the direct result of wave fucntion, 
    provided we properly defines the initial wave fucntion.
    
    Input taken, are x_Range,
    grid step, time step
    and initial wave fucntion, at t = 0
    Potential array
    m- which is fixed in our case as 0.5
    p2_zeta is required, which is amtrix obtained from P2Zeta functions
    
    """
    J = len(x_Range)
    N = len(t)

    P2_PSI      = np.zeros((J,N),complex)
    P2_PSI[:,0] = psi0
    
    P2_U1, P2_U2 = CN_P2(J,V,p2_zeta, dx,dt,m)
    P2_LU  = splin.splu(P2_U1)
    
    for n in range(0,N - 1):          
        b = P2_U2.dot(P2_PSI[:,n])          
        P2_PSI[:,n + 1] = P2_LU.solve(b)
        
    return P2_PSI


#####################################################################################################
