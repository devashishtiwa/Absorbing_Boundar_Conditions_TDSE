# -*- coding: utf-8 -*-
"""
Created on Sat Apr  8 01:31:48 2023

@author: Devashish
"""

import numpy as np
import scipy
import matplotlib.pyplot as plt
from scipy import sparse
from scipy.sparse import diags, linalg
import scipy.sparse.linalg as splin


def P4_ABC(q1,q2,q3,q4,V,dt,dx,m):       
    
    """
    ABC with p=4, 
    This code also computes the matrix for zeta, for p = 4. 
    This gives the first and last row elements for new matrix for Crank Nikolson method
    """
    
    g1 = m*(q1 + q2 + q3 + q4)
    g2 = m**2*(q1*q2 + q1*q3 + q1*q4 + q2*q3 + q2*q4 +q3*q4)
    g3 = m**3*q1*q2*q3*q4*(1/q1 + 1/q2 + 1/q3 + 1/q4)
    g4 = m**4*q1*q2*q3*q4
    p1 = -4*m**2
    p2 = 2*m*g1
    p3_0 = (1j*2*m*g2 - 1j*8*(m**2)*V[0]) 
    p3_J = (1j*2*m*g2 - 1j*8*(m**2)*V[-1]) 
    p4_0 = (1j*2*m*g1*V[0] - 1j*g3)
    p4_J = (1j*2*m*g1*V[-1] - 1j*g3)
    p5_0 = (4*(m**2)*(V[0])**2 - 2*m*g2*V[0] + g4)
    p5_J = (4*(m**2)*(V[-1])**2 - 2*m*g2*V[-1] + g4)
    
    
    p4_zeta1_0 = (  p1/(2*dt*dt) - p2/(dx*dt) + p3_0/(2*dt) - p4_0/(2*dx) + p5_0/4 )
    p4_zeta1_J = (  p1/(2*dt*dt) - p2/(dx*dt) + p3_J/(2*dt) - p4_J/(2*dx) + p5_J/4 )
    p4_zeta2_0 = (  p1/(2*dt*dt) + p2/(dx*dt) + p3_0/(2*dt) + p4_0/(2*dx) + p5_0/4 )
    p4_zeta2_J = (  p1/(2*dt*dt) + p2/(dx*dt) + p3_J/(2*dt) + p4_J/(2*dx) + p5_J/4 )
    p4_zeta3_0 = (  p1/(dt*dt) - p2/(dx*dt) + p3_0/(2*dt) + p4_0/(2*dx) - p5_0/4 )
    p4_zeta3_J = (  p1/(dt*dt) - p2/(dx*dt) + p3_J/(2*dt) + p4_J/(2*dx) - p5_J/4 )
    p4_zeta4_0 = (  p1/(dt*dt) + p2/(dx*dt) + p3_0/(2*dt) - p4_0/(2*dx) - p5_0/4 )
    p4_zeta4_J = (  p1/(dt*dt) + p2/(dx*dt) + p3_J/(2*dt) - p4_J/(2*dx) - p5_J/4 )
    p4_zeta5   = ( -p1/(2*dt*dt) )  
    p4_zeta6   = ( -p1/(2*dt*dt) )
    
    P4_ZETA = np.array([[0,p4_zeta1_0,p4_zeta2_0,p4_zeta3_0,p4_zeta4_0,p4_zeta5,p4_zeta6],
                      [0,p4_zeta1_J,p4_zeta2_J,p4_zeta3_J,p4_zeta4_J,0,0]])
    P4_ZETA = np.transpose(P4_ZETA)
    return P4_ZETA


################################################

def Z_matrix(J,p4_zeta):      
    
    """
    The function computes the Z Matrix which is required to obtain the necessary solutions. 
    The derivation of this matrix is shown in our report.
    
    The input required for this function is just J (len of grid points), 
    and zeta matrix for p = 3
    
    """
    
    row  = np.array([0,0,J-1,J-1])
    col  = np.array([0,1,J-2,J-1])
    z    = np.array([p4_zeta[5,0],p4_zeta[6,0],p4_zeta[6,0],p4_zeta[5,0]])
    Z    = sparse.csr_matrix((z,(row,col)), shape=(J,J))
    Z    = Z.tocsc()
    return Z



################################################

def CN_P4(J,V,p4_zeta,dx,dt,m):
    
    
    """
    The function takes the input as, 
    J - length of array
    V - Potential array
    4_Zeta - ABC which is defined for P = 4 Absorbing boundary Conditions
    dx - grid step
    dt - time step
    m - mass which is fixed in our case as 0.5
    
    This also requires the initial matrixces of crank nikolson mathod, the reason for them is well shown in 
    our report. 
    
    """
    
    one     = np.ones((J),complex)
    alpha   = (1j)*dt/(4*m*dx**2)                
    xi    = one + 1j*dt/2*(1/(m*dx**2)*one + V)    
    gamma   = one - 1j*dt/2*(1/(m*dx**2)*one + V)   
    xi[0]  = p4_zeta[1,0]; xi[J-1]   = p4_zeta[1,-1]
    gamma[0] = p4_zeta[3,0]; gamma[J-1] = p4_zeta[3,-1]
    up1     = (-alpha)*np.copy(one) ; up2     = alpha*np.copy(one); 
    up1[1]  = p4_zeta[2,0]           ; up2[1]  = p4_zeta[4,0]
    dn1     = (-alpha)*np.copy(one) ; dn2     = alpha*np.copy(one);
    dn1[J-2]= p4_zeta[2,-1]           ; dn2[J-2]= p4_zeta[4,-1]
    vecs1 = np.array([dn1,xi,up1]) ; vecs2 = np.array([dn2,gamma,up2])
    diags = np.array([-1,0,+1])
    U1    = sparse.spdiags(vecs1,diags,J,J)
    U1    = U1.tocsc()
    U2    = sparse.spdiags(vecs2,diags,J,J)
    U2    = U2.tocsc()
    
    def CrankNikolson(J,V,dx,dt,m):
        
        """
        The function defines the variables that we derived in our report, 
        and then uses the scipy library to solve n-linear equations,
        using sparse.
        
        The function will take the input as len(x), Potential(in array),
        dx, dt, and mass. It will return the Matrix U1 and U2 (which is obtained in CN Method)
        
        J = len(x)
        V = Array of Potential
        dx = amount by which x is increases(distance between two successive grid points)
        dt = distance between two given time
        m = mass , which is generally set as 0.5 ion our overall calculations
        
        """
        
        one     = np.ones((J),complex)
        alpha   = (1j)*dt/(m*4*dx**2)               
        gamma  = one - 1j*dt/2*(1/(m*dx**2)*one + V)   
        xi    = one + 1j*dt/2*(1/(m*dx**2)*one + V)   
        off   = alpha*one
        
        
        diags = np.array([-1,0,+1])      
        vecs1 = np.array([-off,xi,-off])                   
        U1    = sparse.spdiags(vecs1,diags,J,J)
        U1    = U1.tocsc()                      
        vecs2 = np.array([off,gamma,off])
        U2    = sparse.spdiags(vecs2,diags,J,J)
        U2    = U2.tocsc() 
        return U1, U2
    
    W1,W2 = CrankNikolson(J,V,dx,dt,m)
        
    return U1, U2, W1, W2



############################################

def p4_Solve_CN(x_Range,dx,t,dt,psi0,V, m, p4_zeta, Z):
    
    """
    This fucntion solves the Crank Niksolson metod matrix, and obtains the direct result of wave fucntion, 
    provided we properly defines the initial wave fucntion.
    
    Input taken, are x_Range,
    grid step, time step
    and initial wave fucntion, at t = 0
    Potential array
    m- which is fixed in our case as 0.5
    p4_zeta is required for the fucntioning of this functions, otherwise,
    it is giving error for defining itwithin a fucntion
    
    Z is the Z matrix that is newly obtained in the code only for p =4
    """
    J = len(x_Range)
    N = len(t)

    p4_PSI      = np.zeros((J,N),complex)
    p4_PSI[:,0] = psi0
    
    P4_U1, P4_U2, W1, W2 = CN_P4(J,V,p4_zeta, dx,dt,m)
    P4_LU  = splin.splu(P4_U1)
    LW     = splin.splu(W1)
      
    for n in range(0,N-1):
        if n == 0:
            b = W2.dot(p4_PSI[:,n])  
            p4_PSI[:,n+1] = LW.solve(b)
            
        else:
            b = P4_U2.dot(p4_PSI[:,n]) + Z.dot(p4_PSI[:,n-1])
            p4_PSI[:,n+1] = P4_LU.solve(b)
        
    return p4_PSI




############################################################################################