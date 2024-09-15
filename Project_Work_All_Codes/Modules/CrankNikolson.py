# -*- coding: utf-8 -*-
"""
Created on Sat Apr  8 01:28:39 2023

@author: Devashish
"""

import numpy as np
import scipy
import matplotlib.pyplot as plt
from scipy import sparse
from scipy.sparse import diags, linalg
import scipy.sparse.linalg as splin

   
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
    

####################################################

def Solve_CN(x_Range,dx,t,dt,psi0,V, m):
    
    """
    This fucntion solves the Crank Niksolson metod matrix, and obtains the direct result of wave fucntion, 
    provided we properly defines the initial wave fucntion.
    
    Input taken, are x_Range,
    grid step, time step
    and initial wave fucntion, at t = 0
    Potential array
    m- which is fixed in our case as 0.5
    
    """
    J = len(x_Range)
    N = len(t)

    PSI      = np.zeros((J,N),complex)
    PSI[:,0] = psi0
    
    U1, U2 = CrankNikolson(J,V,dx,dt,m)
    LU  = splin.splu(U1)
    
    for n in range(0,N - 1):          
        b = U2.dot(PSI[:,n])          
        PSI[:,n + 1] = LU.solve(b)
        
    return PSI

################################################################################################

#defining a guassian wave packet, for keeping this as the initial wave function

def Gaussian_Function(x_Range,centre,k0,sigma):         
    S = 1/(sigma**2*np.pi)**(1./4.)
    A = k0*x_Range
    B = (x_Range - centre)**2/(2*sigma**2)
    exp = np.exp(1j*A - B)
    return S*exp


##############################################################################################