# -*- coding: utf-8 -*-
"""
Created on Sat Apr  8 01:26:11 2023

@author: Devashish

"""

"""
RATIONAL FUCNTION APPROXIMATION

"""

import numpy as np
import scipy
import matplotlib.pyplot as plt
from scipy import sparse
from scipy.sparse import diags, linalg
import scipy.sparse.linalg as splin



def RFA_ABC(V,dt,dx,m, k0):
    """
    The fucntions atke the inpur as, 
    
    V - Potential array, 
    dt
    dx, 
    mass which is fixed as 0.5
    and k0, parameter for Gaussian
    
    This computes the required elements for changing ABCs. 
    
    """
    
    import numpy as np
    
    d = 1/(dt*dx)
    RFA_zeta1_0 = ( -1j*(3*(k0)**2)/(2*dx) - 1j*3*k0/(2*dt) + d + (V[0] - k0**3)/4 )
    RFA_zeta1_J = ( -1j*(3*(k0)**2)/(2*dx) - 1j*3*k0/(2*dt) + d + (V[-1] - k0**3)/4 )
    RFA_zeta2_0 = (  1j*(3*(k0)**2)/(2*dx) - 1j*3*k0/(2*dt) - d + (V[0] - k0**3)/4 )
    RFA_zeta2_J = (  1j*(3*(k0)**2)/(2*dx) - 1j*3*k0/(2*dt) - d + (V[-1] - k0**3)/4 )
    RFA_zeta3_0 = (  1j*(3*(k0)**2)/(2*dx) - 1j*3*k0/(2*dt) + d +(V[0] - k0**3)/4 )
    RFA_zeta3_J = (  1j*(3*(k0)**2)/(2*dx) - 1j*3*k0/(2*dt) + d + (V[-1] - k0**3)/4 )
    RFA_zeta4_0 = ( -1j*(3*(k0)**2)/(2*dx) - 1j*3*k0/(2*dt) - d + (V[0] - k0**3)/4 )
    RFA_zeta4_J = ( -1j*(3*(k0)**2)/(2*dx) - 1j*3*k0/(2*dt) - d + (V[-1] - k0**3)/4 )

    RFA_ZETA = np.array([[0,RFA_zeta1_0, RFA_zeta2_0, RFA_zeta3_0,RFA_zeta4_0],
                       [0,RFA_zeta1_J,RFA_zeta2_J,RFA_zeta3_J,RFA_zeta4_J]])
    RFA_ZETA = np.transpose(RFA_ZETA)
    return RFA_ZETA


############################################################


def CN_RFA(J,V,RFA_zeta,dx,dt,m):
    """
    The function takes the input as, 
    J - length of array
    V - Potential array
    RFA_Zeta - ABC which is defined for Rational Functiuon Approximation
    dx - grid step
    dt - time step
    m - mass which is fixed in our case as 0.5
    
    """
    import numpy as np
    
    RFA_one     = np.ones((J),complex)
    RFA_alpha   = (1j)*dt/(4*m*dx**2)               
    RFA_xi    = RFA_one + 1j*dt/2*(1/(m*dx**2)*RFA_one + V)   
    RFA_gamma   = RFA_one - 1j*dt/2*(1/(m*dx**2)*RFA_one + V)   

    RFA_xi[0]  = RFA_zeta[1,0];                                                    RFA_xi[J-1]   = RFA_zeta[1,-1]
    RFA_gamma[0] = RFA_zeta[3,0];                                                  RFA_gamma[J-1] = RFA_zeta[3,-1]
    RFA_up1     = (-RFA_alpha)*np.copy(RFA_one) ;                                  RFA_up2     = RFA_alpha*np.copy(RFA_one); 
    RFA_up1[1]  = RFA_zeta[2,0]           ;                                        RFA_up2[1]  = RFA_zeta[4,0]
    RFA_dn1     = (-RFA_alpha)*np.copy(RFA_one) ;                                  RFA_dn2     = RFA_alpha*np.copy(RFA_one);
    RFA_dn1[J-2]= RFA_zeta[2,-1]           ;                                       RFA_dn2[J-2]= RFA_zeta[4,-1]


    RFA_vecs1 = np.array([RFA_dn1,RFA_xi,RFA_up1]) ;                    RFA_vecs2 = np.array([RFA_dn2,RFA_gamma,RFA_up2])
    diags = np.array([-1,0,+1])


    RFA_U1    = sparse.spdiags(RFA_vecs1,diags,J,J)
    RFA_U1    = RFA_U1.tocsc()
    RFA_U2    = sparse.spdiags(RFA_vecs2,diags,J,J)
    RFA_U2    = RFA_U2.tocsc()

    return RFA_U1, RFA_U2


######################################################

#defining the code for solving the equation with Linear Approximation method

def RFA_CN_Solve(x_Range,dx,t,dt,RFA_psi0,V,m, k0):
    
    
    """
    This fucntion solve the schrodinger wave equation for Linear Approximation method, knowing the fucntions, CM_LAM, 
    and LAM_ZETA
    
    This requires the input, 
    
    """
    
    import numpy as np
    J = len(x_Range)
    N = len(t)
    
    RFA_PSI     = np.zeros((J,N),complex)
    RFA_PSI[:,0] = RFA_psi0
    
    RFA_zeta   = RFA_ABC(V,dt,dx,m, k0)
    RFA_U1, RFA_U2 = CN_RFA(J,V,RFA_zeta,dx,dt,m)
    
    RFA_LU = splin.splu(RFA_U1)   
    
    for n in range(0,N - 1):          
        b = RFA_U2.dot(RFA_PSI[:,n])           
        RFA_PSI[:,n + 1] = RFA_LU.solve(b)
        
    return RFA_PSI


################################################################################################
