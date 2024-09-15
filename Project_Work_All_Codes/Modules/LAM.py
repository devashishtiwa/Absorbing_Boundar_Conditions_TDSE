# -*- coding: utf-8 -*-
"""
Created on Sat Apr  8 01:27:04 2023

@author: Devashish

"""



"""
LINEAR APPROXIMATION METHOD
"""

import numpy as np
import scipy
import matplotlib.pyplot as plt
from scipy import sparse
from scipy.sparse import diags, linalg
import scipy.sparse.linalg as splin


def LAM_ABC(g1, g2, V,dt,dx,m, k0):
    
    """
    The function will provide the necessary matrix, for solving Schodinger Wave Equation using
    Crank Nikolson Scheme, with Linear Approximation method of Absorbing Boundary Conditions
    
    
    The functions take the input as, 
    
    Potential Array, whose length should be equal to number of gird points
    g1 and g2 are parameters derived in our report. 

    
    dt - Time step
    dx - grid step
    m - mass, which is fixed in our case as 0.5
    k0 - Parameter of guassian, changing this, changes the shape of guassian, to much extent
    """
    
    import numpy as np

    c1 = 1
    c2 = g2/g1        #again, named c1 as ratio of g2/g1

    LAM_zeta1_0 = ( -1j/(2*dx*g1) - 1j*c1/(2*dt) + (c1*V[0] - c2)/4 )        #each term was derived in our report
    LAM_zeta1_J = ( -1j/(2*dx*g1) - 1j*c1/(2*dt) + (c1*V[-1] - c2)/4 )
    LAM_zeta2_0 = (  1j/(2*dx*g1) - 1j*c1/(2*dt) + (c1*V[0] - c2)/4 )
    LAM_zeta2_J = (  1j/(2*dx*g1) - 1j*c1/(2*dt) + (c1*V[-1] - c2)/4 )
    LAM_zeta3_0 = (  1j/(2*dx*g1) - 1j*c1/(2*dt) - (c1*V[0] - c2)/4 )
    LAM_zeta3_J = (  1j/(2*dx*g1) - 1j*c1/(2*dt) - (c1*V[-1] - c2)/4 )
    LAM_zeta4_0 = ( -1j/(2*dx*g1) - 1j*c1/(2*dt) - (c1*V[0] - c2)/4 )
    LAM_zeta4_J = ( -1j/(2*dx*g1) - 1j*c1/(2*dt) - (c1*V[-1] - c2)/4 )
    LAM_ZETA = np.array([[0,LAM_zeta1_0,LAM_zeta2_0,LAM_zeta3_0,LAM_zeta4_0],      #matrix was made, so that we can use each coefficients
                       [0,LAM_zeta1_J,LAM_zeta2_J,LAM_zeta3_J,LAM_zeta4_J]])
    LAM_ZETA = np.transpose(LAM_ZETA)
    
    return LAM_ZETA


################################################

def CN_LAM(J,V,LAM_zeta,dx,dt,m):
    
    
    """
    The function takes the input as, 
    J - length of array
    V - Potential array
    LAM_zeta - ABC which is defined for Rational Functiuon Approximation
    dx - grid step
    dt - time step
    m - mass which is fixed in our case as 0.5
    

    """
    
    import numpy as np

    LAM_one     = np.ones((J),complex)                      #General Variables, must need to define for CN Method
    LAM_alpha   = (1j)*dt/(4*m*dx**2)               
    LAM_xi    = LAM_one + 1j*dt/2*(1/(m*dx**2)*LAM_one + V)   
    LAM_gamma   = LAM_one - 1j*dt/2*(1/(m*dx**2)*LAM_one + V)  
    LAM_xi[0]  = LAM_zeta[1,0];                                        LAM_xi[J-1]   = LAM_zeta[1,-1]   
    LAM_gamma[0] = LAM_zeta[3,0];                                      LAM_gamma[J-1] = LAM_zeta[3,-1]
    LAM_up1     = (-LAM_alpha)*np.copy(LAM_one) ;                      LAM_up2 = LAM_alpha*np.copy(LAM_one); 
    LAM_up1[1]  = LAM_zeta[2,0]           ;                            LAM_up2[1]  = LAM_zeta[4,0]
    LAM_dn1     = (-LAM_alpha)*np.copy(LAM_one) ;                      LAM_dn2     = LAM_alpha*np.copy(LAM_one);
    LAM_dn1[J-2]= LAM_zeta[2,-1]           ;                           LAM_dn2[J-2]= LAM_zeta[4,-1]
    LAM_vecs1 = np.array([LAM_dn1,LAM_xi,LAM_up1]) ;                   LAM_vecs2 = np.array([LAM_dn2,LAM_gamma,LAM_up2])


    diags = np.array([-1,0,+1])
    LAM_U1    = sparse.spdiags(LAM_vecs1,diags,J,J)
    LAM_U1    = LAM_U1.tocsc()
    LAM_U2    = sparse.spdiags(LAM_vecs2,diags,J,J)
    LAM_U2    = LAM_U2.tocsc()
    
    return LAM_U1, LAM_U2


##############################################


#defining the code for solving the equation with Linear Approximation method

def LAM_CN_Solve(g1, g2, x_Range,dx,t,dt,LAM_psi0,V, m, k0):
    """
    This fucntion solve the schrodinger wave equation for Linear Approximation method, knowing the fucntions, CM_LAM, 
    and LAM_ZETA
    
    This requires the input,
    x_Range - Range of grid points
    dx, dt, time points, 
    m which is fixed
    LAM_psi0, which is initial wave packet
    
    
    """
    
    def LAM_ABC(g1, g2, V,dt,dx,m, k0):
        
        """
        The function will provide the necessary matrix, for solving Schodinger Wave Equation using
        Crank Nikolson Scheme, with Linear Approximation method of Absorbing Boundary Conditions
        
        
        The functions take the input as, 
        
        Potential Array, whose length should be equal to number of gird points
        g1 and g2 are parameters derived in our report. 

        
        dt - Time step
        dx - grid step
        m - mass, which is fixed in our case as 0.5
        k0 - Parameter of guassian, changing this, changes the shape of guassian, to much extent
        """
        
        import numpy as np

        c1 = 1
        c2 = g2/g1        #again, named c1 as ratio of g2/g1

        LAM_zeta1_0 = ( -1j/(2*dx*g1) - 1j*c1/(2*dt) + (c1*V[0] - c2)/4 )        #each term was derived in our report
        LAM_zeta1_J = ( -1j/(2*dx*g1) - 1j*c1/(2*dt) + (c1*V[-1] - c2)/4 )
        LAM_zeta2_0 = (  1j/(2*dx*g1) - 1j*c1/(2*dt) + (c1*V[0] - c2)/4 )
        LAM_zeta2_J = (  1j/(2*dx*g1) - 1j*c1/(2*dt) + (c1*V[-1] - c2)/4 )
        LAM_zeta3_0 = (  1j/(2*dx*g1) - 1j*c1/(2*dt) - (c1*V[0] - c2)/4 )
        LAM_zeta3_J = (  1j/(2*dx*g1) - 1j*c1/(2*dt) - (c1*V[-1] - c2)/4 )
        LAM_zeta4_0 = ( -1j/(2*dx*g1) - 1j*c1/(2*dt) - (c1*V[0] - c2)/4 )
        LAM_zeta4_J = ( -1j/(2*dx*g1) - 1j*c1/(2*dt) - (c1*V[-1] - c2)/4 )
        LAM_ZETA = np.array([[0,LAM_zeta1_0,LAM_zeta2_0,LAM_zeta3_0,LAM_zeta4_0],      #matrix was made, so that we can use each coefficients
                           [0,LAM_zeta1_J,LAM_zeta2_J,LAM_zeta3_J,LAM_zeta4_J]])
        LAM_ZETA = np.transpose(LAM_ZETA)
        
        return LAM_ZETA
    
    import numpy as np

    J = len(x_Range)
    N = len(t)
    
    LAM_PSI      = np.zeros((J,N),complex)
    LAM_PSI[:,0] = LAM_psi0
    
    LAM_zeta   = LAM_ABC(g1, g2, V,dt,dx,m, k0)
    LAM_U1, LAM_U2 = CN_LAM(J,V,LAM_zeta,dx,dt,m)
    
    LAM_LU = splin.splu(LAM_U1)   
    
    for n in range(0,N - 1):          
        b = LAM_U2.dot(LAM_PSI[:,n])           
        LAM_PSI[:,n + 1] = LAM_LU.solve(b)
        
    return LAM_PSI


#############################################################################################