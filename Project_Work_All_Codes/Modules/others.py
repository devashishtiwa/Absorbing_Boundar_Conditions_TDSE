# -*- coding: utf-8 -*-
"""
Created on Sat Apr  8 02:29:09 2023

@author: Devashish
"""

import numpy as np

def Gaussian_Function(x_Range,centre,k0,sigma):         
    S = 1/(sigma**2*np.pi)**(1./4.)
    A = k0*x_Range
    B = (x_Range - centre)**2/(2*sigma**2)
    exp = np.exp(1j*A - B)
    return S*exp

