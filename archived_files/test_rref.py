import sympy
import numpy as np
import pickle
import numpy as np
import scipy as sp
from SLSFinite import *
from Polytope import *
from functions_archived import *
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as patches
import math
import time

np.random.seed(1)

A = np.random.uniform(-1,1,[42,42])
U, S, Vh = np.linalg.svd(A)
n=13
k=10
U_float64 = sympy.Matrix((10**k*np.round(U[:,0:n],k)).astype('int64'))
S_float64 = sympy.Matrix((10**k*np.round(np.diag(S[0:n]),k)).astype('int64'))
Vh_float64 = sympy.Matrix((10**k*np.round(Vh[0:n,:],k)).astype('int64'))
#U_float64 = (10**k*np.round(U[:,0:n],k)).astype('int')
#S_float64 = (10**k*np.round(np.diag(S[0:n]),k)).astype('int')
#Vh_float64 = (10**k*np.round(Vh[0:n,:],k)).astype('int')

row_reduced_data = (U_float64*S_float64*Vh_float64)[0:7,:].rref()
#row_reduced_data = sympy.Matrix(U_float64.dot(S_float64).dot(Vh_float64)).rref()
A_trunc = U[:,0:n].dot(np.diag(S[0:n])).dot(Vh[0:n,:])
A_trunc_int = np.array(10**(-3*k)*U_float64*S_float64*Vh_float64)

print('Error truncation rref:', np.max(np.abs(A_trunc - A_trunc_int)))
print('rank rref', len(row_reduced_data[1]))
print()

#########################################################################################################3
