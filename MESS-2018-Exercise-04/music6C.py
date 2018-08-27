import numpy as np
import itertools

from hilbert import *
from polvect6C import *

class structtype():
    pass

def music6C(data,test_param,wtype,v_scal,samp,W,l,stab,l_auto_perc):
    ''' 
    Six-component MUSIC algorithm after Sollberger et al. (2018)
    
    data: Nx6 Matrix containing 6-C data (N samples) ordered as 
          [v_x,v_y,v_z,omega_x,omega_y,omega_z]. Acceleration and
          rotation rate can be used instead of particle velocity and
          rotation angle

    test_param: STRUCTURE ARRAY CONTAINING PARAMETER SPACE TO BE SEARCHED
                ONLY THE PARAMETERS OF THE SPECIFIED WAVE TYPE ARE REQUIRED
                test_param.vp   : vector containing P-wave velocities (ms)
                test_param.vs   : vector containnig S-wave velocities
                test_param.vR   : vector containing Rayleigh wave velocities
                test_param.vL   : vector containing Love wave velocities
                test_param.theta: vector containing incidence angles (degree)
                test_param.phi  : vector containing azimuth angles (degree)
                test_param.xi   : vector containing ellipticity angles (radians)

    wtype: WAVE TYPE
           'P' : P-wave
           'SV': SV-wave
           'SH': SH-wave
           'L' : Love-wave
           'R' : Rayleigh-wave

    v_scal: SCALING VELOCITY (m/s)

    samp: SAMPLE AT WHICH THE WINDOW IS CENTERED

    W: WINDOW LENGTH (IN SAMPLES)

    l: DETERMINES THE DIMENSION OF THE NULL-SPACE OF THE
       COHERENCY MATRIX (see Eq.45 in Sollberger et al., 2018)
       l=4: isolated wave, one dominant eigenvalue
       l=3: two waves, two dominant eigenvalues
       l='auto': automatically determine the size of the null space from 
                 the eigenvalue range, determines the number of eigenvalues 
                 that are smaller than l_auto_perc*lambda_max

    stab: OPTIONAL STABILISATION PARAMETER TO AVOID DIVISION BY 0.
          DEFAULT VALUE IS stab=1e-9;

    l_auto_perc: OPTIONAL PARAMETER FOR THE AUTOMATIC DETERMINATION OF THE
                 DIMENSION OF THE NULL-SPACE. THE DIMENSION OF THE NULL
                 SPACE IS ESTIMATED BY DETERMINING THE NUMBER OF EIGENVALUES
                 THAT ARE SMALLER THAN l_auto_perc*lambda_max. DEFAULT VALUE
                 IS l_auto_perc=0.01 (one percent)
    ''' 
    param = structtype()

    # Calculate null space of coherency matrix
    if np.remainder(W,2):
        W = W + 1 # make window length even

    data = hilbert(data, axis=0) # convert to the analytic signal

    C = np.matrix.getH(data[int(samp-W/2):int(samp+W/2),:]) @ data[int(samp-W/2):int(samp+W/2),:] # compute coherency matrix
    C = C / W # average over window length

    Cprime,Q = np.linalg.eigh(C,UPLO='U')     # eigenvalue decomposition (Q: eigenvectors, Cprime: eigenvalues)
    lambda_  = np.sort(Cprime)[::-1] # sort eigenvalues in descending order
    loc =   np.argsort(Cprime)[::-1]
    Q   =   Q[:,loc]                 # sort eigenvectors

    # determination of the size of the null space
    I = np.nonzero(lambda_[1:]/lambda_[0] < l_auto_perc)
    I = (list(itertools.chain.from_iterable(I)))[0] + 1
    if l == 'auto':
        l = (5 - I) - 1
    Q = Q[:,5-l:5] @ np.matrix.getH(Q[:,5-l:5]) # null space

    ## P-wave
    if wtype == 'P':
        L = np.zeros((len(test_param.theta),len(test_param.phi),len(test_param.vp),len(test_param.vs)))
        for it1 in range(0,len(test_param.theta)):
            for it2 in range(0,len(test_param.phi)):
                for it3 in range(0,len(test_param.vp)):
                    for it4 in range(0,len(test_param.vs)):
                        param.theta = test_param.theta[it1]
                        param.phi   = test_param.phi[it2]
                        param.vp    = test_param.vp[it3]
                        param.vs    = test_param.vs[it4]
                        v = polvect6C(param,v_scal,'P')                           # calculate test polarization vector
                        v = v / np.linalg.norm(v)                                 # convert to unit vector
                        L[it1,it2,it3,it4] = 1/(np.matrix.getH(v) @ Q @ v + stab) # MUSIC estimator
        return L
    
    ## SV-wave
    if wtype == 'SV':
        L = np.zeros((len(test_param.theta),len(test_param.phi),len(test_param.vp),len(test_param.vs)))
        for it1 in range(0,len(test_param.theta)):
            for it2 in range(0,len(test_param.phi)):
                for it3 in range(0,len(test_param.vp)):
                    for it4 in range(0,len(test_param.vs)):
                        param.theta = test_param.theta[it1]
                        param.phi   = test_param.phi[it2]
                        param.vp    = test_param.vp[it3]
                        param.vs    = test_param.vs[it4]
                        v = polvect6C(param,v_scal,'SV')                          # calculate test polarization vector
                        v = v / np.linalg.norm(v)                                 # convert to unit vector
                        L[it1,it2,it3,it4] = 1/(np.matrix.getH(v) @ Q @ v + stab) # MUSIC estimator
        return L
                                             
    ## SH-wave
    if wtype == 'SH':
        L = np.zeros((len(test_param.theta),len(test_param.phi),len(test_param.vs)))
        for it1 in range(0,len(test_param.theta)):
            for it2 in range(0,len(test_param.phi)):
                for it3 in range(0,len(test_param.vs)):
                    param.theta = test_param.theta[it1]
                    param.phi   = test_param.phi[it2]
                    param.vs    = test_param.vs[it3]
                    v = polvect6C(param,v_scal,'SH')                      # calculate test polarization vector
                    v = v / np.linalg.norm(v)                             # convert to unit vector
                    L[it1,it2,it3] = 1/(np.matrix.getH(v) @ Q @ v + stab) # MUSIC estimator
        return L
    
    ## Rayleigh-wave
    if wtype == 'R':
        L = np.zeros((len(test_param.phi),len(test_param.xi),len(test_param.vR)))
        for it1 in range(0,len(test_param.phi)):
            for it2 in range(0,len(test_param.xi)):
                for it3 in range(0,len(test_param.vR)):
                    param.phi  = test_param.phi[it1]
                    param.xi = test_param.xi[it2]
                    param.vR  = test_param.vR[it3]
                    v = polvect6C(param,v_scal,'R')                       # calculate test polarization vector
                    v = v / np.linalg.norm(v)                             # convert to unit vector
                    L[it1,it2,it3] = 1/(np.matrix.getH(v) @ Q @ v + stab) # MUSIC estimator
        return L
    
    ## Love-wave
    if wtype == 'L':
        L = np.zeros((len(test_param.phi),len(test_param.vL)))
        for it1 in range(0,len(test_param.phi)):
            for it2 in range(0,len(test_param.vL)):
                param.phi = test_param.phi[it1]
                param.vL  = test_param.vL[it2]
                v = polvect6C(param,v_scal,'L')                   # calculate test polarization vector
                v = v / np.linalg.norm(v)                         # convert to unit vector
                L[it1,it2] = 1/(np.matrix.getH(v) @ Q @ v + stab) # MUSIC estimator
        return L