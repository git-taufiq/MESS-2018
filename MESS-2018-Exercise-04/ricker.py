import warnings
import numpy as np

def ricker(dt, fdom, tlength):
    '''
    ricker returns a ricker wavelet
    
    dt: desired temporal sample rate (in seconds)
    fdom: dominant frequency of Ricker wavelet (in Hz)
    tlength = wavelet length (in seconds)
  
    '''
    
    # check input parameter
    if fdom <= 0:
        raise ValueError('Dominant frequency (fdom) needs to be positive.')

    if tlength <= 0:
        raise ValueError('Signal length (tlength) needs to be positive.')

    if dt <= 0:
        raise ValueError('Time interval (dt) needs to be positive.')

    else:
        # create a time vector
        nt   = round(tlength/dt)+1
        tmin = -dt*round(nt/2)
        tmax = -tmin
        tw   = np.arange(tmin,tmax+dt,dt)
        
        # create the wavelet
        pf = np.pi**2 * fdom**2
        wavelet = (1 - 2 * pf * tw**2) * np.exp(-pf * tw**2)
        
        # normalize
        # generate a reference sinusoid at the dominant frequency
        refwave = np.sin(2 * np.pi * fdom * tw)
        reftest = np.convolve(refwave,wavelet,'same')
        fact = max(refwave)/max(reftest)
        wavelet = wavelet * fact
        
        return wavelet