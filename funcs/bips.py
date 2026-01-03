import numpy as np
from scipy.signal.windows import tukey


class beep:

    def __init__(self):
        self.fs = 44100.0
    
    def make_bip(self, dur, freqs, envs):
        self.dur = dur
        self.freqs = freqs
        self.Nf = len(self.freqs)       # number of frequencies
        self.Nt = int(self.dur * self.fs)    # number of time points
        self.t_vec = np.linspace(0,self.dur,self.Nt) 
        
        self.oscmat = np.zeros((self.Nf,self.Nt)) 
        # Nf x Nt matrix that is the oscillator bank:
        # Nf rows = oscillators at each frequency. 
        # Nt columns = time
        self.env_amps = np.zeros((self.Nf,self.Nt))     # envelope functions for each oscillator
            # same dimensions as oscmat
        self.env_amps = envs  
        # fill in the oscillator bank matrix  
        for ifrx,f in enumerate(self.freqs):  
            self.oscmat[ifrx,:] = np.exp(1j*2*np.pi*f*self.t_vec) * self.env_amps[ifrx,:]

        # compress the oscillator bank into a single time series: waveform, wf
        wf = np.sum(self.oscmat, axis=0)
        wf = wf/(np.max(np.abs(wf))) # normalize
        self.wf = wf # assign to self
        return wf # wf = waveform
    
# =====================================
# FUNCTIONS OUTSIDE THE CLASS
# =====================================

def sliding_gaussian(length, stds, means, x=None):
    """
    Generate a stack of Gaussian windows with sliding means.
    
    Parameters:
        length (int): Length of each window.
        std (float): Standard deviation of the Gaussian.
        means (array-like): Array of mean positions (center of the Gaussian).
        x (array-like, optional): The x-axis values. If None, uses np.arange(length).
        
    Returns:
        np.ndarray: 2D array of shape (len(means), length), each row is a Gaussian.
    """
    if x is None:
        x = np.arange(length)
    gaussians = []
    for mu,std in zip(means,stds):
        g = np.exp(-0.5 * ((x - mu) / std) ** 2)
        gaussians.append(g)
    gaussmat = np.asarray(gaussians) # got rid of the transpose
    return gaussmat # shape (len(means), length)

def taper_waveform(wf, alpha=0.9):
    """
    Taper the ends of a timeseries to zero using a Tukey window.
    
    Parameters:
        signal (np.ndarray): Input 1D timeseries.
        alpha (float): Shape parameter of the Tukey window (0=rectangular, 1=Hann).
        
    Returns:
        np.ndarray: Tapered timeseries.
    """
    window = tukey(len(wf), alpha)
    return wf * window


def plot_waveforms(ax1, ax2, t_short, wf_short, t_full, wf_full,
                   short_title='A short segment', full_title='Full Duration Waveform',
                   xlabel='Time (s)', ylabel='Amplitude'):
    """Plot a short segment and the full waveform on the provided axes."""

    ax1.plot(t_full, wf_full)
    ax1.set_title(full_title)
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel(ylabel)
    ax1.vlines(t_short[0], -1, 1, colors='red', linestyle='--')
    ax1.vlines(t_short[-1],-1, 1, colors='red', linestyle='--')

    ax2.plot(t_short, wf_short)
    ax2.set_title(short_title)
    ax2.set_xlabel(xlabel)
    ax2.set_ylabel(ylabel)
    #ax1.grid(True)


    #ax2.grid(True)
