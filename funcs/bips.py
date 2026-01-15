import numpy as np
from scipy.signal.windows import tukey
import matplotlib.pyplot as plt
import matplotlib.patches as patches


class beep:

    def __init__(self):
        self.fs = 44100.0
        self.pitch_classes = ['c','c#','d','d#','e','f','f#','g','g#','a','a#','b']

    def make_bip_1f(self, dur, freq, env):
        self.dur = dur
        self.freq = freq
        self.Nt = int(self.dur * self.fs)    # number of time points
        self.t_vec = np.linspace(0,self.dur,self.Nt) 
        
        self.env_amp = env  
        self.osc = np.exp(1j*2*np.pi*self.freq * self.t_vec) 

        # compress the oscillator bank into a single time series: waveform, wf
        wf = self.osc * self.env_amp[:]
        wf = wf/(np.max(np.abs(wf))) # normalize
        self.wf = wf # assign to self
        return wf # wf = waveform
    
    def make_bip(self, dur, freqs, envs):
        self.dur = dur
        self.freqs = freqs
        #if type(freqs) is not np.ndarray:
        #    self.freqs = np.asarray([freqs]) # make it an array if a single frequency is given
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
# FUNCTIONS OUTSIDE THE BEEP CLASS
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

def make_3pt_envelope(Npts,peak_posn):
    envelope = np.zeros(Npts)
    peak = int(Npts*peak_posn)

    # from 0 to the peak index (peak):
    up = np.linspace(0,1,peak)
    envelope[:peak] = up

    # and fill in the rest: 
    down = np.linspace(1,0,(Npts-peak))
    envelope[peak:] = down
    
    return envelope


def makePitchRing(indexes):
    pitch_classes = ['c','c#','d','d#','e','f','f#','g','g#','a','a#','b']
    circle = np.linspace(0,2*np.pi,64)
    r = 1.0
    x = r*np.sin(circle)
    y = r*np.cos(circle)

    # the note locations. 
    base_dots = np.linspace(0,2*np.pi,13)
    xd = r*np.sin(base_dots)
    yd = r*np.cos(base_dots)

    # the text locations
    r = 1.15
    xt = r*np.sin(base_dots)
    yt = r*np.cos(base_dots)

    # ========================
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111, aspect='equal')

    # (0) plot a filled square with a filled circle in it...
    # patches.Rectangle((x,y,lower left corner),width,height)
    #ax1.add_patch(patches.Rectangle((0.1, 0.1),0.5,0.5,facecolor="red"))

    ax1.add_patch(patches.Rectangle((-1.25, -1.25),2.5,2.5,facecolor=[0.6, 0.6, 0.6]))
    ax1.plot(x,y,'k-')
    ax1.plot(xd,yd,'w.')

    radius_norm = 0.08  # radius normalized, scaled to size of box

    for ind,interval in enumerate(indexes):
        interval = int(interval)
        # print(ind,interval)
        ax1.add_patch(patches.Circle((xd[interval], yd[interval]),radius_norm,facecolor="red")) 
        ax1.text(xt[interval], yt[interval], pitch_classes[interval])
        
    ax1.get_xaxis().set_visible(False)
    ax1.get_yaxis().set_visible(False)
    plt.show()