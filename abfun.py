# Mildred A. Herrera, Wojciech Ambrosiak
# Jan Siemens lab
# University of Heidelberg

import matplotlib.pyplot as plt #library for plotting
import numpy as np #numeric library
import pyabf #library to open *.abf files
from scipy.signal import medfilt
from scipy.signal import convolve
from scipy.signal import savgol_filter
from scipy import sparse
from scipy.sparse.linalg import spsolve
from collections import Counter
from bisect import bisect_left
import struct
import matplotlib.colors as mcolors
import itertools,operator

def abf_me(filename, sweep, no_channels):
  
    abf = pyabf.ABF(filename)
    # plot the first channel
    abf.setSweep(sweep) 
    
    abf.setSweep(sweepNumber=0, channel=no_channels[0]) # Voltage
    t = np.array(abf.sweepX) #time
    v = np.array(abf.sweepY) #voltage, both are np.ndarray

    # plot the second channel
    abf.setSweep(sweepNumber=0, channel=no_channels[1]) # Temperature
    c = np.array(abf.sweepY) #temperature
    
    return t, v, c

def baseline_als(y, lam, p, niter):
    L = len(y)
    D = sparse.diags([1,-2,1],[0,-1,-2], shape=(L,L-2))
    w = np.ones(L)
    for i in range(niter):
        W = sparse.spdiags(w, 0, L, L)
        Z = W + lam * D.dot(D.transpose())
        z = spsolve(Z, w*y)
        w = p * (y > z) + (1-p) * (y < z)
        return z
    #------
    # Based on: "Asymmetric Least Squares Smoothing" by P. Eilers and H. Boelens in 2005.
    # https://stackoverflow.com/questions/29156532/python-baseline-correction-library
    # Attribution-ShareAlike 3.0 Unported (CC BY-SA 3.0)
    # Comment on stackoverflow: 
    # We found that generally 0.001 ≤ p ≤ 0.1 is a good choice (for a signal with positive peaks) 
    # and 10^2 ≤ λ ≤ 10^9 , but exceptions may occur. 
    # In any case one should vary λ on a grid that is approximately linear for log λ>> – 
    # glycoaddict May 30 '18 at 19:08
    #------

def valleyser(z, t, thr):
    # As a test only a range. Later Y = z
    Y = z
    T = t

    # The derivative of the signal can be extracted by using a kernel and performing a convolution.
    kernel = [1, 0, -1]
    dY = convolve(z, kernel, 'valid') 


    # *Checking for sign-flipping.
    # *The sign function returns -1 if x < 0, 0 if x==0, 1 if x > 0. nan is returned for nan inputs.*
    S = np.sign(dY)
    ddS = convolve(S, kernel, 'valid')

    # *These candidates are basically all negative slope positions dY < 0
    # *Add one since using 'valid' shrinks the arrays:
    candidates = np.where(dY < 0)[0] + (len(kernel) - 1) 

    # *Here they are filtered on actually being the final such position in a run of negative slopes:
    valleys = []
    valleys = sorted(set(candidates).intersection(np.where(ddS == 2)[0] + 1))


    # Identifying wrong valleys. Wrong valleys are those that are in peaks areas.
    wrong_valleys_val = []
    for idx, val in enumerate(valleys):
        if Y[val] > thr:
            wrong_valleys_val.append(valleys[idx])
        
    # Removing wrong valleys:
    valleys_id = sorted(set(valleys) - set(wrong_valleys_val)) #Set operation.
    
    return valleys_id

    #The initial part of this function comes from: https://newbedev.com/find-peaks-location-in-a-spectrum-numpy 
    # licensed under a Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License.  @ 2021 newbedev
    # The idea was to extract instead of the peaks the valleys, so we could have the data we are not interested excluded from the analysis, 
    # as the APs varied significantly between conditions(temperatures).

def peakstartend(z, valleys):
    # Requires valleys id (valleys), a signal (z).
    # Creates a vector of ones the size of z.
    # Assigns a 0 if the idx is a valley position.
    # Peaks become only the remaining ones.
    # Starting and end position of a peak are saved as pairs = peaks_st_end

    # Making a vector of zeros where there are valleys and ones when there are peaks.
    binvec = np.ones(len(z))
    for idx, val in enumerate(valleys):
        binvec[val] = 0
    peaks_st_end = []
    # Finding the first and last position of a peak:
    peaks_st_end = np.where(np.diff(np.hstack(([False],binvec==1,[False]))))[0].reshape(-1,2)

    return peaks_st_end

#Calling the function:
#peaks_st_end = peakstartend(z, valleys)
#c1_st_end = peakstartend(c, c_1)

def cstartend(c, c_x):
    # Making a vector of zeros where there are valleys and ones when there are peaks.
    binvec = np.zeros(len(c))
    for idx, val in enumerate(c_x):
        binvec[val] = 1
    c_st_end = []
    # Finding the first and last position of a peak:
    c_st_end = np.where(np.diff(np.hstack(([False],binvec==1,[False]))))[0].reshape(-1,2)

    return c_st_end

#Calling the function:
#c1_st_end = cstartend(c, c_1)

def peaklen(peaks_st_end):
    # ex. [100, 7000]
    # Finding the length of a peak.
    lenpeak = []
    for ii in range (0,len(peaks_st_end)):
        lenpeak.append(np.diff(peaks_st_end[ii,:]))
    lenpeak = np.array(np.hstack(lenpeak))
    return lenpeak

#Calling the function:
#lenpeak = peaklen(peaks_st_end)

#c1_lenpeak= peaklen(c1_st_end)
#print(lenpeak[0:100])

def idx_peaks_startend(peaks_st_end,lenpeak,min_length):
    
    # Obtaining individual start and end of peaks:
    idx_peaks = []
    idx_peak_start = []
    idx_peak_end = []
    for ii in range (0,len(peaks_st_end)):
        if lenpeak[ii] > min_length:
            idx_peak_start.append(peaks_st_end[ii,0])
            idx_peak_end.append(peaks_st_end[ii,-1])
    return idx_peak_start, idx_peak_end
    
#Calling the function:
#min_length = 400
#idx_peak_start, idx_peak_end = idx_peaks_startend(peaks_st_end,lenpeak,min_length)

#min_length_c = sampling_rate * window
#idx_c1_start, idx_c1_end = idx_peaks_startend(c1_st_end,c1_lenpeak,min_length_c)

def peaksy(idx_peak_start, idx_peak_end):
    # Making an index vector for each peak Ex. [1,2,3],[100,101,102].
    peaks_vec = []
    for tt in range (0,len(idx_peak_start)):
        peaks_vec.append((np.array(range(idx_peak_start[tt],idx_peak_end[tt]))))

    peaks = []
    peaks =np.array(np.hstack(peaks_vec))
   
    return peaks_vec, peaks

#Calling the function:
#peaks_vec, peaks = peaksy(idx_peak_start, idx_peak_end)
#c1_vec, c1_id = peaksy(idx_c1_start, idx_c1_end)

#len(c1_vec[0])

def peaktop(peaks_vec, z, t, thr, change_l, change_r, total_change):
    peaktop_v1 = []
    peaktop_id = []
    peaktop_t = []
    peaktop_v = []
    # Finding maxima of peaks in voltage:
    for ii in range (0,len(peaks_vec)):
        peaktop_v1.append(np.max(z[peaks_vec[ii]]))    
        
    peaktop_v_corr = []
    for jj in range (0,len(peaktop_v1)):
        if peaktop_v1[jj] > thr:
            peaktop_v_corr.append(peaktop_v1[jj])
            
    peaktop_v1 = list(np.hstack(peaktop_v_corr))
    #peaktop_v = np.array(np.hstack(peaktop_v))

    for ii in range (0,len(peaktop_v1)):
        peaktop_id.append(np.where(z == peaktop_v1[ii]))
    peaktop_id = np.array(np.hstack(peaktop_id))
    peaktop_id = np.array(np.hstack(peaktop_id)) # for some reason I have to repeat this.
    
    peaktop_id_corr = []

    l_ch = []; r_ch = []
    for ii in range(0,len(peaktop_id)):
        if (abs(z[peaktop_id[ii]] - z[peaktop_id[ii]-change_l])>= total_change) and         (abs(z[peaktop_id[ii]] - z[peaktop_id[ii]+change_r])>= total_change):
            peaktop_id_corr.append(peaktop_id[ii])

        #l_ch.append(abs(z[peaktop_id[ii]] - z[peaktop_id[ii]-change_l])> total_change)
        #r_ch.append(abs(z[peaktop_id[ii]] - z[peaktop_id[ii]+change_r])> total_change)
          #  peaktop_id_corr = []
    
        #peaktop_id_corr.append(np.where(peaks_change[ii] == [True,True]))
    peaktop_id = np.array(np.hstack(peaktop_id_corr))

    for ii in range (0,len(peaktop_id)):
        peaktop_v.append((z[peaktop_id[ii]]))
        peaktop_t.append((t[peaktop_id[ii]]))

    #return peaktop_id, peaktop_v, peaktop_t
    return peaktop_id, peaktop_v, peaktop_t

def freqx(t,z,peaktop_id,sampling_rate,window):

    peaktops = peaktop_id
    bins = np.arange(0,len(z),sampling_rate * window)
    digitized = np.digitize(peaktops,bins)
    list_peaks = [peaktop_id[digitized == i] for i in range(1, len(bins)+1)]
    list_peaks = np.array(list_peaks)

    peak_times = []
    for ii in range(0, len(peaktop_id)):
        peak_times.append(t[peaktop_id[ii]])
    peak_times = np.array(peak_times)
    
    freq_total = []
    for ii in np.arange(0,len(list_peaks)):
        freq_total.append(len(list_peaks[ii]))

    time_ch = t[range(0, len(t), sampling_rate * window)] 
    loc_int = np.ravel(np.searchsorted(time_ch, peak_times, side ='right'))
    freq_vec = list(Counter(loc_int).items())
    freq_list = []
    freq_list = np.hstack([item[1] for item in freq_vec])
    return freq_list, list_peaks

def freqx2(t,T,z,peaktop_id,sampling_rate,window):

    peaktops = peaktop_id
    bins = np.arange(0,len(z),sampling_rate * window)
    digitized = np.digitize(peaktops,bins)
    list_peaks = [peaktop_id[digitized == i] for i in range(1, len(bins)+1)]
    list_peaks = np.array(list_peaks)

    #peak_times = []
    #time_ch = np.arange(T[0], T[-1], sampling_rate * window)
    #for ii in range(0, len(peaktop_id)):
    #    peak_times.append(T[peaktop_id[ii]])
    #peak_times = np.array(peak_times)
    
    #freq_total = []
    #for ii in np.arange(0,len(list_peaks)):
    #    freq_total.append(len(list_peaks[ii]))

    #loc_int = np.ravel(np.searchsorted(time_ch, peak_times, side ='right'))
    #freq_list = list(Counter(list_peaks[ii] for ii in range (0, len(list_peaks))))
    #freq_list = []
    #freq_list = np.hstack([item[1] for item in freq_vec])
    return list_peaks

def xpex(z,t,peaktop_id,lim_l,lim_r, var):
    #var = either signal, time or mean
    sZ = []
    sT = []
    # maximum_left =2000, maximum_right = 3500
    for ii in range(0,len(peaktop_id)):
        sZ.append(z[peaktop_id[ii]-lim_l:peaktop_id[ii]+lim_r]) #Making new vectors
        sT.append(t[peaktop_id[ii]-lim_l:peaktop_id[ii]+lim_r]) #Making new vectors

    sZ = np.array(sZ)
    sT = np.array(sT)
    #sT0 = sT[0]
    mZ = sZ.mean(axis=0)
            
    if var == 'signal':
        return sZ
    if var == 'time':
        return sT
    if var == 'mean':
        return mZ

def peakbase(mZ, t, chunk_size, savgol_window, savgol_polynomial, var):
    #mZ0 = np.array(mZ[0])
#Cutting the mean AP into chunks of 50:
    chunk_mZ = []
    mZ = np.array(mZ)
    chunk_mZ = mZ.reshape(-1,chunk_size)
    mean_vals = np.mean(chunk_mZ,axis = 1) #making means across rows.

# If there was a spike or change in the signal - each value minus the mean per row should be a number not close to 0.
# for example the mean of a row is 20, and the spike is 100, 100 - 20 = 80. 
# If there are almost no changes the value should be close to 0.
# We use abs, because there are negative values.

    temp_vals = []
    for ii in range(0,len(chunk_mZ)):
        temp_vals = np.append(temp_vals,abs(mean_vals[ii] - chunk_mZ[ii,:])*10)
    pol_vals = savgol_filter(temp_vals, savgol_window, savgol_polynomial) # Using a window of x datapoints, and a x order polynomial.
    template = []
    template = np.multiply(np.isclose(pol_vals,pol_vals.min(), atol=0.2),1)
    #template, template_vals = find_nearest(pol_vals, pol_vals.min())
#temp_site_ch = np.multiply(np.isclose(pol_vals,-2, atol=0.5),1)
    if var == 'peakchange1':
        return pol_vals
    if var == 'peakchange2':
        return template


#chunk_size = 50 # Lim_r + Lim_l should be divisible by this number. 
# E.g. 50 for 120 will raise an error, contrary to 150.
#pol_vals = []
#template = []
#savgol_window = 101 # Needs to finish with 1.
#savgol_polynomial = 1 # first order polynomial
#temp_site, max_change = peakbase(peaktop_id, z, t, savgol_window, savgol_polynomial)
# Result should be a pyramid. If the sides do not reach the base, lim_r or lim_r should be longer.
# Second image sides should reach 1.
#for ii in range(0, len(mZ)):
#    pol_vals.append(peakbase(mZ[ii], t, savgol_window, savgol_polynomial, var = 'peakchange1'))
#for ii in range(0, len(mZ)):
#    template.append(peakbase(mZ[ii], t, savgol_window, savgol_polynomial, var = 'peakchange2'))

#Plotting all mean APs in all bins of 30 seconds:
#color=iter(cm.cool(np.linspace(0,1,len(pol_vals))))
#plt.title('Check pyramid')
#for ii in range(0,len(pol_vals)):
#    c=next(color)
#    plt.plot(pol_vals[ii], c=c, linestyle=':') #only means of all windows
#plt.show()

#color=iter(cm.cool(np.linspace(0,1,len(pol_vals))))
#plt.title('Check at least one top 1, base 0')
#for ii in range(0,len(pol_vals)):
#    c=next(color)
#    plt.plot(template[ii], c=c, linestyle=':') #only means of all windows
#plt.show()


def find_nearest(array, value):
    array = np.array(array)
    idx = (np.abs(array - value)).argmin()
    return idx, array[idx]

def basepoints(template):
    no_change_st_end = []
    #no_change_st_end = np.where(np.diff(np.hstack(([False],template==1,[False]))))[0].reshape(-1,2)
    r = max((list(y) for (x,y) in itertools.groupby((enumerate(template)),operator.itemgetter(1)) if x == 0), key=len)
    #base_l_id, base_r_id = no_change_st_end[0,1], no_change_st_end[-1,0]
    base_id = []
    base_id = [r[0][0], r[-1][0]]
    return base_id
    #return no_change_st_end


#base_id = []
#base_id2 = []

#for ii in range(0, len(template)):
#    base_id.append(basepoints(template[ii]))

#print(base_id)

def peakdur(mZ, t, sampling_rate, lim_r, lim_l, base_r):
    #baseline_areas = np.hstack(np.where(temp_site == temp_site.max())) #no major change areas
    #mean_peaktop_id = np.hstack(np.where(mZ == mZ.max())) 
    mean_peaktop_id = lim_r # The peaktop will be at the position of lim_r
    print(mZ[mean_peaktop_id])
    
    sTW = t[0:(lim_r + lim_l)] #Time vector the duration of lim_r + lim_l
    half_id_v0 = 0 - (mZ[mean_peaktop_id] - mZ[base_r]) / 2
    half_id_v = mZ[mean_peaktop_id] + half_id_v0
    print(len(sTW))
    print(len(mZ))
    
    #half_id_v = (sTW[base_r] - sTW[mean_peaktop_id]) / 2
    print(half_id_v)

    #half_value = mZ[mean_peaktop_id] - half_id
    
    array_l_v = np.array(mZ[0:mean_peaktop_id])
    array_r_v = np.array(mZ[mean_peaktop_id:-1])
    
    #array_l_v = np.array(sTW[0:mean_peaktop_id])
    #array_r_v = np.array(sTW[mean_peaktop_id:-1])
    
    #print(array_r_v[100])
    ##array_r = mZ0[mean_peaktop_id,-1]
    
    hpoint_2, hpoint_2_val = find_nearest(array_r_v, half_id_v + sTW[lim_r])
    #hpoint_1, hpoint_1_val = find_nearest(array_l_v, mZ[hpoint_2]) #array_r_v[hpoint_2])
    hpoint_1, hpoint_1_val = find_nearest(array_l_v, half_id_v - sTW[lim_r]) #array_r_v[hpoint_2])
    hpoint_2 = lim_l + hpoint_2
    duration = sTW[hpoint_1]-sTW[hpoint_2]
    return mean_peaktop_id, duration, hpoint_1, hpoint_2

#mean_peaktop_id, duration, hpoint_1, hpoint_2 = peakdur(mZ[0], t, sampling_rate, lim_r, lim_l, base_id[0][1])

#print(mean_peaktop_id, duration, hpoint_1, hpoint_2)
#sTW = t[0:(lim_r + lim_l)]
#mZW = mZ[0]
#bZW = base_id[0][1]
#bZW2 = base_id[1][0]

#plt.plot(sTW, mZW)
#plt.plot(sTW[baseline_l_id], mZ0[baseline_l_id], 'bo')
#plt.plot(sTW[bZW], mZW[bZW], 'go')
#plt.plot(sTW[bZW2], mZW[bZW2], 'ro')

#plt.plot(sTW[mean_peaktop_id], mZW[mean_peaktop_id], 'k*')
#plt.plot(sTW[hpoint_1], mZW[hpoint_1], 'r*')
#plt.plot(sTW[hpoint_2], mZW[hpoint_2], 'c*')

#plt.show()

def raster_vec(z, t, peaktop_id, sampling_rate, window):
    time_vec = t[0:sampling_rate*window]
    binvec = np.zeros(len(z))
    for idx, val in enumerate(peaktop_id):
        binvec[val] = 1    
   
    freq_vec = []
    freq_id = []
    rows = int(len(binvec)/(sampling_rate * window))
    binvec = np.array(binvec[0:int(rows * sampling_rate * window)])
    
    freq_vec = np.reshape(binvec,(-1, sampling_rate * window))
    for ii in range (0,len(freq_vec)):
        freq_id.append(time_vec[np.nonzero(freq_vec[ii])])
    spike_data = np.array(freq_id)
    
    #freq_id = np.where(freq_vec, axis=1)
    #freq_idx = [np.nonzero(t)[0] for t in freq_id]

    return spike_data
#Calling the function:
#sampling_rate = 20000; win = 30
#spike_data = raster_vec(z,t,peaktop_id, sampling_rate, win)


# In[ ]:


#Plotting raster plot - Slow! Plot it in R.


# Set spike colors for each neuron

#lineSize = [0.4, 0.3, 0.2, 0.8, 0.5, 0.6, 0.7, 0.9]                                  

# Draw a spike raster plot
#cm_cool = plt.cm.get_cmap('cool')

#cNorm  = mcolors.Normalize(vmin=0, vmax=len(spike_data))
#scalarMap = cm.ScalarMappable(norm=cNorm, cmap=cm_cool)
#colorVal = [scalarMap.to_rgba([ii]) for ii in range(0,len(spike_data))]

#plt.eventplot(spike_data, color = colorVal)     
#plt.title('Spike raster plot C1')
#plt.xlabel('Neuron')
#plt.ylabel('Spike')
#plt.show()

