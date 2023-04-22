import pyabf
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve
from scipy import signal
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def get_file_name(abf_file):
    return abf_file.split(str(Path("/")))[-1]

def rolling_std(a, window=20):
    pd_a = pd.Series(a)
    pd_std = pd_a.rolling(window, min_periods=1).std()
    return np.array(pd_std.values)

def smooth_data(dataset, window=20):
    # takes in dataset as a 1D np array, returns moving time avg
    return np.convolve(dataset, np.ones((window,))/float(window), mode='same')

def nanargmax(a):
    idx = np.argmax(a, axis=None)
    multi_idx = np.unravel_index(idx, a.shape)
    if np.isnan(a[multi_idx]):
        nan_count = np.sum(np.isnan(a))
        # In numpy < 1.8 use idx = np.argsort(a, axis=None)[-nan_count-1]
        idx = np.argpartition(a, -nan_count-1, axis=None)[-nan_count-1]
        multi_idx = np.unravel_index(idx, a.shape)
    return multi_idx

def baseline_als(y, lam, p, niter=10):
  L = len(y)
  D = sparse.diags([1,-2,1],[0,-1,-2], shape=(L,L-2))
  w = np.ones(L)
  for i in range(niter):
    W = sparse.spdiags(w, 0, L, L)
    Z = W + lam * D.dot(D.transpose())
    z = spsolve(Z, w*y)
    w = p * (y > z) + (1-p) * (y < z)
  return z

def unpack_data(data_pair):
    patch_r = pyabf.ABF(data_pair[0])
    nea_r = pyabf.ABF(data_pair[1])
    patch_r.setSweep(0)
    nea_r.setSweep(0)
    patch_x = patch_r.sweepX
    patch_y = patch_r.sweepY
    nea_x = nea_r.sweepX
    nea_y = nea_r.sweepY
    patch_f = patch_r.dataRate
    nea_f = nea_r.dataRate
    patch = patch_x, patch_y, patch_f
    nea = nea_x, nea_y, nea_f
    return(patch, nea, data_pair)

def trim_data(data_pair, offset=0, lam=2*10**13, p=0.015, niter=3):
    patch, nea, names = data_pair
    patch_x, patch_y, patch_f = patch
    nea_x, nea_y, nea_f = nea
    patch_std = rolling_std(patch_y, 20)
    nea_std = rolling_std(nea_y, 20)
    patch_max = nanargmax(patch_std)[0] + int(offset*patch_f)
    nea_first = np.argmax(np.abs(nea_y)) + int(offset*nea_f)
    nea_stdmax = nanargmax(nea_std)[0] + int(offset*nea_f)
    nea_max = min(nea_first, nea_stdmax)
    if patch_max >= len(patch_y) or nea_max >= len(nea_y):
        return
    patch_t_zero = patch_x[patch_max]
    nea_t_zero = nea_x[nea_max]
    
    patch_t = patch_x[patch_max:] - patch_t_zero
    nea_t = nea_x[nea_max:] - nea_t_zero
    patch_v = smooth_data(patch_y, 20)[patch_max:]
    nea_v = smooth_data(nea_y, 20)[nea_max:]
    
    patch_downsampling_ratio = patch_f//nea_f
    #print(patch_downsampling_ratio)
    patch_t = patch_t[0::patch_downsampling_ratio]
    patch_v = patch_v[0::patch_downsampling_ratio]
    
    bas_patch = baseline_als(patch_v, lam, p, niter)
    bas_nea = baseline_als(nea_v, lam, p, niter)
    
    return ((patch_t, patch_v, bas_patch), (nea_t, nea_v, bas_nea), names)

def plot_trim_data(data):
    if data is None:
        print("Data was not trimmed successfully. Please check raw data.")
        return
    ((patch_t, patch_y, bas_patch), (nea_t, nea_y, bas_nea), names) = data
    fig, ax = plt.subplots(2,1,figsize=(8,6))
    fig.suptitle("Trimmed baseline preview")
    ax[0].plot(patch_t, patch_y)
    ax[0].plot(patch_t, bas_patch)
    ax[0].set_title(get_file_name(names[0]))
    ax[1].plot(nea_t, nea_y)
    ax[1].plot(nea_t, bas_nea)
    ax[1].set_title(get_file_name(names[1]))

def correlate_signal(signal_pair, crop_signal_time=10, crop_signal_start=25):
    if signal_pair is None:
        print("Data was not trimmed successfully. Please check raw data.")
    ((patch_t, patch_y, bas_patch), (nea_t, nea_y, bas_nea), data_pair_names) = signal_pair
    patch_y = patch_y - bas_patch
    nea_y = nea_y - bas_nea
    dx = np.mean(np.diff(nea_t))
    start = int(crop_signal_start/dx)
    stop = int((crop_signal_start+crop_signal_time)/dx)             
    patch_y_crop = patch_y[start:stop]
    nea_y_crop = nea_y[start:stop]
    nea_y_crop = nea_y_crop - min(nea_y_crop)
    patch_y_crop = patch_y_crop - min(patch_y_crop)
    nea_y_crop = nea_y_crop/max(nea_y_crop)
    patch_y_crop = patch_y_crop/max(patch_y_crop)
    patch_t_crop = patch_t[start:stop]
    nea_t_crop=nea_t[start:stop]
    shift = (np.argmax(signal.correlate(patch_y_crop, nea_y_crop, mode='full')) - len((nea_y_crop)-1))*dx
    raw = ((patch_t, patch_y), (nea_t, nea_y))
    crop = ((patch_t_crop, patch_y_crop), (nea_t_crop, nea_y_crop))
    
    return(raw, crop, shift, dx, data_pair_names)

def correlate_with_std(signal_pair, crop_signal_time=10, crop_signal_start=25):
    if signal_pair is None:
        print ("Data was not trimmed successfully. Please check raw data.")
    ((patch_t, patch_y, bas_patch), (nea_t, nea_y, bas_nea), data_pair_names) = signal_pair
    patch_y = patch_y - bas_patch
    nea_y = nea_y - bas_nea
    dx = np.mean(np.diff(nea_t))
    start = int(crop_signal_start/dx)
    stop = int((crop_signal_start+crop_signal_time)/dx)             
    patch_y_crop = patch_y[start:stop]
    nea_y_crop = nea_y[start:stop]
    nea_y_crop = nea_y_crop - min(nea_y_crop)
    patch_y_crop = patch_y_crop - min(patch_y_crop)
    nea_y_crop = nea_y_crop/max(nea_y_crop)
    patch_y_crop = patch_y_crop/max(patch_y_crop)
    patch_t_crop = patch_t[start:stop]
    nea_t_crop=nea_t[start:stop]
    # shift in time (add to nea) calculated by cross-correlation
    shift_idx = (np.argmax(signal.correlate(patch_y_crop, nea_y_crop, mode='full')) - len((nea_y_crop)-1))
    print(shift_idx)
    dx_idx = int(1/dx)
    if shift_idx >= 0:
        # need to trim off from patch
        patch_slice = patch_y_crop[shift_idx:shift_idx+dx_idx]
        nea_slice = nea_y_crop[0:dx_idx]
    else:
        patch_slice = patch_y_crop[0:dx_idx]
        nea_slice = nea_y_crop[-shift_idx:-shift_idx+dx_idx]
    patch_slice_std = rolling_std(patch_slice)[20:]
    nea_slice_std = rolling_std(nea_slice)[20:]
    shift_id2 = np.argmax(patch_slice_std) - np.argmax(nea_slice_std)
    print(shift_id2)
    shift = (shift_idx + shift_id2)*dx
    raw = ((patch_t, patch_y), (nea_t, nea_y))
    crop = ((patch_t_crop, patch_y_crop), (nea_t_crop, nea_y_crop))
    return(raw, crop, shift, dx, data_pair_names)

def plot_correlation(correlated_data, stretch=1):
    raw, crop, shift, dx, data_pair_names = correlated_data
    ((patch_t, patch_y), (nea_t, nea_y)) = raw 
    ((patch_t_crop, patch_y_crop), (nea_t_crop, nea_y_crop)) = crop
    shift_idx = int(shift/dx)
    if shift_idx >= 0:
        patch_y_crop = patch_y_crop[shift_idx:]
        patch_t_crop = patch_t_crop[shift_idx:]
        nea_t_crop = nea_t_crop[:-shift_idx]
        nea_y_crop = nea_y_crop[:-shift_idx]
    else:
        nea_t_crop = nea_t_crop[-shift_idx:]
        nea_y_crop = nea_y_crop[-shift_idx:]
        patch_t_crop = patch_t_crop[:shift_idx]
        patch_y_crop = patch_y_crop[:shift_idx]
    nea_t_crop = np.linspace(patch_t_crop[0], patch_t_crop[-1]*stretch, len(patch_t_crop))
    fig, ax = plt.subplots(2,1,figsize=(8,6))
    fig.suptitle(get_file_name(data_pair_names[1]))
    ax[0].plot(patch_t, patch_y, label='patch')
    ax[0].plot(nea_t, nea_y, label='NEA')
    ax[0].set_title('Pre-align preview')
    ax[0].legend(loc=1)
    ax[1].plot(patch_t_crop, patch_y_crop, label='patch')
    ax[1].plot(nea_t_crop, nea_y_crop, label='NEA')
    ax[1].set_title('Aligned data')
    ax[1].set_ylabel('arbitrary units')
    ax[1].set_xlabel('time')
    ax[1].legend(loc=1)

"""
def plot_correlation(correlated_data):
    raw, crop, shift, data_pair_names = correlated_data
    ((patch_t, patch_y), (nea_t, nea_y)) = raw 
    ((patch_t_crop, patch_y_crop), (nea_t_crop, nea_y_crop)) = crop
    fig, ax = plt.subplots(2,1,figsize=(8,6))
    fig.suptitle(get_file_name(data_pair_names[1]))
    ax[0].plot(patch_t, patch_y, label='patch')
    ax[0].plot(nea_t, nea_y, label='NEA')
    ax[0].set_title('Pre-align preview')
    ax[0].legend(loc=1)
    ax[1].plot(patch_t_crop, patch_y_crop, label='patch')
    ax[1].plot(nea_t_crop+shift, nea_y_crop, label='NEA')
    ax[1].set_title('Aligned data')
    ax[1].set_ylabel('arbitrary units')
    ax[1].set_xlabel('time')
    ax[1].legend(loc=1)
"""
