# STFT
from scipy.signal import blackman
from scipy.fftpack import fft
from collections import deque
import numpy as np
import math
import pylab
import time
import pdb
import itertools
import theano
import theano.tensor as T
import resource


def pre_compute_kernels(sample_freq, frequency_range, default_width, output="kernel"):
    '''
    The kernels are computed before the CQT calculation which 
    leaves us with just a multiplication of the kernel with signal over the
    t -> t+ width duration. The width is the maximum of the given width and the one
    computed.

    Parameters
    ----------
    sample_freq
        The sampling frequency.
    frequency_range:
        The Frequency range over which CQT is computed
    default_width
        Taken as 1/3rd of the Sampling Frequency by convention.
    output
        Requires what output has to be. Could be either kernel or windows Depending on where the
        exponential is being computed

    '''

    if output is not "windows" and output is not "kernel":
        raise TypeError("The output has to be either windows / kernel")

    kernel = []
    computed_window = []
    comp_width = []
    comp_width.append(int(default_width))
    window_list = []
    # Here, the width and window array is constructed
    # From the maximum width that we return, we can compute the 
    # i_exp variable outside the for loop, which makes a the O(N**2) algorithm, O(N)

    for f in frequency_range:
        current_width = int(1/(.03*f)*sample_freq)
        computed_window.append(blackman(current_width))
        comp_width.append(int(current_width))

    max_width = max(comp_width)

    # The Kernels are computed after padding the windows.
    # All the windows aren't the same and for multiplication to be performed later,
    # The windows are padded.
    for f_count, freq in enumerate(frequency_range):
        if max_width > len(computed_window[f_count]):
            if (max_width - len(computed_window[f_count])) %2 == 0 :
                windows = np.pad(computed_window[f_count], (max_width - len(computed_window[f_count])) / 2, 'constant')

            else:
                windows = np.append(np.pad(computed_window[f_count], (max_width - len(computed_window[f_count]) - 1) / 2, 'constant'), 0)

        else:
            windows = computed_window[f_count]

        i_exp = np.arange(0, len(windows))
        window_list.append(windows)
        kernel.append(np.exp( i_exp * 2j * math.pi * freq / sample_freq) * windows)

    if output == "windows":
        return window_list, max(comp_width)
    return kernel, max(comp_width)

    
def threeloop_cqt(signal, sample_freq):
    '''
    Computes the CQT of a signal using three loops.
    This is the most primitive version of the algorithm.

    Parameters
    ----------
    signal:
        The signal whose CQT has to be determined.
    sample_freq:
        The sampling frequency of the signal
    '''
    max_width = 1/(.03*100)*sample_freq
    frequency_range = 100.*2.**(1./12.*np.array(np.arange(0, 50)))
    time_range = range(0, int(len(signal) - max_width), int(max_width/2))
    output = np.zeros((len(time_range), len(frequency_range)), dtype=np.complex)
    prior = resource.getrusage(resource.RUSAGE_SELF)
    for m, t in enumerate(time_range):
        for n, f in enumerate(frequency_range):
            current_width = int(1/(.03*f)*sample_freq)
            window = blackman(current_width)
            for i in range(t, t+int(current_width)):
                output[m, n] += np.exp(i * 1j * (-2 * math.pi) * f / sample_freq) * (signal[i] * window[i-t])
    post = resource.getrusage(resource.RUSAGE_SELF)
    print("System time: " + str(post.ru_stime - prior.ru_stime))
    print("User time: " + str(post.ru_utime - prior.ru_utime))
    return output, frequency_range, time_range

def cqt_two(signal, s_freq):
    '''
    Computes the CQT of the signal using two loops without the precomputation of the kernels
    '''
    max_width = 1/(.03*100)*s_freq
    freq = 100.*2.**(1./12.*np.array(range(0, 50)))
    time = range(0, int(len(signal) - max_width), int(max_width/2))
    output = np.zeros((len(time), len(freq)), dtype=np.complex)
    prior = resource.getrusage(resource.RUSAGE_SELF)
    for l_1, t in enumerate(time):
        for l_2, f in enumerate(freq):
            temp_w = int(1/(.03*f)*s_freq)
            windows = blackman(temp_w)
            w_i = np.arange(t, t+temp_w)
            n_signal = signal[t:t+temp_w]
            output[l_1, l_2] = np.sum((np.exp(w_i * 1j * (-2 * math.pi) * f / s_freq) * windows) * n_signal)
    
    post = resource.getrusage(resource.RUSAGE_SELF)
    print("System time: " + str(post.ru_stime - prior.ru_stime))
    print("User time: " + str(post.ru_utime - prior.ru_utime))
 
    return output, freq, time


def stft_twoloops(signal, s_freq):
    '''
    Computes the CQT of the signal using two loops with pre-computation of the kernels
    '''
    signal_width = 1/(.03*100)*s_freq
    freq = 100.*2.**(1./12.*np.array(range(0, 50)))
    time_list = range(0, int(len(signal) - signal_width), int(signal_width/2))
    windows, max_width= pre_compute_kernels(s_freq, freq, signal_width, "windows")
    output = np.zeros((len(time_list), len(freq)))
    i_exp = np.arange(0, max_width)
    prior = resource.getrusage(resource.RUSAGE_SELF)
    for l_1, t in enumerate(time_list):
        # import pdb; pdb.set_trace()
        for l_2, f in enumerate(freq):
            n_signal = signal[t:t+max_width]
            output[l_1, l_2] = np.sum(np.exp(i_exp * 2j * math.pi * f / s_freq) * windows[l_2] * n_signal)
    
    post = resource.getrusage(resource.RUSAGE_SELF)
    print("System time: " + str(post.ru_stime - prior.ru_stime))
    print("User time: " + str(post.ru_utime - prior.ru_utime))
    return output, freq, time_list
        
# windows[l_2].extend([0] * (len() - (len(my_list))))


def cqt_single(signal, sample_freq):
    ''' Compute the CQT of a signal with a single loop
    '''
    max_width = 1/(.03*100)*sample_freq
    frequency_range = 100.*2.**(1./12.*np.array(range(0, 50)))
    time_list = range(0, int(len(signal) - max_width), int(max_width/2))
    output = np.zeros((len(time_list), len(frequency_range)))
    kernel, max_width = pre_compute_kernels(sample_freq, frequency_range, max_width, "kernel")
    prior = resource.getrusage(resource.RUSAGE_SELF)
    n_sig = np.ones((len(time_list) , max_width))
    for l_1, t in enumerate(time_list):
        output[l_1] = np.sum(signal[l_1:l_1 + max_width] * kernel, axis=1)
    # output = n_sig.dot(np.asarray(kernel).transpose())
    post = resource.getrusage(resource.RUSAGE_SELF)
    print("System time: " + str(post.ru_stime - prior.ru_stime))
    print("User time: " + str(post.ru_utime - prior.ru_utime))

    return output, frequency_range, time_list


def theano_stft(signal, width, s_freq):
    max_width = 1/(.03*100)*s_freq
    freq = 100.*2.**(1./12.*np.array(range(0, 50)))
    time = range(0, int(len(signal) - max_width), int(max_width/2))

    width_var = T.cscalar() #used as i in the numpy implementation
    freq_var = T.fscalar() #used as f 
    time_var = T.fscalar() #used as t
    signal_var = T.fscalar() # Components of Signal
    stft_out = T.exp(width_var * freq_var) * signal_var
    func = theano.function([width_var, freq_var, signal_var], stft_out, allow_input_downcast=True)

    output = np.zeros((len(freq), len(time)), dtype=np.complex)
    for l_1, t in enumerate(time):
        for l_2, f in enumerate(freq):
            width = 1/(.03*f)*s_freq
            for i in range(t, t+int(width)):

                output[l_2, l_1] += func(i * -6.28j, f/s_freq, signal[i]).tolist()  

    return output, freq, time
 

def vectorized_theano(signal, width, s_freq):

    '''
    The Code is currently being tested. It's not fully implemented.
    It's being tested with memory and speed.
    '''

    max_width = 1/(.03*100)*s_freq
    freq = 100.*2.**(1./12.*np.array(range(0, 50)))
    time_list = range(0, int(len(signal) - max_width), int(max_width/2))

    signal_matrix = T.fvector("signal_matrix")
    kernel_matrix = T.fmatrix()
    cqt_compute = kernel_matrix * signal_matrix
    kernel, k_size = pre_compute_kernels(s_freq, freq, width)
    func = theano.function([kernel_matrix, signal_matrix], cqt_compute, allow_input_downcast=True)
    print("Computing theano time")
    time0 = time.time()
    n_signal = np.zeros((len(time_list), k_size))
    for l_1, t in enumerate(time_list):
        n_signal[l_1] = signal[l_1:l_1 + k_size]
    result, updates = theano.scan(fn=lambda n_signal, kernel: np.sum(n_signal * kernel),
                                  non_sequences=kernel, n_steps=len(n_signal))
    n_s = T.fmatrix()
    n_k = T.fmatrix()
    
    compute_cqt = theano.function([n_s, n_k], result, updates=updates)
    compute_cqt(n_signal, kernel)
    time1 = time.time()
    print(time1 - time0)
    return output, freq, time_list

noise = np.random.randn(132300)
pure_signal = np.cos(2 * math.pi * 440./44100. * np.arange(132300))
final_signal = pure_signal + noise
# out = threeloop_cqt(signal, 44100)
ground_out = cqt_single(final_signal, 44100)
pylab.imshow(np.log(np.abs(ground_out[0])), origin='lower', aspect='auto')
pylab.show()

