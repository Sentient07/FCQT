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
import matplotlib.pyplot as plt


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
    kernel = np.empty(int(max_width))
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
        kernel = np.vstack((kernel, np.exp( i_exp * 2j * math.pi * freq / sample_freq) * windows))
    kernel = np.delete(kernel, (0), axis=0)
    if output == "windows":
        return window_list, max(comp_width)
    return kernel, max(comp_width)


def threeloop_cqt(signal, max_width, time_range, frequency_range, kernel_details=None, channels=1):
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
    output = np.zeros((len(time_range), len(frequency_range)), dtype=np.complex)
    prior = resource.getrusage(resource.RUSAGE_SELF)
    for m, t in enumerate(time_range):
        for n, f in enumerate(frequency_range):
            current_width = int(1/(.03*f)*sample_freq)
            window = blackman(current_width)
            for i in range(t, t+int(current_width)):
                output[m, n] += np.exp(i * 1j * (-2 * math.pi) * f / sample_freq) * (signal[i] * window[i-t])
    post = resource.getrusage(resource.RUSAGE_SELF)
    user_time = post.ru_utime - prior.ru_utime
    print("System time for three loop computation : " + str(post.ru_stime - prior.ru_stime))
    print("User time for three loop computation : " + str(post.ru_utime - prior.ru_utime))
    return output, user_time

def cqt_two_with_kernel(signal, max_width, time_range, frequency_range, kernel_details=None, channels=1):
    '''
    Computes the CQT of the signal using two loops without the precomputation of the kernels
    '''
    kernel, max_width = kernel_details
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
    user_time = post.ru_utime - prior.ru_utime
    print("System time for two loop computation : " + str(post.ru_stime - prior.ru_stime))
    print("User time for two loop computation : " + str(post.ru_utime - prior.ru_utime))
 
    return output, user_time

def cqt_two_without_kernel(signal, max_width, time_range, frequency_range, kernel_details=None, channels=1):
    '''
    Computes the CQT of the signal using two loops with pre-computation of the kernels
    '''
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
    user_time = post.ru_utime - prior.ru_utime
    print("System time for two loop with pre computed kernel: " + str(post.ru_stime - prior.ru_stime))
    print("User time for two loop with pre computed kernel: " + str(post.ru_utime - prior.ru_utime))
    return output, user_time
        
# windows[l_2].extend([0] * (len() - (len(my_list))))

def cqt_single(signal, time_range, frequency_range, kernel_details, channels=1):
    ''' 
    Compute the CQT of a signal with a single loop
    '''
    output = np.zeros((len(time_range), len(frequency_range)))
    prior = resource.getrusage(resource.RUSAGE_SELF)
    kernel, max_width = kernel_details
    n_sig = np.ones((len(time_range) , max_width))
    for l_1, t in enumerate(time_range):
        output[l_1] = np.sum(signal[t:t + max_width] * kernel, axis=1)
    # output = n_sig.dot(np.asarray(kernel).transpose())
    post = resource.getrusage(resource.RUSAGE_SELF)
    user_time = post.ru_utime - prior.ru_utime
    print("System time for single loop computation: " + str(post.ru_stime - prior.ru_stime))
    print("User time for single loop computation: " + str(post.ru_utime - prior.ru_utime))

    return output, user_time

def matrix_cqt(signal, time_range, frequency_range, kernel_details, channels=1):
    '''
    Direct computation of CQT using numpy.
    The signal matrix is constructed by advanced indexing of the
    signal vector. The CQT is the dot product of signal with the transpose of the kernel
    matrix.
    '''
    kernel, max_width = kernel_details
    output = np.zeros((len(time_range), len(frequency_range)))
    h = int(max_width/2)
    n = time_range[-1] + h
    col_index = np.arange(0, n, h)
    row_index = np.arange(max_width)
    index = col_index[:,np.newaxis] + row_index[np.newaxis, :]
    prior = resource.getrusage(resource.RUSAGE_SELF)
    sliced_signal = signal[index]
    output = sliced_signal.dot(np.asarray(kernel).transpose())
    post = resource.getrusage(resource.RUSAGE_SELF)
    user_time = post.ru_utime - prior.ru_utime
    print("System time for direct computation : " + str(post.ru_stime - prior.ru_stime))
    print("User time for direct computation : " + str(post.ru_utime - prior.ru_utime))
    return output, user_time
# Computation of CQT using theano.


def vectorized_theano(signal, time_range, frequency_range, kernel_details, channels=1):
    '''
    The Code is currently being tested. It's not fully implemented.
    It's being tested with memory and speed.
    '''
    kernel, max_width = kernel_details
    signal_matrix = T.fvector("signal_matrix")
    kernel_matrix = T.fmatrix("kernel matrix")
    output = np.zeros((len(time_range), len(frequency_range)))
    cqt_compute = kernel_matrix * signal_matrix
    func = theano.function([kernel_matrix, signal_matrix], cqt_compute, allow_input_downcast=True)
    prior = resource.getrusage(resource.RUSAGE_SELF)
    for l_1, t in enumerate(time_range):
        n_signal = signal[l_1:l_1 + max_width]
        output[l_1] = np.sum(func(kernel, n_signal), axis=1)
    post = resource.getrusage(resource.RUSAGE_SELF)
    user_time = post.ru_utime - prior.ru_utime
    print("System time for vectorized theano : " + str(post.ru_stime - prior.ru_stime))
    print("User time for vectorized theano: " + str(post.ru_utime - prior.ru_utime))
    return output, user_time

def matrix_theano(signal, time_range, frequency_range, kernel_details, channels=1):
    '''
    Direct computation of CQT without any loop using theano
    '''
    kernel, max_width = kernel_details
    signal_matrix = T.fmatrix("Signal Matrix")
    kernel_matrix = T.fmatrix("Kernel Matrix")
    h = int(max_width/2)
    import pdb;pdb.set_trace()
    n = time_range[-1] + h
    col_index = np.arange(0, n, h)
    row_index = np.arange(max_width)
    index = col_index[:,np.newaxis] + row_index[np.newaxis, :]
    cqt_compute = T.dot(signal_matrix, kernel_matrix.T)
    func = theano.function([signal_matrix, kernel_matrix], cqt_compute, allow_input_downcast=True)
    prior = resource.getrusage(resource.RUSAGE_SELF)
    sliced_signal = signal[index]
    output = func(sliced_signal, kernel)
    post = resource.getrusage(resource.RUSAGE_SELF)
    user_time = post.ru_utime - prior.ru_utime
    print("System time for matrix theano: " + str(post.ru_stime - prior.ru_stime))
    print("User time for matrix theano: " + str(post.ru_utime - prior.ru_utime))
    return output, user_time

def generate_signal(signal_base, mul_factor=1):
    pure_signal = np.cos(2 * math.pi * 440./44100. * np.arange(signal_base * mul_factor))
    noise = np.random.randn(signal_base * mul_factor)
    final_signal = pure_signal + noise
    return pure_signal


def plot_graph(cqt_values, title, mode, labels, axes_label):
    fig, ax = plt.subplots()
    plt.title(title)
    colour_list = ['black', 'green', 'red', 'blue']
    colour_labelmap = ["Black - Single Looped Numpy", "Green - Single Loop Theano",
                       "Red - Matrix Multiplication Theano", "Blue - Matrix Multiplication, Numpy"]
    plot_array = []
    for index, cv in enumerate(cqt_values):
        temp_plots = ax.plot(labels, cv, colour_list[index], marker='o')
        plot_array.append(temp_plots)
    plt.legend(tuple([i[0] for i in plot_array]), tuple(colour_labelmap), loc=0)
    ax.ticklabel_format(useOffset=False)
    ax.set_xlabel(axes_label[0])
    ax.set_ylabel(axes_label[1])
    plt.ylim()
    plt.show()

def usertime_graph(signal_base, mul_factor, final_signal, time_range, frequency_range, kernel_details, max_width):
    mul_count = 0
    three_loop, two_kernel, two_w_kernel = [], [], []
    single_cqt, theano_vectorized, theano_matrix = [], [], []
    direct_cqt = []
    signal_length = []
    while  True:
        if mul_count >= 8:
            break
        final_signal = generate_signal(signal_base, mul_factor)
        time_range = np.arange(0, len(final_signal) - int(max_width), int(max_width/2))
        single_cqt.append(cqt_single(final_signal, time_range, frequency_range, kernel_details)[1])
        theano_vectorized.append(vectorized_theano(final_signal, time_range, frequency_range, kernel_details)[1])
        theano_matrix.append(matrix_theano(final_signal, time_range, frequency_range, kernel_details)[1])
        direct_cqt.append(matrix_cqt(final_signal, time_range, frequency_range, kernel_details)[1])
        signal_length.append(signal_base)
        signal_base = signal_base * mul_factor
        mul_count += 1
    cqt_values = [single_cqt, theano_vectorized, theano_matrix, direct_cqt]
    plt_title = "User time vs Signal Length Graph"
    axes_label = ["Signal Length", "Time taken for execution (in sec)"]
    plot_graph(cqt_values, plt_title, "user-time", signal_length, axes_label)

    # plt.xticks(np.arange(signal_length[0], signal_length[-1] , signal_length[1] - signal_length[0]))

def channel_graph(final_signal, channels, time_range, sample_freq, max_width):
    three_loop, two_kernel, two_w_kernel = [], [], []
    single_cqt, theano_vectorized, theano_matrix = [], [], []
    direct_cqt = []
    cqt_values = []
    for i in channels:
        # three_loop.append(threeloop_cqt(final_signal, 44100, i)[1])
        # two_kernel.append(cqt_two_with_kernel(final_signal, 44100, i)[1])
        # two_w_kernel.append(cqt_two_without_kernel(final_signal, 44100, i)[1])
        frequency_range = 100. * 2. ** (1./(12. * i) * np.arange(0, 50, (1.0/i)))
        kernel_details = pre_compute_kernels(sample_freq, frequency_range, max_width)
        single_cqt.append(cqt_single(final_signal, time_range, frequency_range, kernel_details, channels=i)[1])
        theano_vectorized.append(vectorized_theano(final_signal, time_range, frequency_range, kernel_details, channels=i)[1])
        theano_matrix.append(matrix_theano(final_signal, time_range, frequency_range, kernel_details, channels=i)[1])
        direct_cqt.append(matrix_cqt(final_signal, time_range, frequency_range, kernel_details, channels=i)[1])
    cqt_values = [single_cqt, theano_vectorized, theano_matrix, direct_cqt]
    plt_title = "Variable channel vs Time Graph"
    axes_label = ["Number of Channels", "Time taken for execution (in sec)"]
    plot_graph(cqt_values, plt_title, "channel", channels, axes_label)

if __name__ == '__main__':
    signal_base = 132300
    mul_factor = 2
    sample_freq = 44100
    max_width = 1/(.03*100)*sample_freq
    frequency_range = 100.*2.**(1./(12.)*np.arange(0, 50))
    final_signal = generate_signal(signal_base)
    time_range = np.arange(0, len(final_signal) - int(max_width), int(max_width/2))
    kernel_details = pre_compute_kernels(sample_freq, frequency_range, max_width)
    channels = np.arange(1, 5)
    channel_graph(final_signal, channels, time_range, sample_freq, max_width)
    #usertime_graph(signal_base, mul_factor, final_signal, time_range, frequency_range, kernel_details, max_width)
    # matrix = matrix_cqt(final_signal, 44100)
    # plot_matrix = matrix[0].transpose()
    '''
    pylab.imshow(np.log(np.abs(plot_matrix)), origin='lower', aspect='auto')
    pylab.xlabel('Time (sec)')
    pylab.ylabel('Frequency in Log scale')
    pylab.show()
    '''