# STFT
from scipy.signal import blackman
import numpy as np
import math
import time
import theano
import theano.tensor as T
import resource
import matplotlib.pyplot as plt

mode_with_gpu = theano.compile.mode.get_mode(
    'FAST_RUN').including('gpuarray').excluding('gpu')


def pre_compute_kernels(sample_freq, frequency_range,
                        default_width, output="kernel"):
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
    # i_exp variable outside the for loop, which makes a the O(N**2)
    # algorithm, O(N)

    for f in frequency_range:
        current_width = int(1 / (.03 * f) * sample_freq)
        computed_window.append(blackman(current_width))
        comp_width.append(int(current_width))

    max_width = max(comp_width)
    kernel = np.empty(int(max_width))
    # The Kernels are computed after padding the windows.
    # All the windows aren't the same and for multiplication to be performed later,
    # The windows are padded.
    for f_count, freq in enumerate(frequency_range):
        if max_width > len(computed_window[f_count]):
            if (max_width - len(computed_window[f_count])) % 2 == 0:
                windows = np.pad(computed_window[
                                 f_count], (max_width - len(computed_window[f_count])) / 2, 'constant')

            else:
                windows = np.append(np.pad(computed_window[
                                    f_count], (max_width - len(computed_window[f_count]) - 1) / 2, 'constant'), 0)

        else:
            windows = computed_window[f_count]

        i_exp = np.arange(0, len(windows))
        window_list.append(windows)
        kernel = np.vstack(
            (kernel,
             np.exp(
                 i_exp *
                 2j *
                 math.pi *
                 freq /
                 sample_freq) *
                windows))
    kernel = np.delete(kernel, (0), axis=0)
    if output == "windows":
        return window_list, max(comp_width)
    return kernel, max(comp_width)


def pre_compute_algebraic_kernel(sample_freq, frequency_range, default_width):
    '''
    Pre computes the kernel after applying the transforming the exponential term to cosine and sine terms.
    e^{\frac {-j2\pi f t}{s_f}}} = {cos(\frac {-2\pi f t}{s_f}) + jsin(\frac {-2\pi f t}{s_f})}

    Parameters
    ----------
    sample_freq
        The sampling frequency.
    frequency_range:
        The Frequency range over which CQT is computed
    default_width
        Taken as 1/3rd of the Sampling Frequency by convention.
    '''
    computed_window = []
    comp_width = []
    comp_width.append(int(default_width))
    # Here, the width and window array is constructed
    # From the maximum width that we return, we can compute the
    # i_exp variable outside the for loop, which makes a the O(N**2)
    # algorithm, O(N)

    for f in frequency_range:
        current_width = int(1 / (.03 * f) * sample_freq)
        computed_window.append(blackman(current_width))
        comp_width.append(int(current_width))

    max_width = max(comp_width)
    sine_terms = np.empty(int(max_width))
    cosine_terms = np.empty(int(max_width))
    window_list = np.empty(int(max_width))

    for f_count, freq in enumerate(frequency_range):
        if max_width > len(computed_window[f_count]):
            if (max_width - len(computed_window[f_count])) % 2 == 0:
                windows = np.pad(computed_window[
                                 f_count], (max_width - len(computed_window[f_count])) / 2, 'constant')
            else:
                windows = np.append(np.pad(computed_window[
                                    f_count], (max_width - len(computed_window[f_count]) - 1) / 2, 'constant'), 0)
        else:
            windows = computed_window[f_count]
        window_list = np.vstack((window_list, windows))
        i_exp = np.arange(0, len(windows))
        sine_terms = np.vstack((sine_terms, np.sin(2 * np.pi * freq * i_exp)))
        cosine_terms = np.vstack(
            (cosine_terms, np.cos(2 * np.pi * freq * i_exp)))
    half_length = int(max_width / 2)
    half_windows = window_list.transpose()[0:half_length].transpose()
    half_cosine = cosine_terms.transpose()[0:half_length].transpose()
    half_sine = sine_terms.transpose()[0:half_length].transpose()
    cosine_kernel = half_windows * half_cosine
    sin_kernel = half_windows * half_sine
    # Deleting because we have initialzed an empty array with first element as
    # 0
    cosine_kernel = np.delete(cosine_kernel, (0), axis=0)
    sin_kernel = np.delete(sin_kernel, (0), axis=0)
    combined_kernel = cosine_kernel + 1j * sin_kernel
    return cosine_kernel, sin_kernel, max_width, combined_kernel


def threeloop_cqt(signal, max_width, time_range,
                  frequency_range, s_freq, kernel_details=None):
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
    output = np.zeros(
        (len(time_range),
         len(frequency_range)),
        dtype=np.complex)
    prior = resource.getrusage(resource.RUSAGE_SELF)
    for m, t in enumerate(time_range):
        for n, f in enumerate(frequency_range):
            current_width = int(1 / (.03 * f) * sample_freq)
            window = blackman(current_width)
            for i in range(t, t + int(current_width)):
                output[m, n] += np.exp(i * 1j * (-2 * math.pi)
                                       * f / sample_freq) * (signal[i] * window[i - t])
    post = resource.getrusage(resource.RUSAGE_SELF)
    user_time = post.ru_utime - prior.ru_utime
    print("System time for three loop computation : " +
          str(post.ru_stime - prior.ru_stime))
    print("User time for three loop computation : " +
          str(post.ru_utime - prior.ru_utime))
    return output, user_time


def cqt_two_with_kernel(signal, max_width, time_range,
                        frequency_range, s_freq, kernel_details=None):
    '''
    Computes the CQT of the signal using two loops without the precomputation of the kernels
    '''
    kernel, max_width = kernel_details
    output = np.zeros((len(time), len(frequency_range)), dtype=np.complex)
    prior = resource.getrusage(resource.RUSAGE_SELF)
    for l_1, t in enumerate(time):
        for l_2, f in enumerate(frequency_range):
            temp_w = int(1 / (.03 * f) * s_freq)
            windows = blackman(temp_w)
            w_i = np.arange(t, t + temp_w)
            n_signal = signal[t:t + temp_w]
            output[l_1, l_2] = np.sum(
                (np.exp(w_i * 1j * (-2 * math.pi) * f / s_freq) * windows) * n_signal)

    post = resource.getrusage(resource.RUSAGE_SELF)
    user_time = post.ru_utime - prior.ru_utime
    print("System time for two loop computation : " +
          str(post.ru_stime - prior.ru_stime))
    print("User time for two loop computation : " +
          str(post.ru_utime - prior.ru_utime))

    return output, user_time


def cqt_two_without_kernel(
        signal, max_width, time_range, frequency_range, s_freq, kernel_details=None):
    '''
    Computes the CQT of the signal using two loops with pre-computation of the kernels
    '''
    windows, max_width = pre_compute_kernels(
        s_freq, frequency_range, max_width, "windows")
    output = np.zeros((len(time_range), len(frequency_range)))
    i_exp = np.arange(0, max_width)
    prior = resource.getrusage(resource.RUSAGE_SELF)
    for l_1, t in enumerate(time_range):
        # import pdb; pdb.set_trace()
        for l_2, f in enumerate(frequency_range):
            n_signal = signal[t:t + max_width]
            output[
                l_1,
                l_2] = np.sum(
                np.exp(
                    i_exp *
                    2j *
                    math.pi *
                    f /
                    s_freq) *
                windows[l_2] *
                n_signal)

    post = resource.getrusage(resource.RUSAGE_SELF)
    user_time = post.ru_utime - prior.ru_utime
    print("System time for two loop with pre computed kernel: " +
          str(post.ru_stime - prior.ru_stime))
    print("User time for two loop with pre computed kernel: " +
          str(post.ru_utime - prior.ru_utime))
    return output, user_time

# windows[l_2].extend([0] * (len() - (len(my_list))))


def vectorized_numpy(signal, time_range, frequency_range, kernel_details):
    '''
    Compute the CQT of a signal with a single loop
    '''
    output = np.zeros((len(time_range), len(frequency_range)))
    prior = resource.getrusage(resource.RUSAGE_SELF)
    kernel, max_width = kernel_details
    for l_1, t in enumerate(time_range):
        output[l_1] = np.sum(signal[t:t + max_width] * kernel, axis=1)
    # output = n_sig.dot(np.asarray(kernel).transpose())
    post = resource.getrusage(resource.RUSAGE_SELF)
    user_time = post.ru_utime - prior.ru_utime
    print("System time for single loop computation: " +
          str(post.ru_stime - prior.ru_stime))
    print("User time for single loop computation: " +
          str(post.ru_utime - prior.ru_utime))

    return output, user_time


def full_vectorized_numpy(signal, time_range, frequency_range, kernel_details):
    '''
    Direct computation of CQT using numpy.
    The signal matrix is constructed by advanced indexing of the
    signal vector. The CQT is the dot product of signal with the transpose of the kernel
    matrix.
    '''
    kernel, max_width = kernel_details
    output = np.zeros((len(time_range), len(frequency_range)))
    h = int(max_width / 2)
    n = time_range[-1] + h
    col_index = np.arange(0, n, h)
    row_index = np.arange(max_width)
    index = col_index[:, np.newaxis] + row_index[np.newaxis, :]
    prior = resource.getrusage(resource.RUSAGE_SELF)
    sliced_signal = signal[index]
    output = sliced_signal.dot(np.asarray(kernel).transpose())
    post = resource.getrusage(resource.RUSAGE_SELF)
    user_time = post.ru_utime - prior.ru_utime
    print("System time for direct computation : " +
          str(post.ru_stime - prior.ru_stime))
    print("User time for direct computation : " +
          str(post.ru_utime - prior.ru_utime))
    return output, user_time
# Computation of CQT using theano.


def vectorized_theano(signal, time_range, frequency_range, kernel_details):
    '''
    The Code is currently being tested. It's not fully implemented.
    It's being tested with memory and speed.
    '''
    kernel, max_width = kernel_details
    signal_matrix = T.fvector("signal_matrix")
    kernel_matrix = T.fmatrix("kernel matrix")
    output = np.zeros((len(time_range), len(frequency_range)))
    cqt_compute = kernel_matrix * signal_matrix
    func = theano.function([kernel_matrix, signal_matrix],
                           cqt_compute, allow_input_downcast=True)
    prior = resource.getrusage(resource.RUSAGE_SELF)
    for l_1, t in enumerate(time_range):
        n_signal = signal[l_1:l_1 + max_width]
        output[l_1] = np.sum(func(kernel, n_signal), axis=1)
    post = resource.getrusage(resource.RUSAGE_SELF)
    user_time = post.ru_utime - prior.ru_utime
    print("System time for vectorized theano : " +
          str(post.ru_stime - prior.ru_stime))
    print("User time for vectorized theano: " +
          str(post.ru_utime - prior.ru_utime))
    return output, user_time


def full_vectorized_theano(
        signal, time_range, frequency_range, kernel_details):
    '''
    Direct computation of CQT without any loop using theano
    '''
    kernel, max_width = kernel_details
    signal_matrix = T.fmatrix("Signal Matrix")
    kernel_matrix = T.fmatrix("Kernel Matrix")
    h = int(max_width / 2)
    n = time_range[-1] + h
    col_index = np.arange(0, n, h)
    row_index = np.arange(max_width)
    index = col_index[:, np.newaxis] + row_index[np.newaxis, :]
    cqt_compute = T.dot(signal_matrix, kernel_matrix.T)
    func = theano.function([signal_matrix, kernel_matrix],
                           cqt_compute, allow_input_downcast=True)
    prior = resource.getrusage(resource.RUSAGE_SELF)
    sliced_signal = signal[index]
    output = func(sliced_signal, kernel)
    post = resource.getrusage(resource.RUSAGE_SELF)
    print(func.maker.fgraph.toposort())
    user_time = post.ru_utime - prior.ru_utime
    print("System time for Fully Vectorized theano: " +
          str(post.ru_stime - prior.ru_stime))
    print("User time for Fully Vectorized theano: " +
          str(post.ru_utime - prior.ru_utime))
    return output, user_time


def symmetric_numpy(signal, time_range, frequency_range, kernel_details):
    '''
    Computes CQT using the kernel's cosine and sine components separately.
    The cosine and sine components are the result of Euler transformation of
    exponential term in the pre computation of the kernel
    '''
    cosine_kern, sine_kern, max_width, combined_kernel = kernel_details
    output = np.zeros((len(time_range), len(frequency_range)))
    h = int(max_width / 2)
    n = time_range[-1] + h
    col_index = np.arange(0, n, h)
    row_index = np.arange(max_width)
    index = col_index[:, np.newaxis] + row_index[np.newaxis, :]
    prior = resource.getrusage(resource.RUSAGE_SELF)
    sliced_former = signal[index][:, 0:h]
    sliced_latter = signal[index][:, h:max_width]
    cosine_signal = sliced_former + sliced_latter
    sine_signal = sliced_former - sliced_latter
    cosine_comp = cosine_signal.dot(cosine_kern.transpose())
    sine_comp = sine_signal.dot(sine_kern.transpose())
    output = np.sqrt(cosine_comp ** 2 + sine_comp ** 2)
    post = resource.getrusage(resource.RUSAGE_SELF)
    user_time = post.ru_utime - prior.ru_utime
    print("System time for Numpy's symmetric computation : " +
          str(post.ru_stime - prior.ru_stime))
    print("User time for Numpy's Symmetric computation : " +
          str(post.ru_utime - prior.ru_utime))
    return output, user_time


def symmetric_theano(signal, time_range, frequency_range, kernel_details):
    '''
    Same as euler_computation, but the computation is done using theano
    '''
    cosine_kern, sine_kern, max_width, combined_kernel = kernel_details
    output = np.zeros((len(time_range), len(frequency_range)))
    h = int(max_width / 2)
    n = time_range[-1] + h
    col_index = np.arange(0, n, h)
    row_index = np.arange(max_width)
    index = col_index[:, np.newaxis] + row_index[np.newaxis, :]
    cosine_signal = T.fmatrix("Cosine Signal")
    sine_signal = T.fmatrix("Sine Signal")
    cosine_matrix = T.fmatrix("Cosine Matrix")
    sine_matrix = T.fmatrix("Sine Matrix")
    cqt_compute = T.dot(cosine_signal, cosine_matrix.T) + \
        1j * T.dot(sine_signal, sine_matrix.T)
    func = theano.function([cosine_signal,
                            sine_signal,
                            cosine_matrix,
                            sine_matrix],
                           cqt_compute,
                           allow_input_downcast=True)
    prior = resource.getrusage(resource.RUSAGE_SELF)
    sliced_former = signal[index][:, 0:h]
    sliced_latter = signal[index][:, h:max_width]
    cosine_comp = sliced_former + sliced_latter
    sine_comp = sliced_former - sliced_latter
    output = func(cosine_comp, sine_comp, cosine_kern, sine_kern)
    post = resource.getrusage(resource.RUSAGE_SELF)
    print(func.maker.fgraph.toposort())
    user_time = post.ru_utime - prior.ru_utime
    print("System time for Theano's Symmetric computation : " +
          str(post.ru_stime - prior.ru_stime))
    print("User time for Theano's Symmetric computation : " +
          str(post.ru_utime - prior.ru_utime))
    return output, user_time


def generate_signal(signal_base, mul_factor=1):
    '''
    Generates signal of a given length after adding Gaussian white noise of the same length.

    Parameters
    ----------
    signal_base
        The length of the signal.
    mul_factor
        The multiplication factor
    '''

    pure_signal = np.cos(
        2 *
        math.pi *
        440. /
        44100. *
        np.arange(
            signal_base *
            mul_factor))
    noise = np.random.randn(signal_base * mul_factor)
    final_signal = pure_signal + noise
    return final_signal


def plot_graph(cqt_values, title, labels, axes_label):
    '''
    Plots graph with respect to time.
    Parameters
    ----------
    cqt_values
        list containing time taken by different methods that
        computes CQT.

    title
        The title of the plot.

    labels
        list containing the Y coordinate details.

    axes_label
        The labels for X and Y axes.

    '''
    fig, ax = plt.subplots()
    plt.title(title)
    colour_list = ['black', 'green', 'red', 'blue', 'yellow', 'purple']
    colour_labelmap = ["Black - Vectorized Numpy", "Green - Vectorized Theano",
                       "Red - Fully vectorized Theano", "Blue - Fully Vectorized, Numpy",
                       "Yellow - Symmetric Numpy", "Purple - Symmetric Theano"]
    plot_array = []
    for index, cv in enumerate(cqt_values):
        temp_plots = ax.plot(labels, cv, colour_list[index], marker='o')
        plot_array.append(temp_plots)
    plt.legend(tuple([i[0] for i in plot_array]),
               tuple(colour_labelmap), loc=0)
    ax.ticklabel_format(useOffset=False)
    ax.set_xlabel(axes_label[0])
    ax.set_ylabel(axes_label[1])
    plt.ylim()
    plt.show()


def usertime_graph(signal_base, mul_factor, time_range,
                   frequency_range, kernel_details, max_width):
    '''
    Signal length vs Time taken for computation of CQT graph
    '''

    mul_count = 0

    numpy_vectorized, theano_vectorized, theano_fully_vectorized = [], [], []
    numpy_fully_vectorized, cqt_values, numpy_symmetric, theano_symmetric = [], [], [], []

    signal_length = []
    while True:
        if mul_count >= 5:
            break
        final_signal = generate_signal(signal_base, mul_factor)
        time_range = np.arange(0, len(final_signal) -
                               int(max_width), int(max_width / 2))
        print("CQT is computed for Signal Length of : " +
              str(int((signal_base * mul_factor) / 44100)))
        numpy_vectorized.append(
            vectorized_numpy(
                final_signal,
                time_range,
                frequency_range,
                kernel_details[0])[1])
        theano_vectorized.append(
            vectorized_theano(
                final_signal,
                time_range,
                frequency_range,
                kernel_details[0])[1])
        theano_fully_vectorized.append(
            full_vectorized_theano(
                final_signal,
                time_range,
                frequency_range,
                kernel_details[0])[1])
        numpy_fully_vectorized.append(
            full_vectorized_numpy(
                final_signal,
                time_range,
                frequency_range,
                kernel_details[0])[1])
        numpy_symmetric.append(
            symmetric_numpy(
                final_signal,
                time_range,
                frequency_range,
                kernel_details[1])[1])
        theano_symmetric.append(
            symmetric_theano(
                final_signal,
                time_range,
                frequency_range,
                kernel_details[1])[1])
        signal_length.append(signal_base)
        signal_base = signal_base * mul_factor
        mul_count += 1
    cqt_values = [numpy_vectorized, theano_vectorized, theano_fully_vectorized, numpy_fully_vectorized,
                  numpy_symmetric, theano_symmetric]
    print(cqt_values[2])
    print(cqt_values[-1])
    plt_title = "User time vs Signal Length Graph"
    axes_label = ["Signal Length", "Time taken for execution (in sec)"]
    plot_graph(cqt_values, plt_title, signal_length, axes_label)

    # plt.xticks(np.arange(signal_length[0], signal_length[-1] , signal_length[1] - signal_length[0]))


def channel_graph(final_signal, freq_bins, time_range, sample_freq, max_width):
    '''
    Number of Frequency Bins vs Time taken graph
    '''

    numpy_vectorized, theano_vectorized, theano_fully_vectorized = [], [], []
    numpy_fully_vectorized, cqt_values, numpy_symmetric, theano_symmetric = [], [], [], []

    for i in freq_bins:
        # three_loop.append(threeloop_cqt(final_signal, 44100, i)[1])
        # two_kernel.append(cqt_two_with_kernel(final_signal, 44100, i)[1])
        # two_w_kernel.append(cqt_two_without_kernel(final_signal, 44100, i)[1])
        frequency_range = 100. * \
            2. ** (1. / (12. * i) * np.arange(0, 50, (1.0 / i)))
        kernel_details = pre_compute_kernels(
            sample_freq, frequency_range, max_width)
        symmetric_kernels = pre_compute_algebraic_kernel(
            sample_freq, frequency_range, max_width)
        numpy_vectorized.append(
            vectorized_numpy(
                final_signal,
                time_range,
                frequency_range,
                kernel_details)[1])
        theano_vectorized.append(
            vectorized_theano(
                final_signal,
                time_range,
                frequency_range,
                kernel_details)[1])
        theano_fully_vectorized.append(
            full_vectorized_theano(
                final_signal,
                time_range,
                frequency_range,
                kernel_details)[1])
        numpy_fully_vectorized.append(
            full_vectorized_numpy(
                final_signal,
                time_range,
                frequency_range,
                kernel_details)[1])
        numpy_symmetric.append(
            symmetric_numpy(
                final_signal,
                time_range,
                frequency_range,
                symmetric_kernels)[1])
        theano_symmetric.append(
            symmetric_theano(
                final_signal,
                time_range,
                frequency_range,
                symmetric_kernels)[1])
        cqt_values = [numpy_vectorized, theano_vectorized, theano_fully_vectorized, numpy_fully_vectorized,
                      numpy_symmetric, theano_symmetric]
    plt_title = "Variable channel vs Time Graph"
    axes_label = ["Frequency Bins", "Time taken for execution (in sec)"]
    plot_graph(cqt_values, plt_title, freq_bins, axes_label)


def test_cqts(final_signal, time_range, sample_freq,
              frequency_range, max_width):
    default_kernel = pre_compute_kernels(
        sample_freq, frequency_range, max_width)
    symmetric_kernel = pre_compute_algebraic_kernel(
        sample_freq, frequency_range, max_width)
    numpy_vectorized = vectorized_numpy(
        final_signal,
        time_range,
        frequency_range,
        default_kernel)
    theano_vectorized = vectorized_theano(
        final_signal, time_range, frequency_range, default_kernel)
    numpy_fully_vectorized = full_vectorized_numpy(
        final_signal, time_range, frequency_range, default_kernel)
    theano_fully_vectorized = full_vectorized_theano(
        final_signal, time_range, frequency_range, default_kernel)
    numpy_symmetric = symmetric_numpy(
        final_signal,
        time_range,
        frequency_range,
        symmetric_kernel)
    theano_symmetric = symmetric_theano(
        final_signal,
        time_range,
        frequency_range,
        symmetric_kernel)
    assert np.allclose(numpy_vectorized[0].all(), theano_vectorized[0].all())
    assert np.allclose(
        numpy_fully_vectorized[0].all(),
        theano_fully_vectorized[0].all())
    assert np.allclose(
        numpy_fully_vectorized[0].all(),
        numpy_vectorized[0].all())
    assert np.allclose(
        theano_fully_vectorized[0].all(),
        numpy_symmetric[0].all())
    assert np.allclose(numpy_symmetric[0].all(), theano_symmetric[0].all())

if __name__ == '__main__':
    signal_base = 132300
    mul_factor = 4
    sample_freq = 44100
    max_width = 1 / (.03 * 100) * sample_freq
    frequency_range = 100. * 2.**(1. / (12.) * np.arange(0, 50))
    final_signal = generate_signal(signal_base)
    time_range = np.arange(0, len(final_signal) -
                           int(max_width), int(max_width / 2))
    default_kernel = pre_compute_kernels(
        sample_freq, frequency_range, max_width)
    symmetric_kernel = pre_compute_algebraic_kernel(
        sample_freq, frequency_range, max_width)
    resolution = np.arange(1, 5)
    kernel_details = (default_kernel, symmetric_kernel)
    # channel_graph(final_signal, resolution, time_range, sample_freq, max_width)
    test_cqts(
        final_signal,
        time_range,
        sample_freq,
        frequency_range,
        max_width)
    usertime_graph(
        signal_base,
        mul_factor,
        time_range,
        frequency_range,
        kernel_details,
        max_width)

    '''
    import pylab
    pylab.imshow(np.log(np.abs(plot_matrix)), origin='lower', aspect='auto')
    pylab.xlabel('Time (sec)')
    pylab.ylabel('Frequency in Log scale')
    pylab.show()
    '''
