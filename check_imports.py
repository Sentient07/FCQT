
@profile
def check_imports():
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

@profile
def check_random_compile():
	import theano
	import theano.tensor as T
	x = T.fmatrix()
	y = T.fmatrix()
	outp = x.dot(y.T)
	func = theano.function([x, y], outp, allow_input_downcast=True)

if __name__ == '__main__':
	check_imports()
	check_random_compile()