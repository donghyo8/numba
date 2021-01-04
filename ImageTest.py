from __future__ import print_function, division, absolute_import
from scipy.misc import ascent
from numpy import ones
import numpy
from numba.decorators import jit
from timeit import default_timer as time
from pylab import subplot, imshow, show, title, gray


@jit(nopython=True)
def filter2d_core_1(image, filt, result):
    M, N = image.shape
    Mf, Nf = filt.shape
    print(M,N,Mf,Nf)
    Mf2 = Mf // 2
    Nf2 = Nf // 2
    for i in range(Mf2, M - Mf2):
        for j in range(Nf2, N - Nf2):
            num = 0
            for ii in range(Mf):
                for jj in range(Nf):
                    num += (filt[Mf-1-ii, Nf-1-jj] * image[i-Mf2+ii,j-Nf2+jj])
            result[i, j] = num


@jit(nopython=True)
def filter2d_1(image, filt):
    result = numpy.zeros_like(image)
    filter2d_core_1(image, filt, result)
    return result

image = ascent().astype(numpy.float64)
filter = ones((7,7), dtype=image.dtype)
result = filter2d_1(image, filter)

start = time()
result = filter2d_1(image, filter)
duration = time() - start
print("Numba Image Filter time = %f\n" % (duration))




def filter2d_core_2(image, filt, result):
    M, N = image.shape
    Mf, Nf = filt.shape
    Mf2 = Mf // 2
    Nf2 = Nf // 2
    for i in range(Mf2, M - Mf2):
        for j in range(Nf2, N - Nf2):
            num = 0
            for ii in range(Mf):
                for jj in range(Nf):
                    num += (filt[Mf-1-ii, Nf-1-jj] * image[i-Mf2+ii,j-Nf2+jj])
            result[i, j] = num

def filter2d_2(image, filt):
    result = numpy.zeros_like(image)
    filter2d_core_2(image, filt, result)
    return result


image = ascent().astype(numpy.float64)
filter = ones((7,7), dtype=image.dtype)
result = filter2d_2(image, filter)

start = time()
result = filter2d_2(image, filter)
duration = time() - start
print("General Image Filter time = %f\n" % (duration))


subplot(1,3,1)
imshow(image)
title('Original Image')
gray()

subplot(1,3,2)
imshow(result)
title('Numba Filtered Image')
gray()


show()