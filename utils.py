import numpy as np
import tensorflow as tf

# utils

def radial_profile(power_spectrum_2d):
    """
    Compute the radial profile of 2d image
    :param data: 2d image
    :return: radial profile
    """
    center = power_spectrum_2d.shape[0]/2
    v, u = np.indices((power_spectrum_2d.shape))
    k = np.sqrt((u - center)**2 + (v - center)**2)
    k = k.astype('int32')

    tbin = np.bincount(k.ravel(), power_spectrum_2d.ravel())
    nr = np.bincount(k.ravel())
    radialprofile = tbin / nr
    return radialprofile

def measure_power_spectrum(map_data, pixel_size):
    """
    measures power 2d data
    :param map_data: map (n x n)
    :param pixel_size: pixel size (rad/pixel)
    :return: k
    :return: pk
    """
    map_size = map_data.shape[0]
    data_ft = np.abs(np.fft.fft2(map_data))
    data_ft_shifted = np.fft.fftshift(data_ft) 
    power_spectrum_2d = np.abs(data_ft_shifted * np.conjugate(data_ft_shifted)) / map_size**2
    nyquist = np.int(data_ft_shifted.shape[0] / 2)
    radialprofile = radial_profile(power_spectrum_2d)
    power_spectrum_1d = radialprofile[:nyquist]

    k = np.arange(power_spectrum_1d.shape[0])
    #k = np.fft.fftfreq(power_spectrum_1d.shape[0])
    return k, power_spectrum_1d


def make_power_map(power_spectrum, size, kps=None): 
  k1 = np.arange(size)
  k2 = np.arange(size)
  kcoords = np.meshgrid(k1,k2)
  # Now we can compute the k vector
  k = np.sqrt((kcoords[0]-size/2)**2 + (kcoords[1]-size/2)**2)
  
  ps_map = np.interp(k.flatten(), kps, power_spectrum).reshape([size,size])

  return ps_map # Carefull, this is not fftshifted

def get_ps_map(map_size, resolution):
  ps = np.load('ps_halofit.npy')
  
  pixel_size = np.pi * resolution / 180. / 60. #rad/pixel
  
  ell = ps[0,:]
  ps_halofit = ps[1,:] # normalisation by pixel size
  
  ks = ell / 2 / np.pi * pixel_size * 360
  pk = ps_halofit / (pixel_size)**2

  # Interpolate the Power Spectrum in Fourier Space
  ps_map = np.array(make_power_map(pk, map_size, kps=ks))
  return ps_map


def ks93(g1, g2):
  """Direct inversion of weak-lensing shear to convergence.
  This function is an implementation of the Kaiser & Squires (1993) mass
  mapping algorithm. Due to the mass sheet degeneracy, the convergence is
  recovered only up to an overall additive constant. It is chosen here to
  produce output maps of mean zero. The inversion is performed in Fourier
  space for speed.
  Parameters
  ----------
  g1, g2 :  2-D Tensor 
      2D input arrays corresponding to the first and second (i.e., real and
      imaginary) components of shear, binned spatially to a regular grid.
  Returns
  -------
  kE, kB : tuple of numpy arrays
      E-mode and B-mode maps of convergence.
  Raises
  ------
  AssertionError
      For input arrays of different sizes.
  """
  # Check consistency of input maps
  assert g1.shape == g2.shape

  # Compute Fourier space grids
  (batch_size, nx, ny) = g1.shape
  k1, k2 = tf.meshgrid(np.fft.fftfreq(ny), np.fft.fftfreq(nx))

  g1hat = tf.signal.fft2d(tf.cast(g1, dtype=tf.complex64))
  g2hat = tf.signal.fft2d(tf.cast(g2, dtype=tf.complex64))

  # Apply Fourier space inversion operator
  p1 = k1 * k1 - k2 * k2
  p2 = 2 * k1 * k2
  k2 = k1 * k1 + k2 * k2
  mask = np.zeros(k2.shape)
  mask[0, 0] = 1
  k2 = k2 + tf.convert_to_tensor(mask)
  p1 = tf.cast(p1, dtype=tf.complex64)
  p2 = tf.cast(p2, dtype=tf.complex64)
  k2 = tf.cast(k2, dtype=tf.complex64)
  kEhat = (p1 * g1hat + p2 * g2hat) / k2
  kBhat = -(p2 * g1hat - p1 * g2hat) / k2

  # Transform back to pixel space
  kE = tf.math.real(tf.signal.ifft2d(kEhat))
  kB = tf.math.real(tf.signal.ifft2d(kBhat))

  return kE, kB


def ks93inv(kE, kB):
  """Direct inversion of weak-lensing convergence to shear.
  This function provides the inverse of the Kaiser & Squires (1993) mass
  mapping algorithm, namely the shear is recovered from input E-mode and
  B-mode convergence maps.
  Parameters
  ----------
  kE, kB : 2-D Tensor 
      2D input arrays corresponding to the E-mode and B-mode (i.e., real and
      imaginary) components of convergence.
  Returns
  -------
  g1, g2 : tuple of numpy arrays
      Maps of the two components of shear.
  Raises
  ------
  AssertionError
      For input arrays of different sizes.
  See Also
  --------
  ks93
      For the forward operation (shear to convergence).
  """
  # Check consistency of input maps
  assert kE.shape == kB.shape

  # Compute Fourier space grids
  (batch_size, nx, ny) = kE.shape
  k1, k2 = tf.meshgrid(np.fft.fftfreq(ny), np.fft.fftfreq(nx))
  k1 = tf.expand_dims(k1, 0)
  k2 = tf.expand_dims(k2, 0)

  # Compute Fourier transforms of kE and kB
  kEhat = tf.signal.fft2d(tf.cast(kE, dtype=tf.complex64))
  kBhat = tf.signal.fft2d(tf.cast(kB, dtype=tf.complex64))

  # Apply Fourier space inversion operator
  p1 = k1 * k1 - k2 * k2
  p2 = 2 * k1 * k2
  k2 = k1 * k1 + k2 * k2
  mask = np.zeros(k2.shape)
  mask[0, 0, 0] = 1
  k2 = k2 + tf.convert_to_tensor(mask)
  p1 = tf.cast(p1, dtype=tf.complex64)
  p2 = tf.cast(p2, dtype=tf.complex64)
  k2 = tf.cast(k2, dtype=tf.complex64)
  g1hat = (p1 * kEhat - p2 * kBhat) / k2
  g2hat = (p2 * kEhat + p1 * kBhat) / k2

  # Transform back to pixel space
  g1 = tf.math.real(tf.signal.ifft2d(g1hat))
  g2 = tf.math.real(tf.signal.ifft2d(g2hat))

  return g1, g2