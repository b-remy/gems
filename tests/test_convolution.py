"""
This test script run ngmix/examples/metacal/metacal.py, but performing the convolution with GalFlow.

We test here only:
- odd stamp size
- Gaussian profiles (other profiles need a resampling...)
"""

from numpy.testing import assert_allclose

import numpy as np
import ngmix
import galsim
import tensorflow as tf
import galflow
import matplotlib.pyplot as plt

_stamp_size = 44
_pi = np.pi
_scale = 0.263

psf_fwhm = 0.9
psf = galsim.Moffat(
        beta=2.5, fwhm=psf_fwhm,
    ).shear(
        g1=0.02,
        g2=-0.01,
    )

psf = galsim.Convolve(psf, galsim.Pixel(_scale))

dx, dy = 0., 0.

im_psf = psf.drawImage(nx=_stamp_size, ny=_stamp_size, scale=_scale, use_true_center=False, method='no_pixel')

def main():
    args = get_args()

    print(args.lp, 'light profiles')

    shear_true = [0.01, 0.00]
    print('expected shear:', shear_true)
    print()
    
    rng = np.random.RandomState(args.seed)

    # We will measure moments with a fixed gaussian weight function
    weight_fwhm = 1.2
    fitter = ngmix.gaussmom.GaussMom(fwhm=weight_fwhm)
    psf_fitter = ngmix.gaussmom.GaussMom(fwhm=weight_fwhm)

    # these "runners" run the measurement code on observations
    psf_runner = ngmix.runners.PSFRunner(fitter=psf_fitter)
    runner = ngmix.runners.Runner(fitter=fitter)

    # this "bootstrapper" runs the metacal image shearing as well as both psf
    # and object measurements
    #
    # We will just do R11 for simplicity and to speed up this example;
    # typically the off diagonal terms are negligible, and R11 and R22 are
    # usually consistent

    boot = ngmix.metacal.MetacalBootstrapper(
        runner=runner, psf_runner=psf_runner,
        rng=rng,
        psf=args.psf,
        types=['noshear', '1p', '1m'],
    )

    dlist = []

    for i in progress(args.ntrial, miniters=10):

        obs = make_data(rng=rng, noise=args.noise, shear=shear_true)

        resdict, obsdict = boot.go(obs)

        for stype, sres in resdict.items():
            st = make_struct(res=sres, obs=obsdict[stype], shear_type=stype)
            dlist.append(st)

    print()

    data = np.hstack(dlist)

    w = select(data=data, shear_type='noshear')
    w_1p = select(data=data, shear_type='1p')
    w_1m = select(data=data, shear_type='1m')

    g = data['g'][w].mean(axis=0)
    gerr = data['g'][w].std(axis=0) / np.sqrt(w.size)
    g1_1p = data['g'][w_1p, 0].mean()
    g1_1m = data['g'][w_1m, 0].mean()
    R11 = (g1_1p - g1_1m)/0.02

    shear = g / R11
    shear_err = gerr / R11

    m = shear[0]/shear_true[0]-1
    merr = shear_err[0]/shear_true[0]

    s2n = data['s2n'][w].mean()

    print(m)
    assert_allclose(m, 0., atol=1e-5)
    # print('S/N: %g' % s2n)
    # print('R11: %g' % R11)
    # print('m: %g +/- %g (99.7%% conf)' % (m, merr*3))
    # print('c: %g +/- %g (99.7%% conf)' % (shear[1], shear_err[1]*3))


def select(data, shear_type):
    """
    select the data by shear type and size

    Parameters
    ----------
    data: array
        The array with fields shear_type and T
    shear_type: str
        e.g. 'noshear', '1p', etc.

    Returns
    -------
    array of indices
    """

    w, = np.where(
        (data['flags'] == 0) & (data['shear_type'] == shear_type)
    )
    return w


def make_struct(res, obs, shear_type):
    """
    make the data structure

    Parameters
    ----------
    res: dict
        With keys 's2n', 'e', and 'T'
    obs: ngmix.Observation
        The observation for this shear type
    shear_type: str
        The shear type

    Returns
    -------
    1-element array with fields
    """
    dt = [
        ('flags', 'i4'),
        ('shear_type', 'U7'),
        ('s2n', 'f8'),
        ('g', 'f8', 2),
        ('T', 'f8'),
        ('Tpsf', 'f8'),
    ]
    data = np.zeros(1, dtype=dt)
    data['shear_type'] = shear_type
    data['flags'] = res['flags']
    if res['flags'] == 0:
        data['s2n'] = res['s2n']
        # for moments we are actually measureing e, the elliptity
        data['g'] = res['e']
        data['T'] = res['T']
    else:
        data['s2n'] = np.nan
        data['g'] = np.nan
        data['T'] = np.nan
        data['Tpsf'] = np.nan

        # we only have one epoch and band, so we can get the psf T from the
        # observation rather than averaging over epochs/bands
        data['Tpsf'] = obs.psf.meta['result']['T']

    return data


def make_data(rng, noise, shear):
    """
    simulate an exponential or Gaussian object with moffat psf

    Parameters
    ----------
    rng: np.random.RandomState
        The random number generator
    noise: float
        Noise for the image
    shear: (g1, g2)
        The shear in each component

    Returns
    -------
    ngmix.Observation
    """

    args = get_args()

    psf_noise = 1.0e-6

    scale = 0.263

    gal_hlr = 0.5
    dy, dx = 0.,0. # rng.uniform(low=-scale/2, high=scale/2, size=2)

    if args.lp=='gaussian':
        obj0 = galsim.Gaussian(
            half_light_radius=gal_hlr,
        ).shear(
            g1=shear[0],
            g2=shear[1],
        ).shift(
            dx=dx,
            dy=dy,
        )
    
    elif args.lp=='exponential':
        obj0 = galsim.Exponential(
            half_light_radius=gal_hlr,
        ).shear(
            g1=shear[0],
            g2=shear[1],
        ).shift(
            dx=dx,
            dy=dy,
        )
        
    else:
        raise ValueError("Light profile must be either 'exponential' or 'gaussian'.")

    """
    Perform the image and PSF convolution with
        1. GalFlow
        0. galsim
    """
    if args.galflowconv==1:
        im_gal = obj0.drawImage(nx=_stamp_size, ny=_stamp_size, scale=_scale, method='no_pixel')

        # Forward FFT
        im_gal_k = tf.signal.fft2d(im_gal.array)
        im_psf_k = tf.signal.fft2d(im_psf.array)

        # Fourier-based convolution
        im_conv_k = im_gal_k * (im_psf_k)

        # Inverse FFT
        # im = tf.signal.fftshift(tf.math.real(tf.signal.irfft2d(im_conv_k)))
        im = tf.signal.fftshift(tf.math.real(tf.signal.ifft2d(im_conv_k)))
    else:
        obj = galsim.Convolve(psf, obj0)
        im = obj.drawImage(nx=_stamp_size, ny=_stamp_size, scale=scale, method='no_pixel').array

    psf_im = psf.drawImage(nx=_stamp_size, ny=_stamp_size, scale=scale, method='no_pixel').array

    psf_im += rng.normal(scale=psf_noise, size=psf_im.shape)
    im += rng.normal(scale=noise, size=im.shape)

    cen = (np.array(im.shape)-1.0)/2.0
    psf_cen = (np.array(psf_im.shape)-1.0)/2.0

    jacobian = ngmix.DiagonalJacobian(
        row=cen[0] + dy/scale, col=cen[1] + dx/scale, scale=scale,
    )
    psf_jacobian = ngmix.DiagonalJacobian(
        row=psf_cen[0], col=psf_cen[1], scale=scale,
    )

    wt = im*0 + 1.0/noise**2
    psf_wt = psf_im*0 + 1.0/psf_noise**2

    psf_obs = ngmix.Observation(
        psf_im,
        weight=psf_wt,
        jacobian=psf_jacobian,
    )

    obs = ngmix.Observation(
        im,
        weight=wt,
        jacobian=jacobian,
        psf=psf_obs,
    )

    return obs


def progress(total, miniters=1):
    last_print_n = 0
    last_printed_len = 0
    sl = str(len(str(total)))
    mf = '%'+sl+'d/%'+sl+'d %3d%%'
    for i in range(total):
        yield i

        num = i+1
        if i == 0 or num == total or num - last_print_n >= miniters:
            meter = mf % (num, total, 100*float(num) / total)
            nspace = max(last_printed_len-len(meter), 0)

            print('\r'+meter+' '*nspace, flush=True, end='')
            last_printed_len = len(meter)
            if i > 0:
                last_print_n = num

    print(flush=True)


def get_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=31415,
                        help='seed for rng')
    parser.add_argument('--ntrial', type=int, default=1000,
                        help='number of trials')
    parser.add_argument('--noise', type=float, default=1.0e-6,
                        help='noise for images')
    parser.add_argument('--psf', default='gauss',
                        help='psf for reconvolution')
    parser.add_argument('--galflowconv', type=int, default=1,
                        help='whether we perform convolution using TF or galsim')
    parser.add_argument('--lp', type=str, default='gaussian',
                        help='specify the light profile, must be either "exponential" or "gaussian"')
    return parser.parse_args()

def test_convolution_odd_gaussians():
    main()
