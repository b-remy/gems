"""
Metacalibration (https://arxiv.org/abs/1702.02600, https://arxiv.org/abs/1702.02601)

In this example we perform basic metacalibration with no object detection or
object selections

Metacalibration is a method to calibrate weak lensing shear measurements.  It
involves creating artifically sheared images. This means deconvolving,
shearing, and reconvolving by a new function (a new PSF). The shear estimator
is then measured on all of these images in order to form an estimate of a
linear response (the calibration).

In this example, we use a bootstrapper, which is a wraper class to run
measurements on the object and psf.  We use simple gaussian weighted moments
for measurement.

In this example, we set two parameters for the metacal run: the psf type for
the final PSF in the image, and the types of images to generate.  These are set
when constructing the MetacalBootstrapper

the psf
    We deconvolve, shear the image, then reconvolve.

    Setting psf='fitgauss' means we reconvolve by a round gaussian psf, based on
    fitting the original psf with a gaussian and dilating it appropriately.

    Setting psf='gauss' uses a deterministic algorithm to create a Gaussian psf
    that is round and larger than the original.  This algorithm is a bit slower
    and can result in a slightly noisier measurement, because it is more
    conservative.

    The default is 'gauss'

the types
    types is the types of images to produce.  Here we just use minimal set
    of shears to speed up this example, where we only calculate the
    response of the g1 measurement to a shear in g1

        noshear: the deconvolved/reconvolved image but without shear.  This image
          is used to measure the shear estimator and other quantities.
        1p: sheared +g1
        1m: sheared -g1
            1p/1m are are used to calculate the response and selection effects.

    standard default set would also includes shears in g2 (2p, 2m)

This example is low noise without any blending.  It should take about a minute
to run and get a precise final shear estimate.  You should see that the
recovered shear is unbiased.  The printout should look something like this

    > python metacal.py
    S/N: 79381.3
    R11: 0.343469
    m: 4.60743e-06 +/- 0.000419315 (99.7% conf)
    c: -8.87648e-07 +/- 4.01498e-06 (99.7% conf)
"""
import numpy as np
import ngmix
import galsim

import utils

_scale = 0.03 # COSMOS pixel size in arcsec

def main():
    args = get_args()
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

    data, shear_true, sigma_n = utils.load_sims(args.filename)

    for i in progress(data.shape[0], miniters=10):
        obs = make_data(data, i, sigma_n)

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

    print('sigma_n: %g' % sigma_n)
    print('Metacalibration')
    print('----------------------')
    print('S/N: %g' % s2n)
    print('R11: %g' % R11)
    print('shear', shear)
    print('shear true', shear_true)
    print('m: %g +/- %g (99.7%% conf)' % (m, merr*3))
    print('c: %g +/- %g (99.7%% conf)' % (shear[1], shear_err[1]*3))


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

cat = galsim.COSMOSCatalog()

#def make_data(stamp, rng, noise, shear):
def make_data(data, i, sigma_n):
    """
    simulate an exponential object with moffat psf

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
    scale = _scale

    psf = cat.makeGalaxy(2,  gal_type='real', noise_pad_size=0).original_psf
    
    # import here simulations
    im = data[i,...]

    noise = sigma_n
    wt = im*0 + 1.0/noise**2

    psf_im = psf.drawImage(scale=scale).array

    psf_obs = ngmix.Observation(
        psf_im,
    )

    obs = ngmix.Observation(
        im,
        weight=wt,        
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
    parser.add_argument('--filename', default='./data/sims_toymodel1_0.fits',
                        help='path and name of data to process')
    return parser.parse_args()


if __name__ == '__main__':
    main()
