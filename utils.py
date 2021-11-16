from astropy.io import fits
import numpy as np

def save_sims(filename='simulations.fits', sims=None, true_shear=None, sigma_n=None, overwrite=False):
  # saving stamps
  stamps = fits.ImageHDU(sims)
  stamps.name = 'GAL_STAMPS'
  
  # saving simulation parameters
  col1 = fits.Column(name='g1', format='E', array=[true_shear[0]])
  col2 = fits.Column(name='g2', format='E', array=[true_shear[1]])
  col3 = fits.Column(name='sigma_n', format='E', array=[sigma_n])
  cols = fits.ColDefs([col1, col2, col3])
  hdu_params = fits.BinTableHDU.from_columns(cols)
  hdu_params.name = 'PARAMETERS'

  hdr = fits.Header()
  hdr['COMMENT'] = 'Sersic simulations with constant shear'
  empty_primary = fits.PrimaryHDU(header=hdr)
  hdul = fits.HDUList([empty_primary, hdu_params, stamps])

  hdul.writeto(filename, overwrite=overwrite)


def load_sims(filename):
  with fits.open(filename) as hdul:
    sims = hdul['GAL_STAMPS'].data
    params = hdul['PARAMETERS'].data
    true_shear = [params['g1'][0], params['g2'][0]]
    sigma_n = params['sigma_n'][0]

  return sims, true_shear, sigma_n
