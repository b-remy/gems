import numpy as np
import galsim
from scipy.special import gamma

import matplotlib.pyplot as plt

# Test parameters
batch_size = 4
# im_size = 512
stamp_size = 64
pix_scale = 0.263

# Get Disk+Bulge parametric profiles from galsim

# Get fit parameters.  For 'sersicfit', the result is an array of 8 numbers for each
# galaxy:
#     SERSICFIT[0]: intensity of light profile at the half-light radius.
#     SERSICFIT[1]: half-light radius measured along the major axis, in units of pixels
#                   in the COSMOS lensing data reductions (0.03 arcsec).
#     SERSICFIT[2]: Sersic n.
#     SERSICFIT[3]: q, the ratio of minor axis to major axis length.
#     SERSICFIT[4]: boxiness, currently fixed to 0, meaning isophotes are all
#                   elliptical.
#     SERSICFIT[5]: x0, the central x position in pixels.
#     SERSICFIT[6]: y0, the central y position in pixels.
#     SERSICFIT[7]: phi, the position angle in radians.  If phi=0, the major axis is
#                   lined up with the x axis of the image.
# For 'bulgefit', the result is an array of 16 parameters that comes from doing a
# 2-component sersic fit.  The first 8 are the parameters for the disk, with n=1, and
# the last 8 are for the bulge, with n=4.

cosmos_cat = galsim.COSMOSCatalog()
indices = np.arange(batch_size) + 5
param_gal_list = cosmos_cat.makeGalaxy(indices, gal_type='parametric', chromatic=False)

bulgefit = cosmos_cat.getParametricRecord(indices)['bulgefit']

i_disk = bulgefit[:,0]
n_disk = 1.*np.ones(batch_size)
hlr_disk = bulgefit[:,1]

i_bulge = bulgefit[:,0+8] #TODO: need to convert this intensity to flux
n_bulge = 4.*np.ones(batch_size)
hlr_bulge = bulgefit[:,1+8]

q_disk = bulgefit[:,3]
phi_disk = bulgefit[:,7]

q_bulge = bulgefit[:,3+8]
phi_bulge = bulgefit[:,7+8]

# bulge_fracs = .5*np.ones(batch_size)
# knots_fracs = 0.1*np.ones(batch_size)
bulge_fracs = [1., 0., 0., 0.]
knots_fracs = [0., 0., 0.5, 1.]

b_1 = 1.67834699
b_4 = 7.66924944

def get_flux(hlr, n, q, i, b_n):
  gamma(2*n)
  return 2*np.pi*gamma(2*n)*np.exp(b_n)*q*hlr*hlr*i/np.power(b_n, 2*n)

fig = plt.figure()
for i in range(batch_size):
  flux_disk = get_flux(hlr_disk[i], n_disk[i], q_disk[i], i_disk[i], b_1)
  flux_bulge = get_flux(hlr_bulge[i], n_bulge[i], q_bulge[i], i_bulge[i], b_4)

  bulge = galsim.Sersic(n_bulge[i], half_light_radius=hlr_bulge[i])#, flux=flux_bulge)
  disk = galsim.Sersic(n_disk[i], half_light_radius=hlr_bulge[i])#, flux=flux_disk)

  # bulge_shape = galsim.Shear(q=q_bulge[i], beta=phi_bulge[i]*galsim.degrees)
  # bulge = bulge.shear(bulge_shape)
  e1_bulge = 0.1*np.random.rand()
  e2_bulge = 0.1*np.random.rand()
  bulge = bulge.shear(e1=e1_bulge, e2=e2_bulge)

  # disk_shape = galsim.Shear(q=q_disk[i], beta=phi_disk[i]*galsim.degrees)
  # disk = disk.shear(disk_shape)
  e1_disk = 0.2*np.random.rand()
  e2_disk = 0.2*np.random.rand()
  disk = disk.shear(e1=e1_disk, e2=e2_disk)

  knots_npoints = 100
  knots = galsim.RandomKnots(npoints=knots_npoints, profile=disk)

  bulge_frac = bulge_fracs[i]
  knots_frac = knots_fracs[i]
  gal = bulge_frac * bulge + (1-bulge_frac) * (1-knots_frac) * disk + (1-bulge_frac) * knots_frac * knots
  
  gal_flux = 1.e6
  gal = gal.withFlux(gal_flux)
  # Shear the whole galaxy profile
  gamma1 = 0.02
  gamma2 = 0.
  gal_shear = galsim.Shear(g1=gamma1, g2=gamma2)

  # Convolve with PSF
  beta = 3.5
  fwhm = .9
  psf = galsim.Moffat(beta=beta, fwhm=.9)

  psf_q = 0.025
  psf_shape = galsim.Shear(q=psf_q, beta=0.*galsim.degrees)
  psf.shear(psf_shape)

  conv = galsim.Convolve([gal, psf])
  
  # img = gal.drawImage(nx=stamp_size, ny=stamp_size, scale=pix_scale)
  img = conv.drawImage(nx=stamp_size, ny=stamp_size, scale=pix_scale)

  random_seed = 1314662
  rng = galsim.BaseDeviate(random_seed+1)
  gaussian_noise = galsim.GaussianNoise(rng, sigma=1.e2)
  img.addNoise(gaussian_noise)

  fig.add_subplot(2,2,i+1)
  plt.imshow(img.array, cmap='gray')#, vmin=0., vmax=0.005)
  plt.colorbar()

plt.show()