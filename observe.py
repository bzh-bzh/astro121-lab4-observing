import ugradio
from astropy.io import fits
import numpy as np

import argparse
import os


# Constants.
# Not pre-divided by 2; do this in the code.
BASE_LO_FREQ = 1420.4
SWITCHING_DFREQ = 2.369 # Corresponds to moving the spectrum to the right by 500 km/s.

INTEGRATION_TIME = 600 # seconds.

# Loads in a FITS file with multiple spectra, and take the mean and standard deviation of those spectra.
# Save the result as another FITS file, with the same primary header.
def reduce_spectra(in_fits_path, out_fits_path):
    in_hdul = fits.open(in_fits_path)
    nspec = in_hdul[0].header['NSPEC']
    nchan = in_hdul[0].header['NCHAN']
    auto0_spectra, auto1_spectra = np.zeros((nspec, nchan)), np.zeros((nspec, nchan))
    for i in range(1, nspec + 1):
        auto0_spectra[i - 1] = in_hdul[i].data['auto0_real']
        auto1_spectra[i - 1] = in_hdul[i].data['auto1_real']
    auto0_mean, auto0_std = np.mean(auto0_spectra, axis=0), np.std(auto0_spectra, axis=0)
    auto1_mean, auto1_std = np.mean(auto1_spectra, axis=0), np.std(auto1_spectra, axis=0)

    out_data = np.core.records.fromarrays([auto0_mean, auto0_std, auto1_mean, auto1_std], names='auto0_mean,auto0_std,auto1_mean,auto1_std')
    out_data_hdu = fits.BinTableHDU.from_columns(out_data)

    fits.HDUList([in_hdul[0], out_data_hdu]).writeto(out_fits_path)

def main(args):
    output_folder = args.output_folder
    noise_gen = ugradio.leusch.LeuschNoise()
    lo = ugradio.agilent.SynthDirect()
    spec = ugradio.leusch.Spectrometer()

    def take_spec(noise, lo_freq, output_fits_path):
        n_spec = np.ceil(INTEGRATION_TIME / spec.int_time()).astype(int)
        if noise:
            noise_gen.on()
        else:
            noise_gen.off()
        lo.set_frequency(lo_freq / 2, 'MHz')
        # TODO: record galactic coordinates.
        spec.read_spec(output_fits_path, n_spec, (0, 0), 'eq')
        # Add the LO frequency to the FITS file.
        with fits.open(output_fits_path, mode='update') as hdul:
            hdul[0].header.set('LO_FREQ', value=lo_freq, comment='LO frequency in MHz')
    
    # Take frequency-switched spectrum.
    take_spec(False, BASE_LO_FREQ - SWITCHING_DFREQ, os.path.join(output_folder, 'switched.fits'))
    # Noise-free spectrum.
    take_spec(False, BASE_LO_FREQ, os.path.join(output_folder, 'nonoise.fits'))
    # Noise-injected spectrum.
    take_spec(True, BASE_LO_FREQ, os.path.join(output_folder, 'noise.fits'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('output_folder', help='Path of output data folder.')
    args = parser.parse_args()
    main(args)