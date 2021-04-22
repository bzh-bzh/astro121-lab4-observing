import ugradio
from astropy.io import fits
from astropy.time import Time, TimeDelta
import astropy.coordinates
from astropy.coordinates import AltAz
import astropy.units as u
import numpy as np
import pandas as pd
import time

import argparse
import os
from typing import Tuple
import concurrent.futures

"""
DESIGN NOTES

The script takes in an output folder and an input "observing plan" CSV, 
which should have at minimum three columns: galactic coords l, b, and a boolean "observed already?" value.

The outline of the script is:
Read in the observing plan file.
Start from top to bottom.
For each entry that hasn't been observed already, calculate whether the pointing can be tracked by the telescope for the expected integration time.
If so, start tracking the pointing, and concurrently start taking the spectra we need. Save the spectra to a temporary folder.
Once we successfully take the spectra, with no tracking errors, we commit our pointing:
Reduce the spectra and move the reduced spectra to a subfolder under our output folder.
Mark the pointing as "observed" in our observing plan and re-save the CSV to disk.

Why the take spectra then commit flow? In case we run into some kind of tracking or other error during our pointing,
we don't want to mistakenly mark the pointing as done.

We go from top to bottom in the observing plan so we can prioritize a coarse spacing first.
"""


# Constants.
# Not pre-divided by 2; do this in the code.
BASE_LO_FREQ = 1420.4
SWITCHING_DFREQ = 2.369 # Corresponds to moving the spectrum to the left by 500 km/s.

DISH_LOCATION = astropy.coordinates.EarthLocation(lat=ugradio.leo.lat * u.deg, lon=ugradio.leo.lon * u.deg, height=ugradio.leo.alt * u.meter)

NOISE_INTEGRATION_TIME = 60 # seconds.
NONOISE_INTEGRATION_TIME = 300
SWITCHED_INTEGRATION_TIME = 300
TOTAL_INTEGRATION_TIME = NOISE_INTEGRATION_TIME + NONOISE_INTEGRATION_TIME + SWITCHED_INTEGRATION_TIME

RESLEW_TIME = 10

TMP_FOLDER = '/tmp/5quist/'

OBSERVING_PLAN_REQ_COLS = ['ra', 'dec', 'observed', 'jd_start', 'jd_end', 'alt_start', 'az_start', 'alt_end', 'az_end']

telescope = ugradio.leusch.LeuschTelescope()
noise_gen = ugradio.leusch.LeuschNoise()
lo = ugradio.agilent.SynthDirect()
spec = ugradio.leusch.Spectrometer()

# Loads in a FITS file with multiple spectra, and takes the mean and standard deviation of those spectra.
# Save the result as another FITS file, with the same primary header.
def reduce_spectra(in_fits_path: str, out_fits_path: str):
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

# Takes a single set of spectra.
def take_spec(noise: bool, lo_freq: float, int_time: float, coord: astropy.coordinates.SkyCoord, output_fits_path: str):
    n_spec = np.ceil(int_time / spec.int_time()).astype(int)
    if noise:
        noise_gen.on()
    else:
        noise_gen.off()
    lo.set_frequency(lo_freq / 2, 'MHz')
    spec.read_spec(output_fits_path, n_spec, (coord.l.value, coord.b.value), 'ga')
    # Add the LO frequency to the FITS file.
    with fits.open(output_fits_path, mode='update') as hdul:
        hdul[0].header.set('LO_FREQ', value=lo_freq, comment='LO frequency in MHz')

# Take all the spectra we need.
def take_all_spec(coord: astropy.coordinates.SkyCoord) -> Tuple[float, float]:
    num_spectrometer_retries = 0
    while True:
        if num_spectrometer_retries >= 5:
            raise IOError
        try:
            spec.check_connected()
            break
        except IOError:
            spec = ugradio.leusch.Spectrometer()
            num_spectrometer_retries += 1

    jd_start = Time.now().jd
    # Take frequency-switched spectrum.
    take_spec(False, BASE_LO_FREQ + SWITCHING_DFREQ, SWITCHED_INTEGRATION_TIME, coord, os.path.join(TMP_FOLDER, 'switched.fits'))
    # Noise-free spectrum.
    take_spec(False, BASE_LO_FREQ, NONOISE_INTEGRATION_TIME, coord, os.path.join(TMP_FOLDER, 'nonoise.fits'))
    # Noise-injected spectrum.
    take_spec(True, BASE_LO_FREQ, NOISE_INTEGRATION_TIME, coord, os.path.join(TMP_FOLDER, 'noise.fits'))
    jd_end = Time.now().jd
    return jd_start, jd_end

# Reduce the spectra in the tmp folder we just took, and move them to our final subfolder.
# label the subfolder after the l,b of a coord
def reduce_and_move_spectra(base_folder: str, coord: astropy.coordinates.SkyCoord):
    sf_name = coord.to_string() #turns l,b into a usable string
    sf_path = os.path.join(base_folder, sf_name)
    os.mkdirs(sf_path) #makes a directory to sf_path
    for in_fits_path in os.listdir(TMP_FOLDER):
        if in_fits_path.endswith('.fits'):
            reduce_spectra(os.path.join(TMP_FOLDER, in_fits_path), os.path.join(sf_path, in_fits_path))
            continue
        else:
            continue
    return

# Determines whether the specified coordinate pointing will be within the telescope's moving-limits for expected_int_time (in s).
# Could also try to exclude the hills in the north (< 30 deg altitude).
def is_pointing_trackable(coord: astropy.coordinates.SkyCoord, expected_int_time: float) -> bool:
    aa_now = AltAz(location=DISH_LOCATION, obstime=Time.now()) #making an AltAz frame for our coord
    altaz_now = coord.transform_to(aa_now)
    az_now = altaz_now.az
    alt_now = altaz_now.alt
    aa_int = AltAz(location=DISH_LOCATION, obstime=(Time.now() + TimeDelta(expected_int_time * u.second)))
    altaz_int = coord.transform_to(aa_int)
    az_int = altaz_int.az
    alt_int = altaz_int.alt
    ALT_MAX = 85
    ALT_MIN = 14
    #######
    if (-5 <= az_int <= 365) and (-5 <= az_now <= 365):
        if (ALT_MIN <= alt_int <= ALT_MAX) and (ALT_MIN <= alt_now <= ALT_MAX):
            print("you're good to go, pardner! yeehaw!")
            return True
        if not (ALT_MIN <= alt_int <= ALT_MAX):
            print(f"wait! we won't be able to track that for the whole integration! look for a different altitude.")
        if not (ALT_MIN <= alt_now <= ALT_MAX):
            print(f"we can't look at this right now! the current altitude is outside our Leuschner limit!")
    if not (-5 <= az_int <= 365):
        print(f"that won't be in our azimuthal range for the whole integration!")
    if not (-5 <= az_now <= 365):
        print(f"that's outside our azimuthal range right now. try something else!")
    
    return False

# Points towards source.
def point(source_coord: astropy.coordinates.SkyCoord):
    altaz = astropy.coordinates.AltAz(obstime=Time.now(), location=DISH_LOCATION)
    source_altaz = source_coord.transform_to(altaz)

    print('\n-------RESLEW-------')
    print(f'Attempting to reslew telescope to alt/az {source_altaz.alt}/{source_altaz.az}.')

    telescope.point(source_altaz.alt.value, source_altaz.az.value)

# Retry pointing n_retries times, in case it fails due to connection errors.
def point_retry(source_coord: astropy.coordinates.SkyCoord, n_retries: int = 4):
    attempt_num = 0
    while attempt_num < n_retries:
        try:
            point(source_coord)
        except (AssertionError, OSError):
            attempt_num += 1
            print('\nRetrying point.\n')
            time.sleep(5)
            continue
        return
    raise AssertionError

def main(args):
    observing_plan = pd.read_csv(args.observing_plan_path)
    output_folder = args.output_folder

    if args.max_time:
        max_runtime = Time.now() + TimeDelta(args.max_time * u.hour)
    else:
        max_runtime = Time.now() + TimeDelta(u.year)

    # Create temp folder and output folder if they don't exist.
    if not os.path.exists(TMP_FOLDER):
        os.makedirs(TMP_FOLDER)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Make sure our observing plan has all the required columns.
    assert all(col in observing_plan.columns for col in OBSERVING_PLAN_REQ_COLS)

    executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)

    try:
        # Start iterating from top to bottom through the observing plan.
        for i in range(len(observing_plan)):
            # End the script early if we surpass the maximum runtime.
            if Time.now() > max_runtime:
                break

            observing_info = observing_plan.iloc[i]
            if observing_info['observed']:
                print('Observing entry {} already observed; skipping.'.format(i))
                continue

            pointing_coord = astropy.coordinates.SkyCoord(ra=observing_info['ra'] * u.deg, dec=observing_info['dec'] * u.deg)
            pointing_coord = pointing_coord.galactic

            # 120 seconds of extra buffer time, to account for initial telescope moving + other misc stuff.
            if not is_pointing_trackable(pointing_coord, TOTAL_INTEGRATION_TIME + 120):
                print('Observing entry {} not above horizon; skipping.'.format(i))
                continue

            # Do initial pointing towards target.
            print('Doing initial pointing.')
            point_retry(pointing_coord)
            start_altaz = telescope.get_pointing()

            # Start taking our spectra in a different thread.
            spectra_future = executor.submit(take_all_spec, pointing_coord)

            # Start the tracking loop, until the take_all_spec function finishes.
            while not spectra_future.done():
                time.sleep(RESLEW_TIME)
                point_retry(pointing_coord)
            
            end_altaz = telescope.get_pointing()
            
            jd_start, jd_end = spectra_future.result()
            reduce_and_move_spectra(output_folder, pointing_coord)

            # Now that everything's successfully completed, we record that we completed this observation in the observing_plan,
            # and resave the DataFrame to the CSV.
            observing_plan.at[i, 'observed'] = 1
            observing_plan.at[i, 'jd_start'] = jd_start
            observing_plan.at[i, 'jd_end'] = jd_end
            observing_plan.at[i, 'alt_start'] = start_altaz[0]
            observing_plan.at[i, 'az_start'] = start_altaz[1]
            observing_plan.at[i, 'alt_end'] = end_altaz[0]
            observing_plan.at[i, 'az_end'] = end_altaz[1]
            observing_plan.to_csv(args.observing_plan_path, index=False)
    finally:
        print('Stowing telescope.')
        telescope.stow()



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('observing_plan_path', help='Path of observing plan CSV file.')
    parser.add_argument('output_folder', help='Path of output data folder.')
    parser.add_argument('--max-time', help='Max script runtime in hours.', type=float)
    args = parser.parse_args()
    main(args)