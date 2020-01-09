import numpy as np
from time import time
from astropy.wcs import WCS
from astropy.io import fits

#### internal modules
from crossmatching import gen_xmatch
from simult_fit import fit_astrom_simult


root_path = "/storage/goto/gotophoto/storage/pipeline/2020-01-04/final/r0230719_UT8.fits"

def astrom_task(infilepath):
    ''' A testing function, showing how to use the functions.
        infilepath -- path to image to solveself.

        Returns: lots of summary statistics as a placeholder for actual QA functions.
    '''
    tick_xmatch = time()
    _platecoords, _skycoords = gen_xmatch(infilepath, prune=False)
    tock_xmatch = time()
    header = fits.getheader(infilepath, 1)
    head_wcs = WCS(header)
    resid_before = (head_wcs.all_pix2world(_platecoords, 0) - _skycoords*180/np.pi)*3600

    tick_fit = time()
    new_wcs = fit_astrom_simult(_platecoords, _skycoords, header)
    resid = (new_wcs.all_pix2world(_platecoords, 0) - _skycoords*180/np.pi)*3600
    tock_fit = time()

    filename = infilepath.split("/")[-1]
    pre_med = np.average(np.median(resid_before, axis=0))
    post_med = np.average(np.median(resid, axis=0))
    pre_rms = np.average(np.std(resid_before, axis=0))
    post_rms = np.average(np.std(resid, axis=0))
    pre_chisq = np.sum(resid_before**2)
    post_chisq = np.sum(resid**2)
    fittime = np.round(tock_fit - tick_fit, 3)
    xmatchtime = np.round(tock_xmatch - tick_xmatch, 3)
    sourcedens = len(resid)

    output = [filename, pre_med, post_med, pre_rms, post_rms, pre_chisq, post_chisq, fittime, xmatchtime, sourcedens]
    return output

print(astrom_task(root_path))
