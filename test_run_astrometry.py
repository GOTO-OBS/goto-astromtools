import numpy as np
from astropy.wcs import WCS
from astropy.io import fits

#### internal modules
from crossmatching import gen_xmatch, reduce_density
from simult_fit import fit_astrom_simult


root_path = "/storage/goto/gotophoto/storage/pipeline/2019-12-13/final/r0220305_UT7.fits"

def astrom_task(infilepath):
    ''' A testing function, showing how to use the functions.
        infilepath -- path to image to solveself.

        Returns: lots of summary statistics as a placeholder for actual QA functions.
    '''
    _platecoords, _skycoords = gen_xmatch(infilepath, prune=True)

    ### If field is dense even after the pruning in gen_xmatch, reduce density
    if len(_platecoords) > 40000:
        _platecoords, _skycoords = reduce_density(_platecoords, _skycoords, 2)

    header = fits.getheader(infilepath, 1)
    head_wcs = WCS(header)
    resid_before = (head_wcs.all_pix2world(_platecoords, 0) - _skycoords*180/np.pi)*3600

    new_wcs = fit_astrom_simult(_platecoords, _skycoords, header)
    resid = (new_wcs.all_pix2world(_platecoords, 0) - _skycoords*180/np.pi)*3600

    filename = infilepath.split("/")[-1]
    pre_med = np.average(np.median(resid_before, axis=0))
    post_med = np.average(np.median(resid, axis=0))
    pre_rms = np.average(np.std(resid_before, axis=0))
    post_rms = np.average(np.std(resid, axis=0))
    pre_chisq = np.sum(resid_before**2)
    post_chisq = np.sum(resid**2)
    sourcedens = len(resid)

    output = [filename, pre_med, post_med, pre_rms, post_rms, pre_chisq, post_chisq, sourcedens]
    return output

print(astrom_task(root_path))
