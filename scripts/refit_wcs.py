import time

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS

#### internal modules
from goto_astromtools.crossmatching import gen_xmatch, reduce_density
from goto_astromtools.simult_fit import fit_astrom_simult

root_path = "/storage/goto/gotophoto/storage/pipeline/2019-12-13/final/r0220083_UT5.fits"

def astrom_task(infilepath):
    """ A testing function, showing how to use the functions.
        infilepath -- path to image to solveself.

        Returns: lots of summary statistics as a placeholder for actual QA functions.
    """
    print("XMATCH")
    tick = time.time()
    _platecoords, _skycoords = gen_xmatch(infilepath, prune=True)

    # If field is dense even after the pruning in gen_xmatch, reduce density
    if len(_platecoords) > 40000:
        _platecoords, _skycoords = reduce_density(_platecoords, _skycoords, 2)

    tock = time.time()
    print("XMATCH DONE IN %s s" % (np.round(tock - tick, 3)))
    header = fits.getheader(infilepath, 1)
    head_wcs = WCS(header)
    resid_before = (head_wcs.all_pix2world(_platecoords, 0) - _skycoords*180/np.pi)*3600
    print("ASTROMETRY")
    tick = time.time()
    new_wcs = fit_astrom_simult(_platecoords, _skycoords, header)
    resid = (new_wcs.all_pix2world(_platecoords, 0) - _skycoords*180/np.pi)*3600

    ### Sanity check to make sure we haven't made it worse!
    pre_med = np.average(np.median(resid_before, axis=0))
    post_med = np.average(np.median(resid, axis=0))
    pre_rms = np.average(np.std(resid_before, axis=0))
    post_rms = np.average(np.std(resid, axis=0))

    ### Some logic here about the quality of fit compared to the old fit.
    ### If the RMS is greater than the x-match radius we can't expect good fitting.

    ### Check - is the refitted solution worse than the existing one?
    bad_mycode1 = (np.abs(pre_med) - np.abs(post_med) < 0) & (pre_rms - post_rms < 0)

    if bad_mycode1:
        print("Astrometry failed - trying CRPIX tweak trick")
        ### CRPIX tweak - bounce the reference pixel around to try and
        ### get out of the local minimum. This won't work for the worst frames
        ### but for ones with initially good solutions where we make it worse
        ### it seems to do the trick
        xinit, yinit = header["CRPIX1"], header["CRPIX2"]

        randtweak = np.random.uniform(-2, 2, (50,2))

        for l in randtweak:
            header["CRPIX1"] = xinit + l[0]
            header["CRPIX2"] = yinit + l[1]

            new_wcs = fit_astrom_simult(_platecoords, _skycoords, header)
            resid = (new_wcs.all_pix2world(_platecoords, 0) - _skycoords*180/np.pi)*3600

            #post_rms = np.average(np.std(resid, axis=0))
            median_good = np.abs(np.median(resid)) < np.abs(pre_med)
            #rms_good = np.std(resid) < pre_rms
            chisq_good = np.sum(resid**2) < np.sum(resid_before**2)
            good_status = median_good & chisq_good

            if good_status:
                break

        median_good = np.abs(np.median(resid)) < np.abs(pre_med)
        chisq_good = np.sum(resid**2) < np.sum(resid_before**2)
        good_status = median_good & chisq_good

        if good_status:
            print("CRPIX trick succeeded, check summary statistics")
        else:
            print("CRPIX trick failed, astrometry is invalid - reverting to astrometry.net solution.")
            new_wcs = head_wcs

    templ_header = new_wcs.to_header(relax=True)

    for i, x in enumerate(templ_header.keys()):
        header[x] = templ_header[x]

    hdul = fits.open(infilepath)
    hdul[1].header = header
    try:
        hdul.writeto("outfile.fits")
        runcode = "SUCCESS"
    except:
        runcode = "FAIL"

    return None


print(runcode)