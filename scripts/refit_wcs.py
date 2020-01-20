import time

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS

#### internal modules
from scipy.interpolate import RectBivariateSpline

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

    print("XMATCH DONE IN %s s" % (np.round(tock - tick, 3)))
    header = fits.getheader(infilepath, 1)
    head_wcs = WCS(header)
    resid_before = (head_wcs.all_pix2world(_platecoords, 0) - _skycoords * 180 / np.pi) * 3600
    print("ASTROMETRY")
    new_wcs = fit_astrom_simult(_platecoords, _skycoords, header)
    resid = (new_wcs.all_pix2world(_platecoords, 0) - _skycoords * 180 / np.pi) * 3600

    # Sanity check to make sure we haven't made it worse!
    pre_med = np.average(np.median(resid_before, axis=0))
    post_med = np.average(np.median(resid, axis=0))
    pre_rms = np.average(np.std(resid_before, axis=0))
    post_rms = np.average(np.std(resid, axis=0))

    # Some logic here about the quality of fit compared to the old fit.
    # If the RMS is greater than the x-match radius we can't expect good fitting.

    ### Check - is the refitted solution worse than the existing one?
    bad_mycode1 = (np.abs(pre_med) - np.abs(post_med) < 0) & (pre_rms - post_rms < 0)

    if bad_mycode1:
        print("Astrometry failed - trying CRPIX tweak trick")
        # CRPIX tweak - bounce the reference pixel around to try and
        # get out of the local minimum. This won't work for the worst frames
        # but for ones with initially good solutions where we make it worse
        # it seems to do the trick
        xinit, yinit = header["CRPIX1"], header["CRPIX2"]

        randtweak = np.random.uniform(-2, 2, (50, 2))

        for pos in randtweak:
            header["CRPIX1"] = xinit + pos[0]
            header["CRPIX2"] = yinit + pos[1]

            new_wcs = fit_astrom_simult(_platecoords, _skycoords, header)
            resid = (new_wcs.all_pix2world(_platecoords, 0) - _skycoords * 180 / np.pi) * 3600

            # post_rms = np.average(np.std(resid, axis=0))
            median_good = np.abs(np.median(resid)) < np.abs(pre_med)
            # rms_good = np.std(resid) < pre_rms
            chisq_good = np.sum(resid ** 2) < np.sum(resid_before ** 2)
            good_status = median_good & chisq_good

            if good_status:
                break

        median_good = np.abs(np.median(resid)) < np.abs(pre_med)
        chisq_good = np.sum(resid ** 2) < np.sum(resid_before ** 2)
        good_status = median_good & chisq_good

        if good_status:
            print("CRPIX trick succeeded, check summary statistics")
        else:
            print("CRPIX trick failed, astrometry is invalid - reverting to astrometry.net solution.")
            new_wcs = head_wcs

    rmsvec = np.std(resid, axis=0)

    rasig, decsig = rmsvec

    header["RA_SIG"] = rasig
    header["DEC_SIG"] = decsig

    # Compute RMS mesh from residuals.

    stepx = 1022
    stepy = 1022
    sizex = header["NAXIS1"]
    sizey = header["NAXIS2"]

    xcorners = np.arange(0, sizex, stepy)
    ycorners = np.arange(0, sizey, stepy)

    rms_matrix = np.zeros((len(xcorners), len(ycorners), 2))
    median_matrix = np.zeros((len(xcorners), len(ycorners), 2))
    nsources = np.zeros((len(xcorners), len(ycorners), 2))

    _xs, _ys = _platecoords[:, 0], _platecoords[:, 1]

    for i, x in enumerate(xcorners):
        for j, y in enumerate(ycorners):
            mask = (_xs > x) & (_xs < x + stepx) & (_ys > y) & (_ys < y + stepy)
            rms_matrix[i][j] = np.std(resid[mask], axis=0)
            median_matrix[i][j] = np.median(resid[mask], axis=0)
            nsources[i][j] = np.sum(mask)

    ra_rms_matrix, dec_rms_matrix = np.dsplit(rms_matrix, 2)
    ra_rms_matrix.squeeze()
    dec_rms_matrix.squeeze()

    # Move to tile midpoint coord frame. be careful around detector edges!
    xpl = xcorners + stepx / 2
    ypl = ycorners + stepy / 2

    ra_rms_fn = RectBivariateSpline(xpl, ypl, ra_rms_matrix)
    dec_rms_fn = RectBivariateSpline(xpl, ypl, dec_rms_matrix)

    # Write WCS to header
    templ_header = new_wcs.to_header(relax=True)
    for i, x in enumerate(templ_header.keys()):
        header[x] = templ_header[x]

    hdul = fits.open(infilepath, mode='readonly')
    hdul[1].header = header
    photom = hdul[3].data

    # Now time to remake the photometry table, with updated positions and errors.
    xs = photom['x']
    ys = photom['y']
    platecoords = np.vstack((xs, ys)).T

    ra_new_err = ra_rms_fn(xs, ys, grid=False)
    dec_new_err = dec_rms_fn(xs, ys, grid=False)

    newskycoords = new_wcs.all_pix2world(platecoords, 0)
    ras_new, decs_new = newskycoords[:, 0], newskycoords[:, 1]

    photom['RA'] = ras_new
    photom["Dec"] = decs_new
    photom['ra_err'] = ra_new_err  # error from astrometric rms >> point error.
    photom['ra_err'] = dec_new_err

    # Overwrite photometry table with updated positions and errors.
    hdul[3].data = photom

    # Write out file to new FITS.
    try:
        hdul.writeto("outfile.fits")
        runcode = "SUCCESS"
    except:
        runcode = "FAIL"

    hdul.close()
    return runcode


print(astrom_task(root_path))
