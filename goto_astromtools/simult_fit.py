import numpy as np

from astropy.wcs import WCS
from scipy.optimize import least_squares


def tweak_scalerot(arr, _platecoords, _skycoords, in_wcs):
    """
    Return residuals from a linear WCS solution

    Parameters
    ----------
    arr : float
        NumPy array specifying the linear transform - of the form [CRPIX1, CRPIX2, linear scale, rotation]
    _platecoords : float
        Detector coordinates for each matched source, in px
    _skycoords : float
        World coordinates (ra/dec) for each mathced source, given in degrees.
    in_wcs : astropy.wcs.WCS
        AstroPy WCS object representing the coarse solution from astrometry.net
    Returns
    -------

    """

    crx, cry, dscale, drot = arr
    trial_wcs = in_wcs.deepcopy()

    try:
        trn_matr = trial_wcs.wcs.pc
    except AttributeError:
        trn_matr = trial_wcs.wcs.cd

    trial_wcs.wcs.crval = np.array([crx, cry])

    cos, sin = np.cos(drot), np.sin(drot)
    rot_matr = np.array([(cos, -sin), (sin, cos)])

    newpc = np.matmul(rot_matr, trn_matr) * dscale
    trial_wcs.wcs.pc = newpc
    trial_wcs.wcs.cd = newpc

    newcoord = trial_wcs.all_pix2world(_platecoords, 0)
    resid = (newcoord - _skycoords * 180 / np.pi)

    return resid.flatten()


def return_scalerot(arr, in_wcs):
    crx, cry, dscale, drot = arr
    trial_wcs = in_wcs.deepcopy()

    try:
        trn_matr = trial_wcs.wcs.pc
    except AttributeError:
        trn_matr = trial_wcs.wcs.cd

    trial_wcs.wcs.crval = np.array([crx, cry])

    cos, sin = np.cos(drot), np.sin(drot)
    rot_matr = np.array([(cos, -sin), (sin, cos)])

    newpc = np.matmul(rot_matr, trn_matr) * dscale
    trial_wcs.wcs.pc = newpc
    trial_wcs.wcs.cd = newpc

    return trial_wcs


def tweak_all_simult(arr, _platecoords, _skycoords, in_wcs):
    """ The all-important function, takes an array and modifies WCS, then
        computes residual
    """
    crx, cry, dscale, drot, a02, a11, a20, b02, b11, b20, a12, a21, b12, b21, a03, a30, b03, b30 = arr
    trial_wcs = in_wcs.deepcopy()

    ### Linear tweaks
    try:
        trn_matr = trial_wcs.wcs.pc
    except AttributeError:
        trn_matr = trial_wcs.wcs.cd

    trial_wcs.wcs.crval = np.array([crx, cry])

    cos, sin = np.cos(drot), np.sin(drot)
    rot_matr = np.array([(cos, -sin), (sin, cos)])

    newpc = np.matmul(rot_matr, trn_matr) * dscale
    trial_wcs.wcs.pc = newpc
    trial_wcs.wcs.cd = newpc

    ### Now set new SIPs
    trial_wcs.sip.a[0][2] = a02
    trial_wcs.sip.a[1][1] = a11
    trial_wcs.sip.a[2][0] = a20

    trial_wcs.sip.b[0][2] = b02
    trial_wcs.sip.b[1][1] = b11
    trial_wcs.sip.b[2][0] = b20

    trial_wcs.sip.a[1][2] = a12
    trial_wcs.sip.a[2][1] = a21
    trial_wcs.sip.b[1][2] = b12
    trial_wcs.sip.b[2][1] = b21

    trial_wcs.sip.a[0][3] = a03
    trial_wcs.sip.a[3][0] = a30
    trial_wcs.sip.b[0][3] = b03
    trial_wcs.sip.b[3][0] = b30

    newcoord = trial_wcs.all_pix2world(_platecoords, 0)
    resid = (newcoord - _skycoords * 180 / np.pi)

    return resid.flatten()


def return_fullwcs(arr, in_wcs):
    """ Convenience function for modifying WCS in place.
    """
    crx, cry, dscale, drot, a02, a11, a20, b02, b11, b20, a12, a21, b12, b21, a03, a30, b03, b30 = arr
    trial_wcs = in_wcs.deepcopy()

    ### Linear tweaks
    try:
        trn_matr = trial_wcs.wcs.pc
    except AttributeError:
        trn_matr = trial_wcs.wcs.cd

    trial_wcs.wcs.crval = np.array([crx, cry])

    cos, sin = np.cos(drot), np.sin(drot)
    rot_matr = np.array([(cos, -sin), (sin, cos)])

    newpc = np.matmul(rot_matr, trn_matr) * dscale
    trial_wcs.wcs.pc = newpc
    trial_wcs.wcs.cd = newpc

    ### Now set new SIPs
    trial_wcs.sip.a[0][2] = a02
    trial_wcs.sip.a[1][1] = a11
    trial_wcs.sip.a[2][0] = a20

    trial_wcs.sip.b[0][2] = b02
    trial_wcs.sip.b[1][1] = b11
    trial_wcs.sip.b[2][0] = b20

    trial_wcs.sip.a[1][2] = a12
    trial_wcs.sip.a[2][1] = a21
    trial_wcs.sip.b[1][2] = b12
    trial_wcs.sip.b[2][1] = b21

    trial_wcs.sip.a[0][3] = a03
    trial_wcs.sip.a[3][0] = a30
    trial_wcs.sip.b[0][3] = b03
    trial_wcs.sip.b[3][0] = b30

    return trial_wcs


def fit_astrom_simult(_platecoords, _skycoords, header):
    """
    Ingest a set of cross-matched coordinates and an approximate WCS solution from astrometry.net.
    Return an accurate refitted WCS

    Parameters
    ----------
    _platecoords : float
        Detector coordinates for each matched source, in px
    _skycoords : float
        World coordinates (ra/dec) for each mathced source, given in degrees.
    header : astropy.io.fits.Header
        FITS header for the given frame, given as output by GOTOphoto
    Returns
    -------
    header_wcs : astropy.io.fits.Header
        FITS header containing newly-updated WCS
    """

    header_wcs = WCS(header)

    crval = header_wcs.wcs.crval
    sip_a = header_wcs.sip.a
    sip_b = header_wcs.sip.b

    init_lin = [crval[0], crval[1], 1, 0]
    init_quad = [sip_a[0][2], sip_a[1][1], sip_a[2][0], sip_b[0][2], sip_b[1][1], sip_b[2][0]]
    init_cubic1 = [sip_a[1][2], sip_a[2][1], sip_b[1][2], sip_b[2][1]]
    init_cubic2 = [sip_a[0][3], sip_a[3][0], sip_b[0][3], sip_b[3][0]]

    init_vector = init_lin + init_quad + init_cubic1 + init_cubic2

    ### Need bounds for the linear transform otherwise get degeneracy in rotation angle
    bds = ((0, -90, 0, -np.pi / 2), (360, 90, np.inf, np.pi / 2))

    res_lin = least_squares(tweak_scalerot, x0=init_lin, args=(_platecoords, _skycoords, header_wcs), bounds=bds,
                            x_scale='jac')
    header_wcs = return_scalerot(res_lin.x, header_wcs)

    res_cubic = least_squares(tweak_all_simult, x0=init_vector, args=(_platecoords, _skycoords, header_wcs),
                              x_scale='jac')
    header_wcs = return_fullwcs(res_cubic.x, header_wcs)

    return header_wcs
