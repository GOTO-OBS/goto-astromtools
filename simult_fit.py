import numpy as np
from time import time

import catsHTM
from kdsphere import KDSphere

from astropy.io import fits
import astropy.units as u
from astropy.table import Table
from astropy.coordinates import SkyCoord
from astropy.time import Time
from astropy.wcs import WCS, InvalidTransformError
from scipy.optimize import least_squares

def fit_astrom_simult(_platecoords, _skycoords, header):
    ''' Ingest a set of cross-matched coordinates and
        an approximate WCS solution from astrometry.net.
        Return an accurate refitted WCS

        keyword_args:
        _platecoords -- detector coordinates of ref stars, in a 2xN array
        _skycoords -- sky coordinates of ref stars, 2xN array, in decimal degrees.
        header -- the input FITS header, optionally containing SIP coefficients.
    '''

    def tweak_scalerot(arr):
        crx, cry, dscale, drot = arr
        trial_wcs = header_wcs.deepcopy()
        trn_matr = trial_wcs.wcs.pc
        trial_wcs.wcs.crval = np.array([crx, cry])

        cos, sin = np.cos(drot), np.sin(drot)
        rot_matr = np.array([(cos,-sin), (sin, cos)])

        newpc = np.matmul(rot_matr, trn_matr) * dscale
        trial_wcs.wcs.pc = newpc

        newcoord = trial_wcs.all_pix2world(_platecoords, 0)
        resid = (newcoord - _skycoords * 180/np.pi)


        return resid.flatten()

    def return_scalerot(in_wcs, arr):
        crx, cry, dscale, drot = arr
        trial_wcs = in_wcs.deepcopy()
        trn_matr = trial_wcs.wcs.pc
        trial_wcs.wcs.crval = np.array([crx, cry])

        cos, sin = np.cos(drot), np.sin(drot)
        rot_matr = np.array([(cos,-sin), (sin, cos)])

        newpc = np.matmul(rot_matr, trn_matr) * dscale
        trial_wcs.wcs.pc = newpc

        return trial_wcs

    def tweak_all_simult(arr):
        crx, cry, dscale, drot, a02, a11, a20, b02, b11, b20,a03, a30, b03, b30 = arr
        trial_wcs = header_wcs.deepcopy()

        ### Linear tweaks
        trn_matr = trial_wcs.wcs.pc
        trial_wcs.wcs.crval = np.array([crx, cry])

        cos, sin = np.cos(drot), np.sin(drot)
        rot_matr = np.array([(cos,-sin), (sin, cos)])

        newpc = np.matmul(rot_matr, trn_matr) * dscale
        trial_wcs.wcs.pc = newpc

        ### Now set new SIPs
        trial_wcs.sip.a[0][2] = a02
        trial_wcs.sip.a[1][1] = a11
        trial_wcs.sip.a[2][0] = a20

        trial_wcs.sip.b[0][2] = b02
        trial_wcs.sip.b[1][1] = b11
        trial_wcs.sip.b[2][0] = b20

        trial_wcs.sip.a[0][3] = a03
        trial_wcs.sip.a[3][0] = a30
        trial_wcs.sip.b[0][3] = b03
        trial_wcs.sip.b[3][0] = b30

        newcoord = trial_wcs.all_pix2world(_platecoords, 0)
        resid = (newcoord - _skycoords * 180/np.pi)

        return resid.flatten()

    def return_fullwcs(in_wcs, arr):
        crx, cry, dscale, drot, a02, a11, a20, b02, b11, b20,a03, a30, b03, b30 = arr
        trial_wcs = in_wcs.deepcopy()

        ### Linear tweaks
        trn_matr = trial_wcs.wcs.pc
        trial_wcs.wcs.crval = np.array([crx, cry])

        cos, sin = np.cos(drot), np.sin(drot)
        rot_matr = np.array([(cos,-sin), (sin, cos)])

        newpc = np.matmul(rot_matr, trn_matr) * dscale
        trial_wcs.wcs.pc = newpc

        ### Now set new SIPs
        trial_wcs.sip.a[0][2] = a02
        trial_wcs.sip.a[1][1] = a11
        trial_wcs.sip.a[2][0] = a20

        trial_wcs.sip.b[0][2] = b02
        trial_wcs.sip.b[1][1] = b11
        trial_wcs.sip.b[2][0] = b20

        trial_wcs.sip.a[0][3] = a03
        trial_wcs.sip.a[3][0] = a30
        trial_wcs.sip.b[0][3] = b03
        trial_wcs.sip.b[3][0] = b30

        return trial_wcs

    header_wcs = WCS(header)

    crval = header_wcs.wcs.crval
    SIP_A = header_wcs.sip.a
    SIP_B = header_wcs.sip.b

    initLIN = [crval[0], crval[1], 1, 0]
    initQUAD = [SIP_A[0][2], SIP_A[1][1], SIP_A[2][0], SIP_B[0][2], SIP_B[1][1], SIP_B[2][0]]
    initCUBIC = [SIP_A[0][3], SIP_A[3][0], SIP_B[0][3], SIP_B[3][0]]

    init_vector = initLIN + initQUAD + initCUBIC

    init_resid = (header_wcs.all_pix2world(_platecoords, 0) - _skycoords * 180/np.pi)*3600 # in arcsec

    ### Need bounds for the linear transform otherwise get degeneracy in rotation angle
    bds = ((0, -90, 0, -np.pi/2), (360, 90, np.inf, np.pi/2))

    res_lin = least_squares(tweak_scalerot, initLIN, bounds=bds, x_scale='jac')
    header_wcs = return_scalerot(header_wcs, res_lin.x)
    crval = res_lin.x

    res_cubic = least_squares(tweak_all_simult, init_vector, x_scale='jac')
    header_wcs = return_fullwcs(header_wcs, res_cubic.x)

    return header_wcs
