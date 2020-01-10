import numpy as np
from time import time

import catsHTM
from goto_astromtools.kdsphere import KDSphere

from astropy.io import fits
import astropy.units as u
from astropy.table import Table
from astropy.coordinates import SkyCoord
from astropy.time import Time
from astropy.wcs import WCS, InvalidTransformError
from scipy.optimize import least_squares

def tweak_scalerot(arr, _platecoords, _skycoords, in_wcs):
    ''' Return residual from a linear WCS InvalidTransformError
        arr -- [CRPIX1, CRPIX2, linear scale change, rotation]
        _platecoords, _skycoords are detector coords in px
        and world coords, in degrees
        in_wcs is the guess wcs
    '''
    crx, cry, dscale, drot = arr
    trial_wcs = in_wcs.deepcopy()
    trn_matr = trial_wcs.wcs.pc
    trial_wcs.wcs.crval = np.array([crx, cry])

    cos, sin = np.cos(drot), np.sin(drot)
    rot_matr = np.array([(cos,-sin), (sin, cos)])

    newpc = np.matmul(rot_matr, trn_matr) * dscale
    trial_wcs.wcs.pc = newpc

    newcoord = trial_wcs.all_pix2world(_platecoords, 0)
    resid = (newcoord - _skycoords * 180/np.pi)

    return resid.flatten()

def return_scalerot(arr, in_wcs):
    crx, cry, dscale, drot = arr
    trial_wcs = in_wcs.deepcopy()
    trn_matr = trial_wcs.wcs.pc
    trial_wcs.wcs.crval = np.array([crx, cry])

    cos, sin = np.cos(drot), np.sin(drot)
    rot_matr = np.array([(cos,-sin), (sin, cos)])

    newpc = np.matmul(rot_matr, trn_matr) * dscale
    trial_wcs.wcs.pc = newpc

    return trial_wcs

def tweak_all_simult(arr, _platecoords, _skycoords, in_wcs):
    ''' The all-important function, takes an array and modifies WCS, then
        computes residual
    '''
    crx, cry, dscale, drot, a02, a11, a20, b02, b11, b20, a12, a21, b12, b21, a03, a30, b03, b30 = arr
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

    trial_wcs.sip.a[1][2] = a12
    trial_wcs.sip.a[2][1] = a21
    trial_wcs.sip.b[1][2] = b12
    trial_wcs.sip.b[2][1] = b21

    trial_wcs.sip.a[0][3] = a03
    trial_wcs.sip.a[3][0] = a30
    trial_wcs.sip.b[0][3] = b03
    trial_wcs.sip.b[3][0] = b30

    newcoord = trial_wcs.all_pix2world(_platecoords, 0)
    resid = (newcoord - _skycoords * 180/np.pi)

    return resid.flatten()

def return_fullwcs(arr, in_wcs):
    ''' Convenience function for modifying WCS in place.
    '''
    crx, cry, dscale, drot, a02, a11, a20, b02, b11, b20, a12, a21, b12, b21, a03, a30, b03, b30 = arr
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
    ''' Ingest a set of cross-matched coordinates and
        an approximate WCS solution from astrometry.net.
        Return an accurate refitted WCS

        keyword_args:
        _platecoords -- detector coordinates of ref stars, in a 2xN array
        _skycoords -- sky coordinates of ref stars, 2xN array, in decimal degrees.
        header -- the input FITS header, optionally containing SIP coefficients.
    '''


    header_wcs = WCS(header)

    crval = header_wcs.wcs.crval
    SIP_A = header_wcs.sip.a
    SIP_B = header_wcs.sip.b

    initLIN = [crval[0], crval[1], 1, 0]
    initQUAD = [SIP_A[0][2], SIP_A[1][1], SIP_A[2][0], SIP_B[0][2], SIP_B[1][1], SIP_B[2][0]]
    initCUBIC1 = [SIP_A[1][2], SIP_A[2][1], SIP_B[1][2], SIP_B[2][1]]
    initCUBIC2 = [SIP_A[0][3], SIP_A[3][0], SIP_B[0][3], SIP_B[3][0]]

    init_vector = initLIN + initQUAD + initCUBIC1 + initCUBIC2

    init_resid = (header_wcs.all_pix2world(_platecoords, 0) - _skycoords * 180/np.pi)*3600 # in arcsec

    ### Need bounds for the linear transform otherwise get degeneracy in rotation angle
    bds = ((0, -90, 0, -np.pi/2), (360, 90, np.inf, np.pi/2))

    res_lin = least_squares(tweak_scalerot, x0=initLIN, args=(_platecoords, _skycoords, header_wcs), bounds=bds, x_scale='jac')
    header_wcs = return_scalerot(res_lin.x, header_wcs)
    crval = res_lin.x

    res_cubic = least_squares(tweak_all_simult, x0=init_vector, args=(_platecoords, _skycoords, header_wcs), x_scale='jac')
    header_wcs = return_fullwcs(res_cubic.x, header_wcs)

    return header_wcs
