import numpy as np
from astropy.io import fits
import catsHTM
from astropy.table import Table
from astropy.time import Time
import astropy.units as u
from kdsphere import KDSphere

catsHTMpath = "/storage/goto/catalogs/"

def gen_xmatch(fpath):
    ''' Given a FITS file, cross-match the WCS position with a catalogs
        using the catsHTM module. Optionally, prune the catalog of 'bad'
        stars before performing the cross-match.

        fpath -- path to the file that needs crossmatching

        Returns:
        _platecoords - x,y positions of sources from table that were x-matched
        _skycoords - corresponding RA, DEC positionsself.
        Both return as 2xN arrays.
    '''
    hdul = fits.open(fpath, mode='readonly')
    header = hdul[1].header
    imagedata = hdul[1].data
    photom_table = Table(hdul[3].data)
    hdul.close()

    objects = photom_table[photom_table['MAGERR_BEST'] < 0.1]
    sizex = header['NAXIS1']
    sizey = header['NAXIS2']

    ra_c = header['cra'] * u.degree
    ra_c_rad = ra_c.to(u.rad).value
    dec_c = header['cdec'] * u.degree
    dec_c_rad = dec_c.to(u.rad).value

    field_radius = 2 * u.degree
    field_radius_as = field_radius.to(u.arcsec).value

    ## Code from krzul's astrometry checker
    cat_data, col_names, _ = catsHTM.cone_search('GAIADR2', ra_c_rad, dec_c_rad, field_radius_as, catalogs_dir=catsHTMpath)
    cat_data = cat_data[cat_data[:, 15] < 21]  # Mag_G<21
    cat_table = Table(data=cat_data, names=col_names)

    cat_coo = cat_data[:, :2]
    tree = KDSphere(cat_coo)
    det_coo_arr = objects["ra", "dec"].as_array()
    det_coo = det_coo_arr.view((det_coo_arr.dtype[0], 2))
    dist_match_rad = (6*u.arcsec).to(u.radian).value
            # KDSphere requires radians input and gives radians output
    nn_dists, nn_idxs = tree.query(det_coo * np.pi / 180, distance_upper_bound=dist_match_rad)
    mask = np.isfinite(nn_dists)  # select only finite distances (i.e true matches within search radius)

    nn_idxs_xm = nn_idxs[mask]
    table_xm = objects[mask]
    cat_table_xm = cat_table[nn_idxs_xm]

    tot_prop_mot = np.sqrt(cat_table_xm["PMRA"]**2 + cat_table_xm["PMDec"]**2)/1000
    int_prop_mot = tot_prop_mot * (Time(header["DATE-MID"]).decimalyear - cat_table_xm["Epoch"])

    pmflg = ~np.isnan(int_prop_mot) & (int_prop_mot < 1)
    print("Proper motion cut: %s" % np.sum(~pmflg))
    goodflg = (table_xm["s2n"] > 20)
    print("Star quality cut: %s" % np.sum(~goodflg))
    flg = pmflg & goodflg
    print("%s of %s sources included" % (np.sum(flg), len(flg)))

    # Write out all relevant numbers here
    _ras = (cat_table_xm["RA"])[flg]
    _decs = (cat_table_xm["Dec"])[flg]
    _skycoords = np.vstack((_ras, _decs)).T
    _xs = (table_xm['x'])[flg]
    _ys = (table_xm['y'])[flg]
    _platecoords = np.vstack((_xs, _ys)).T

    return _platecoords, _skycoords
