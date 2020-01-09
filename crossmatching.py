import numpy as np
from astropy.io import fits
import catsHTM
from astropy.table import Table
from astropy.time import Time
import astropy.units as u
from kdsphere import KDSphere
from os import environ

try:
    catsHTMpath = environ["CATSHTM_PATH"]
except:
    print("Defaulting to /storage/goto/catalogs/ -- set your environment variable!")
    catsHTMpath = "/storage/goto/catalogs/"


def gen_xmatch(fpath, prune):
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

    if prune:
        ### Gaia recommended quality flags
        astrom_excess_noise_flg = ~np.isnan(cat_table_xm["ExcessNoise"]) & (cat_table_xm["ExcessNoise"] < 1)
        plx_exists_flg = ~np.isnan(cat_table_xm["Plx"])
        nan_cleaned = astrom_excess_noise_flg & plx_exists_flg
        plx_cut_flg = cat_table_xm["Plx"]/cat_table_xm["ErrPlx"] > 5

        astrom_cut_flg = astrom_excess_noise_flg & plx_exists_flg & plx_cut_flg

        # SExtractor cuts
        sat_flag = table_xm["flags"] < 4 # Remove all saturated (and worse) stars from solution
        snr_flag = table_xm["s2n"] > 20
        photom_cut_flg = sat_flag & snr_flag

        #### Reject high proper motion field stars.
        tot_prop_mot = np.sqrt(cat_table_xm["PMRA"]**2 + cat_table_xm["PMDec"]**2)/1000
        int_prop_mot = tot_prop_mot * (Time(header["DATE-MID"]).decimalyear - cat_table_xm["Epoch"])
        pmflg = ~np.isnan(int_prop_mot) & (int_prop_mot < 1)

        flg = astrom_cut_flg & photom_cut_flg & pmflg

        ### Quick sanity check to make sure the field isn't getting too sparse
        avg_density = (1022**2 / (sizex*sizey))*np.sum(flg)

        if avg_density < 80:
            print("Less than 80 sources per tile, caution.")
    else:
        ### set flag to all true.
        flg = np.full((len(cat_table_xm)), True)

    # Write out all relevant numbers here
    _ras = (cat_table_xm["RA"])[flg]
    _decs = (cat_table_xm["Dec"])[flg]
    _skycoords = np.vstack((_ras, _decs)).T
    _xs = (table_xm['x'])[flg]
    _ys = (table_xm['y'])[flg]
    _platecoords = np.vstack((_xs, _ys)).T

    return _platecoords, _skycoords

def reduce_density(platecoords, skycoords, reduce_factor):
    STEPX = 1022
    STEPY = 1022
    SIZEX = 8176
    SIZEY = 6132

    ### Fixed here for convenience
    xcorners = np.arange(0, SIZEX, STEPX)
    ycorners = np.arange(0, SIZEY, STEPY)

    xs, ys = platecoords[:,0], platecoords[:,1]
    ras, decs = skycoords[:,0], skycoords[:,1]

    xiso, yiso = [], []
    raiso, deciso = [], []

    init_source_dens = len(xs) / (SIZEX*SIZEY)

    for i, x in enumerate(xcorners):
        for j, y in enumerate(ycorners):
            mask = (xs > x) & (xs < x + STEPX) & (ys > y) & (ys < y + STEPY)
            srccount = np.sum(mask)

            idxs = np.arange(0, srccount, 1)
            choice_idxs = np.random.choice(idxs, (1, int(np.ceil(srccount/reduce_factor))))

            xiso = np.concatenate((xiso, xs[mask][choice_idxs].flatten()), axis=0)
            yiso = np.concatenate((yiso, ys[mask][choice_idxs].flatten()), axis=0)
            raiso = np.concatenate((raiso, ras[mask][choice_idxs].flatten()), axis=0)
            deciso = np.concatenate((deciso, decs[mask][choice_idxs].flatten()), axis=0)

    platecoord_reduced = np.vstack((xiso, yiso)).T
    skycoord_reduced = np.array((raiso, deciso)).T

    return platecoord_reduced, skycoord_reduced
