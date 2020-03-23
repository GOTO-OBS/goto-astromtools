from time import time

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from astropy.table import Table
from multiprocessing import Pool

#### internal modules
from goto_astromtools.crossmatching import gen_xmatch, reduce_density
from goto_astromtools.simult_fit import fit_astrom_simult

#### SQL query depends
import pandas as pd
import psycopg2
from contextlib import contextmanager


def astrom_task(infilepath):
    tick_xmatch = time()
    _platecoords, _skycoords = gen_xmatch(infilepath, prune=True)
    tock_xmatch = time()
    header = fits.getheader(infilepath, 1)
    UT = header["INSTRUME"]
    JD = header["JD"]
    head_wcs = WCS(header)
    resid_before = (head_wcs.all_pix2world(_platecoords, 0) - _skycoords * 180 / np.pi) * 3600

    tick_fit = time()
    new_wcs = fit_astrom_simult(_platecoords, _skycoords, header)
    resid = (new_wcs.all_pix2world(_platecoords, 0) - _skycoords * 180 / np.pi) * 3600
    tock_fit = time()

    filename = infilepath.split("/")[-1]
    pre_med = np.average(np.median(resid_before, axis=0))
    post_med = np.average(np.median(resid, axis=0))
    pre_rms = np.average(np.std(resid_before, axis=0))
    post_rms = np.average(np.std(resid, axis=0))
    pre_chisq = np.sum(resid_before ** 2)
    post_chisq = np.sum(resid ** 2)
    fittime = np.round(tock_fit - tick_fit, 3)
    xmatchtime = np.round(tock_xmatch - tick_xmatch, 3)
    sourcedens = len(resid)

    output = [filename, UT, JD, pre_med, post_med, pre_rms, post_rms, pre_chisq, post_chisq, fittime, xmatchtime,
              sourcedens]
    return output


def amend_paths(inpath, location):
    '''
    Modify the paths obtained from the database to point at CSC or GOTO data dirs
    '''

    if location == "CSC":
        return inpath.replace("work", "storage").replace("data", "storage/goto").replace("reduced", "final")
    if location == "GOTO":
        ### Need to work out the path mapping for GOTO servers.
        return NotImplementedError


@contextmanager
def gotophoto_connection():
    kwargs = dict(
        dbname='gotophoto_photometry',
        host='goto-observatory.warwick.ac.uk',
        port=522,
        user='gotoreadonly',
    )
    with psycopg2.connect(**kwargs) as conn:
        yield conn
        # or you can return conn and cursor if you use psycopg2's cursor.execute() method's to do DB queries, it isn't needed for pd.read_sql() example below
        # with conn.cursor() as cursor:
        #     yield (conn, cursor)
    conn.close()


query = """
        select * from image where
        image_type = 'SCIENCE' and
        exptime=60 and
        jd between 2458839.00000 and 2458841.00000
        """

with gotophoto_connection() as conn:
    images = pd.read_sql(query, conn)

storage_paths = [amend_paths(i, "CSC") for i in images["filepath"]]

results_table = Table(names=("FNAME", "UT", "JD", "MED_BEF", "MED_AFT", "RMS_BEF", "RMS_AFT", "CHI_BEF", "CHI_AFT",
                             "FITTIME", "XMATCHTIME", "SRCDENS"),
                      dtype=("S", "S", "f8", "f8", "f8", "f8", "f8", "f8", "f8", "f8", "f8", "f8"))

if __name__ == '__main__':
    pool = Pool(20)

    for row in pool.map(astrom_task, storage_paths):
        if row is not None:
            results_table.add_row(row)
        else:
            pass

results_table.write("../data/batchfit_results.csv", format="ascii.ecsv", overwrite=True)
