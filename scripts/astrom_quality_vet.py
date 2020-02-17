import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.table import Table

from goto_astromtools.crossmatching import gen_xmatch
from goto_astromtools.simult_fit import fit_astrom_simult

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("filename", type=str, help="path to the fits file to be analysed")
parser.add_argument("--showplots", type=bool, default=True, help="show plots instead of saving them")

args = parser.parse_args()


def generate_diagnostic_plots(framepath, showplots=True):
    # Get Gaia DR2 astrometry cat, without quality cuts to try and replicate current setup
    platecoords, skycoords = gen_xmatch(framepath, prune=False)

    xs, ys = platecoords[:, 0], platecoords[:, 1]

    header = fits.getheader(framepath, 1)

    sizex, sizey = header["NAXIS1"], header["NAXIS2"]
    new_wcs = fit_astrom_simult(platecoords, skycoords, header)
    resid = (new_wcs.all_pix2world(platecoords, 0) - skycoords * 180 / np.pi) * 3600

    stepx = 1022
    stepy = 1022

    xcorners = np.arange(0, sizex, stepx)
    ycorners = np.arange(0, sizey, stepy)

    rms_matrix = np.zeros((len(xcorners), len(ycorners)))
    median_matrix = np.zeros((len(xcorners), len(ycorners)))
    nsources = np.zeros((len(xcorners), len(ycorners)))

    for i, x in enumerate(xcorners):
        for j, y in enumerate(ycorners):
            mask = (xs > x) & (xs < x + stepx) & (ys > y) & (ys < y + stepy)

            rms_matrix[i][j] = np.std(resid[mask])
            median_matrix[i][j] = np.median(resid[mask])
            nsources[i][j] = np.sum(mask)

            if np.sum(mask) < 10:
                print("Nearly empty tile!")

    xpl = xcorners + stepx / 2
    ypl = ycorners + stepy / 2

    # Residual plot
    axmm = 3 * np.max(np.std(resid, axis=0))  # arcsecs to fix plot limit
    binno = 50

    fixbin = np.linspace(-axmm, axmm, binno)

    fig, ax = plt.subplots(2, 2)
    plt.subplots_adjust(hspace=0, wspace=0)
    plt.minorticks_on()

    ax[0][1].remove()

    ax[0][0].hist(resid[:, 0], bins=fixbin, color='grey', density=True)
    ax[0][0].set_xlim(-axmm, axmm)
    ax[0][0].set_xticklabels([])

    ax[1][1].hist(resid[:, 1], bins=fixbin, color='grey', orientation='horizontal')
    ax[1][1].set_ylim(-axmm, axmm)
    ax[1][1].set_yticklabels([])

    ax[1][0].hist2d(resid[:, 0], resid[:, 1], bins=[fixbin, fixbin], cmap='Greys')

    ax[0][0].axvline(np.median(resid[:, 0]), ls='--', c='k')
    ax[1][1].axhline(np.median(resid[:, 1]), ls='--', c='k')
    ax[1][0].axhline(np.median(resid[:, 1]), ls='--', c='k')
    ax[1][0].axvline(np.median(resid[:, 0]), ls='--', c='k')
    ax[1][0].scatter(*np.median(resid, axis=0), c='r', zorder=20)

    ax[1][0].set_xlim(-axmm, axmm)
    ax[1][0].set_ylim(-axmm, axmm)
    ax[1][0].set_xlabel("resid. RA")
    ax[1][0].set_ylabel("resid. DEC")

    plt.tight_layout()

    if showplots:
        plt.show()
    else:
        plt.savefig("resid_histogram.png")

    phot_table = Table(fits.getdata(framepath, 3))
    src_x, src_y = phot_table["x"], phot_table["y"]

    raw_srcdens = np.zeros((len(xcorners), len(ycorners)))

    for i, x in enumerate(xcorners):
        for j, y in enumerate(ycorners):
            mask = (src_x > x) & (src_x < x + stepx) & (src_y > y) & (src_y < y + stepy)

            raw_srcdens[i][j] = np.sum(mask)

    perc_xmatch = (nsources / raw_srcdens)

    # Contour plots
    fig, ax = plt.subplots(1, 4, sharey=True, figsize=(10, 10), dpi=120)
    plt.minorticks_on()

    ax[0].contourf(xpl, ypl, rms_matrix.T)
    ax[0].set_title("RMS astrom. noise")

    conts = np.linspace(0, np.max(raw_srcdens), 20)

    ax[1].contourf(xpl, ypl, nsources.T, conts, cmap='plasma')
    ax[1].set_title("x-match density")

    ax[2].contourf(xpl, ypl, raw_srcdens.T, conts, cmap='plasma')
    ax[2].set_title("source density")

    ax[3].contourf(xpl, ypl, perc_xmatch.T, np.linspace(0, 1, 10), cmap='plasma')
    ax[3].set_title("percentage_xmatch")

    for a in ax:
        a.set_aspect("equal")

    plt.tight_layout()

    if showplots:
        plt.show()
    else:
        plt.savefig("framedensities.png")

    # Print a summary to the command line
    print(3 * "\n")
    print("### Summary statistics\n")
    print("Filename: {}".format(framepath))
    print("Median astrometric RMS: {:>10.3f}\"".format(np.median(rms_matrix)))
    print("Worst RMS tile: {:18.3f}\"".format(np.max(rms_matrix)))
    print("Global cross-matched dets: {:6.1f}%".format(100 * np.sum(perc_xmatch / np.product(np.shape(perc_xmatch)))))
    print("Worst x-match tile: {:13.1f}%".format(100 * np.min(perc_xmatch)))
    flagstr = "GOOD" if header["QUALITY"] < 128 else "BAD"
    print("Pipeline quality status:     {}".format(flagstr))


generate_diagnostic_plots(args.filename, args.showplots)
