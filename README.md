## GOTO astrometry tools
* `gen_xmatch` - cross-matching tool with catsHTM, implemented from main `gotophoto` repo,   
* `fit_astrom_simult` - robust astrometric tweaks for improving an initial solution    
* `framescore` - QA tool, test version is included in `refit_wcs.py`
* Example scripts using all of the above

### Dependencies:
* catsHTM with local copies of the catalogs you want to cross-match against. Currently set for using Gaia DR2.
* If you aren't on the GOTO or CSC systems, set the environment variable `CATSHTM_PATH` to point to the catalog path.

### Quickstart:
* In the scripts folder, `refit_wcs.py` will apply all the tools in `goto-astromtools` on the frame that will be applied in the pipeline version, and output a new frame with updated WCS and photometry table.

