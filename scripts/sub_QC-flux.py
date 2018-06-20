
# coding: utf-8

# In[ ]:


#!/usr/bin/env python

import os
import sys
from datetime import datetime
import argparse
import numpy as np
import pandas as pd

from sourcefinder.accessors import open as open_accessor
from sourcefinder.accessors import sourcefinder_image_from_accessor

from astropy.io import fits
from astropy.io.fits.hdu.hdulist import HDUList
from astropy.time import Time


# In[ ]:


def get_configuration():
    """
    Returns a populated configuration
    """
    parser = argparse.ArgumentParser()

    parser.add_argument('--indir', type=str, default="/tmp",
                        help="Input directory for fitsfile.")
    parser.add_argument('--fitsfile', type=str, default="./",
                        help="Target fits file.")

    parser.add_argument('--threshold', type=float, default=5.0,
                        help="RMS Threshold to reject image.")
    parser.add_argument('--outdir', type=str, default="./",
                        help="Desitnation directory.")

    parser.add_argument("--detection", default=10, type=float,
                            help="Detection threshold")
    parser.add_argument("--analysis", default=3, type=float,
                            help="Analysis threshold")

    parser.add_argument("--radius", default=0, type=float,
                            help="Radius of usable portion of image (in pixels)")
    parser.add_argument("--grid", default=64, type=float,
                            help="Background grid segment size")

    parser.add_argument("--reference", default="", type=str,
                            help="Path of reference catalogue used for flux fitting. ")

    return parser.parse_args()


# In[ ]:


def distSquared(p0, p1):

    distance  = np.sqrt((p0[0] - p1[0,:])**2 + (p0[1] - p1[1,:])**2)
    if np.min(distance) < 1.0:
        return np.where(distance == np.min(distance))[0]
    else:
        return None

def pol2cart(rho, phi):
    """
    Polar to Cartesian coordinate conversion, for distance measure around celestial pole.
    """
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return(x, y)


# In[ ]:


def compare_flux(sr, catalog_ras, catalog_decs, catalog_fluxs, catalog_flux_errs):

    x = []
    y = []

    w = []
    sr_indexes = []
    cat_indexes = []


    for i in range(len(sr)):

        sr_x, sr_y = pol2cart(np.abs(90-sr[i].dec.value),
                np.deg2rad(sr[i].ra.value))

        cat_x, cat_y = pol2cart(np.abs(90-catalog_decs),
                np.deg2rad(catalog_ras))

        index = distSquared((sr_x,sr_y),
                   np.array([cat_x, cat_y]))

        if type(index) == np.ndarray:
            flux = catalog_fluxs[index]
            flux_err = catalog_flux_errs[index]

            cat_indexes.append(index)
            sr_indexes.append(i)
            y.append(float(sr[i].flux))
            x.append(float(flux))
            w.append(float(sr[i].flux.error))
        else:
            continue
            
    if len(x) > 2:
        w = np.array(w,dtype=float)
        fit = np.polyfit(x,y,1,w=1./w)
    else:
        fit = [1e9,1e9]

    return fit[0], fit[1]


# In[ ]:


def process(cfg):
    """
    Perform an initial quality control filtering step on the incoming image stream. Images
    which are not rejected are then flux calibrated using a reference catalogue.
    """
    print "running process"

    lofarfrequencyOffset = 0.0
    lofarBW = 195312.5
    
    ref_cat = pd.read_csv(cfg.reference)
    
    fitsimg = fits.open(cfg.indir+cfg.fitsfile)[0]
    
    t = Time(fitsimg.header['DATE-OBS'])
    frq = fitsimg.header['RESTFRQ']
    bw = fitsimg.header['RESTBW']


    # Initial quality condition. 
    if np.nanstd(fitsimg.data[0,0,:,:]) < cfg.threshold:

        # Source find 
        configuration = {
            "back_size_x": cfg.grid,
            "back_size_y": cfg.grid,
            "margin": 0,
            "radius": cfg.radius}

        img_HDU = fits.HDUList(fitsimg)
        imagedata = sourcefinder_image_from_accessor(open_accessor(fits.HDUList(fitsimg),
                                                                   plane=0),
                                                     **configuration)

        sr = imagedata.extract(det=cfg.detection, anl=cfg.analysis,
                               labelled_data=None, labels=[],
                               force_beam=True)

        # Reference catalogue compare
        slope_cor, intercept_cor = compare_flux(sr,
                                       ref_cat["ra"],
                                       ref_cat["decl"],
                                       ref_cat["f_int"],
                                       ref_cat["f_int_err"])

        # Slope set to 1e9 if line fit fails
        if slope_cor < 1e8:
            filename = '%s.fits' % (datetime.fromtimestamp(t.unix).strftime('%Y-%m-%dT%H:%M:%S')+                             "-S"+str(round((frq-lofarfrequencyOffset)/lofarBW,1))+                             "-B"+str(int(np.ceil(bw /lofarBW))))

            fitsimg.data[0,0,:,:] = (fitsimg.data[0,0,:,:]-intercept_cor)/slope_cor
            fitsimg.writeto(cfg.outdir+filename,overwrite=True)
            os.remove(cfg.indir+cfg.fitsfile)
        else:
            os.remove(cfg.indir+cfg.fitsfile)
            return
    else:
        os.remove(cfg.indir+cfg.fitsfile)
        return


# In[ ]:


if __name__ == "__main__":
        cfg = get_configuration()
        
        if cfg.outdir[-1] != "/":
            cfg.outdir = cfg.outdir+"/"

        if cfg.indir[-1] != "/":
            cfg.indir = cfg.indir+"/"

        if not os.path.isdir(cfg.outdir):
                os.mkdir(cfg.outdir)
        
        process(cfg)

