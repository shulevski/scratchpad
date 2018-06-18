
# coding: utf-8

# In[ ]:


from twisted.internet.protocol import Protocol, ReconnectingClientFactory
from twisted.internet import task
from twisted.internet import defer
from twisted.internet import reactor
from twisted.internet import threads
from twisted.python import log

from sourcefinder.accessors import open as open_accessor
from sourcefinder.accessors import sourcefinder_image_from_accessor
from sourcefinder.accessors import writefits as tkp_writefits
from sourcefinder.utility.cli import parse_monitoringlist_positions
from sourcefinder.utils import generate_result_maps

from astropy.io.fits.hdu.hdulist import HDUList

import argparse
import Queue
import os
import sys
import numpy as np
import pandas as pd

import threading 

from datetime import datetime

from astropy.io import fits
from astropy.time import Time
# from astropy.visualization import ZScaleInterval

# from PIL import Image
# from PIL import ImageFont
# from PIL import ImageDraw
# from matplotlib import cm

from atv.streamprotocol import StreamProtocol
# from atv.rms import sigmaclip
# from atv.constellations import Constellations


# In[ ]:


FILE_SIZE = 4204800
# IMAGE_RES = 1024
FPS = 25

lock = threading.Lock()
# SOURCES = ['Cas.A', 'Cyg.A', 'Tau.A', 'Vir.A', 'Sun', 'Moon']
# CONSTELLATIONS = ['Ursa Minor']
# CMD = ["ffmpeg",
#        # for ffmpeg always first set input then output

#        # silent audio
#        '-f', 'lavfi',
#        '-i', 'anullsrc=channel_layout=stereo:sample_rate=44100',

#        # image
#        '-re',
#        '-f', 'rawvideo',           # probably required for reading from stdin
#        '-s', '1024x1024',          # should match image size
#        '-pix_fmt', 'rgba',
#        '-i', '-',                  # read from stdin

#        # encoding settings
#        "-r", str(FPS),             # the framerate
#        "-vcodec", "libx264",       # probably required for flv & rtmp
#        "-preset", "ultrafast",     # the encoding quality preset
#        "-g", "20",
#        "-codec:a", "libmp3lame",   # mp3 for audio
#        "-ar", "44100",             # 44k audio rate
#        "-threads", "6",
#        "-bufsize", "512k",
#        "-f", "flv",                # required for rtmp
#        ]


# In[ ]:


def get_configuration():
    """
    Returns a populated configuration
    """
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--host', type=str, default='localhost',
                        help="host (tcp ip4)")
    parser.add_argument('--port', type=int, default=9000,
                        help="port")
    parser.add_argument('--threshold', type=float, default=5.0,
                        help="RMS Threshold to reject image.")
    parser.add_argument('--outdir', type=str, default="./",
                        help="Desitnation directory.")
    parser.add_argument('--nproc', type=int, default=3,
                        help="Max number of processes.")
    
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

            continue


    if len(x) > 2:
        w = np.array(w,dtype=float)
        with lock:
            fit = np.polyfit(x,y,1,w=1./w)
    else:
        fit = [1e9,1e9]

    return fit[0], fit[1]


# In[ ]:


class Stream(Protocol):
    def __init__(self, cfg):
        self.num_processing = 0
        self.b1 = np.zeros(FILE_SIZE, dtype=np.uint8)
        self.b2 = np.zeros(FILE_SIZE, dtype=np.uint8)
        self.bb1 = np.getbuffer(self.b1)
        self.bb2 = np.getbuffer(self.b2)
        self.bytes_received = 0
        self.pqueue = Queue.PriorityQueue()
        self.imgdata = None
        self.nproc = cfg.nproc
        self.threshold = cfg.threshold
        self.grid = cfg.grid
        self.radius = cfg.radius
        self.detection = cfg.detection
        self.analysis = cfg.analysis
        self.ref_cat = pd.read_csv(cfg.reference)
        self.lofarfrequencyOffset = 0.0
        self.lofarBW = 195312.5
        
        if cfg.outdir[-1] == "/":
            self.outdir = cfg.outdir
        else:
            self.outdir = cfg.outdir+"/"

        if not os.path.isdir(self.outdir):
                os.mkdir(self.outdir)

        self.encode_task = task.LoopingCall(self.encode)
        self.encode_task.start(1.0/FPS)


    def process(self, fitsimg):
        """
        Perform an initial quality control filtering step on the incoming image stream. Images
        which are not rejected are then flux calibrated using a reference catalogue.
        """
        if self.num_processing >= self.nproc:
            log.msg("Skipping %s [%i]" % (fitsimg.header['DATE-OBS'], self.num_processing))
            return

        t = Time(fitsimg.header['DATE-OBS'])
        frq = fitsimg.header['RESTFRQ']
        bw = fitsimg.header['RESTBW']
        
        self.num_processing += 1

        # Initial quality condition. 
        if np.nanstd(fitsimg.data[0,0,:,:]) < self.threshold:

            # Source find 
            configuration = {
                "back_size_x": self.grid,
                "back_size_y": self.grid,
                "margin": 0,
                "radius": self.radius}
            
            img_HDU = fits.HDUList(fitsimg)
            imagedata = sourcefinder_image_from_accessor(open_accessor(img_HDU, plane=0),**configuration)
            
            with lock:
                sr = imagedata.extract(
                    det=self.detection, anl=self.analysis,
                    labelled_data=None, labels=[],
                    force_beam=True)
            
            # Reference catalogue compare
            slope_cor, intercept_cor = compare_flux(sr,
                                           self.ref_cat["ra"],
                                           self.ref_cat["decl"],
                                           self.ref_cat["f_int"],
                                           self.ref_cat["f_int_err"])
            # Set to 1e9 if fit fails
            if slope_cor < 1e8:
                img_HDU[0].data[0,0,:,:] = (img_HDU[0].data[0,0,:,:]-intercept_cor)/slope_cor
                self.pqueue.put((t.unix, frq, bw, img_HDU))
                
            else:
                self.pqueue.put((t.unix, frq, bw, None))
                
        else:
            self.pqueue.put((t.unix, frq, bw, None))


    def encode(self):
        """
        Save fits file
        """
        
        if self.pqueue.qsize() < 1:
        # why queue more than 3? 
#             log.msg("Queue size %i" % self.pqueue.qsize())
            return

        t, FRQ, BW, imgdata = self.pqueue.get(block=False)
        self.num_processing -= 1
        self.imgdata = imgdata
        
#         time = Time(t, format='unix').isot
        time = datetime.fromtimestamp(t).strftime('%Y-%m-%dT%H:%M:%S')
        
        filename = '%s.fits' % (time+                                 "-S"+str(round((FRQ-self.lofarfrequencyOffset)/ self.lofarBW,1))+                                 "-B"+str(int(np.ceil(BW /self.lofarBW)))) 
        
        if bool(imgdata):
            if os.path.isfile(self.outdir+filename):
                log.msg("Processed %s [%i], not saved, exists." % (filename, self.num_processing))
                return 
            
            imgdata.writeto(self.outdir+filename)
            log.msg("Processed %s [%i], saved." % (filename, self.num_processing))
        else:
            log.msg("Processed %s [%i], not saved, QC rejected." % (filename, self.num_processing))
            



    def dataReceived(self, data):
        n = min(FILE_SIZE - self.bytes_received, len(data))
        self.bb1[self.bytes_received:self.bytes_received+n] = data[:n]

        if self.bytes_received+n >= FILE_SIZE:
            # process on another thread
            fitsimg = fits.PrimaryHDU().fromstring(self.bb1)
            #self.process(fitsimg) # (for debugging)
            threads.deferToThread(self.process, fitsimg) 
            # swap buffers
            self.bb1, self.bb2 = self.bb2, self.bb1
            # copy remaining data in current buffer
            self.bytes_received = len(data) - n
            self.bb1[:self.bytes_received] = data[n:]
        else:
            self.bytes_received += n


# In[ ]:


class StreamFactory(ReconnectingClientFactory):
    def __init__(self, cfg):
        self.cfg = cfg


    def startedConnecting(self, connector):
        print('Started to connect.')


    def buildProtocol(self, addr):
        print('Connected.')
        self.resetDelay()
        return Stream(self.cfg)


    def clientConnectionLost(self, connector, reason):
        print('Lost connection.  Reason:', reason)
        ReconnectingClientFactory.clientConnectionLost(self, connector, reason)


    def clientConnectionFailed(self, connector, reason):
        print('Connection failed. Reason:', reason)
        ReconnectingClientFactory.clientConnectionFailed(self, connector,
                                                         reason)


# In[ ]:


if __name__ == "__main__":
    cfg = get_configuration()
    log.startLogging(sys.stdout)
    log.msg('%s' % (str(cfg)))
    reactor.connectTCP(cfg.host, cfg.port, StreamFactory(cfg))
    reactor.run()

