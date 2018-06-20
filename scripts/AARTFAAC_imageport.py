
# coding: utf-8

# In[ ]:


import argparse
import Queue
from subprocess import Popen, PIPE
import os
import sys
import numpy as np
import pandas as pd
import random

from twisted.internet.protocol import Protocol, ReconnectingClientFactory
from twisted.internet import task
from twisted.internet import defer
from twisted.internet import reactor
from twisted.internet import threads
from twisted.internet import protocol
from twisted.python import log

from sourcefinder.accessors import open as open_accessor
from sourcefinder.accessors import sourcefinder_image_from_accessor



from astropy.io import fits
from astropy.io.fits.hdu.hdulist import HDUList
from astropy.time import Time

from datetime import datetime
from atv.streamprotocol import StreamProtocol


# In[ ]:


FILE_SIZE = 4204800
FPS = 25

# lock = threading.Lock()


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
        self.reference = cfg.reference
        self.lofarfrequencyOffset = 0.0
        self.lofarBW = 195312.5
        self.namelen = 12
        
        if cfg.outdir[-1] == "/":
            self.outdir = cfg.outdir
        else:
            self.outdir = cfg.outdir+"/"

        if not os.path.isdir(self.outdir):
                os.mkdir(self.outdir)

        self.encode_task = task.LoopingCall(self.encode)
        self.encode_task.start(1.0/FPS)
        
    def gen_hex_code(self):
        return ''.join([random.choice('0123456789ABCDEF') for x in range(self.namelen)])

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
            tmpfilename = self.gen_hex_code()+".fits"
            fitsimg.writeto("/tmp/"+tmpfilename)
            self.pqueue.put((t.unix, frq, bw, tmpfilename))

        else:
            self.pqueue.put((t.unix, frq, bw, None))


    def encode(self):
        """
        Save fits file
        """
        
        if self.pqueue.qsize() < 3:
        # why queue more than 3? 
#             log.msg("Queue size %i" % self.pqueue.qsize())
            return

        t, FRQ, BW, tmpfilename = self.pqueue.get(block=False)
        self.num_processing -= 1
        
        time = Time(t, format='unix').isot
        time = datetime.fromtimestamp(t).strftime('%Y-%m-%dT%H:%M:%S')
        
        filename = '%s.fits' % (time+                                 "-S"+str(round((FRQ-self.lofarfrequencyOffset)/ self.lofarBW,1))+                                 "-B"+str(int(np.ceil(BW /self.lofarBW)))) 
        
        if bool(tmpfilename):
            if os.path.isfile(self.outdir+filename):
                log.msg("Processed %s [%i], not saved, exists." % (filename, self.num_processing))
                return 
            
            Popen(['python','/afhome/kuiack/scratchpad/scripts/sub_QC-flux.py',
                             '--indir','/tmp/','--fitsfile',tmpfilename,
                             '--threshold',str(self.threshold),
                             '--detection',str(self.detection),
                             '--radius',str(self.radius),
                             '--outdir',self.outdir,
                             '--reference',self.reference])
            
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
#             threads.deferToThread(reactor.spawnProcess, PassAll, self.process(fitsimg))
#             reactor.spawnProcess(PassAll, self.process(fitsimg))
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

