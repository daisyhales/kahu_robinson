import panel as pn
import sys
import os
import yaml
import numpy as np
from bokeh.palettes import Viridis
import asyncio

from matipo import Sequence, Unit, CONFIG, DATA_DIR, SEQUENCE_DIR, GLOBALS_DIR
from matipo.util.decimation import decimate
from matipo.util.fft import fft_reconstruction
from matipo.util.etl import deinterlace
from matipo.util import ilt

from matipo.experiment.base_experiment import BaseExperiment, auto_inputs, PlotInterface
from matipo.experiment.models import PLOT_COLORS, SITickFormatter, SIHoverFormatter
from matipo.experiment.plots import SignalPlot, SpectrumPlot

class FID(BaseExperiment):
    def setup(self):
        # load the pulse sequence 
        self.seq = Sequence(os.path.join(SEQUENCE_DIR, 'FID.py'))
        self.default_par_files=[os.path.join(GLOBALS_DIR, 'shims.yaml')]
        
        self.enable_partialplot=True
        self.enable_runloop=True
        
        # set up the signal and spectrum plots
        self.plots = {
            'signal': SignalPlot(),
            'spectrum': SpectrumPlot()
        }
        
        # Adding user inputs to modify parameters that are already defined in the pulse sequence
        # These parameters will be automatically updated when the inputs are modified
        self.inputs = auto_inputs(self.seq, {
            'tx_chan': 0,
            'rx_chan': 0,
            'n_scans': 1,
            't_end': 0.5*Unit('s'),
            't_dw': 10.0*Unit('us'),
            'n_samples': 1000, 
            't_acqdelay': 50*Unit('us')
        })
        
        # Adding display names for the user inputs
        self.inputs['n_scans'].name = 'Scans'
        self.inputs['t_end'].name = 'End Time (s)'
        self.inputs['t_dw'].name = 'Dwell Time (µs)'
        self.inputs['n_samples'].name = 'Samples'
        self.inputs['t_acqdelay'].name = 'Acq. Delay (μs)'
        
    
    # Overriding Pulse Sequence Parameters
    def update_par(self):
        # load default parameter files for channel
        self.seq.loadpar(os.path.join(GLOBALS_DIR, f'frequency_tx{self.seq.par.tx_chan}.yaml'))
        self.seq.loadpar(os.path.join(GLOBALS_DIR, f'hardpulse_90_tx{self.seq.par.tx_chan}.yaml'))

    # update plots with data
    async def update_plots(self, *args, **kwargs):
        await self.seq.fetch_data()
        self.plots['signal'].update(seqdata=self.seq.data, t_dw=self.seq.par.t_dw)
        self.plots['spectrum'].update(seqdata=self.seq.data, t_dw=self.seq.par.t_dw, t0=0)


def main(doc=None):
    fid = FID()
    return fid()
