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
from matipo.experiment.plots import SignalPlot, LinePlot

class CPMG(BaseExperiment):
    def setup(self):
        # load the pulse sequence 
        self.seq = Sequence(os.path.join(SEQUENCE_DIR, 'CPMG.py'))
        self.default_par_files=[os.path.join(GLOBALS_DIR, 'shims.yaml')]
        self.log.setLevel('INFO')
        
        self.enable_partialplot=True
        self.enable_runloop=False
        
        # set up the signal and spectrum plots
        self.plots = {
            'echo_int': LinePlot(figure_opts=dict(title='Integrated Echos')),
            'echo_sum': SignalPlot(figure_opts=dict(title='Summed Echos'))
        }
        
        # Adding user inputs to modify parameters that are already defined in the pulse sequence
        # These parameters will be automatically updated when the inputs are modified
        self.inputs = auto_inputs(self.seq, {
            'tx_chan': 0,
            'rx_chan': 0,
            'n_scans': 2,
            'n_echo': 4000,
            't_echo': 500*Unit('us'),
            't_echo_offset': 0*Unit('us'),
            't_end': 1.0*Unit('s'),
            't_dw': 1.0*Unit('us'),
            'n_samples': 100,
        })
        
        # Adding display names for the user inputs
        self.inputs['n_scans'].name = 'Scans'
        self.inputs['t_end'].name = 'End Time (s)'
        self.inputs['t_dw'].name = 'Dwell Time (µs)'
        self.inputs['n_samples'].name = 'Samples per Echo'
        self.inputs['t_echo'].name = 'Echo Time (μs)'
        
    
    # Overriding Pulse Sequence Parameters
    def update_par(self):
        # load default parameter files for channel
        self.seq.loadpar(os.path.join(GLOBALS_DIR, f'frequency_tx{self.seq.par.tx_chan}.yaml'))
        self.seq.loadpar(os.path.join(GLOBALS_DIR, f'hardpulse_90_tx{self.seq.par.tx_chan}.yaml'))
        self.seq.loadpar(os.path.join(GLOBALS_DIR, f'hardpulse_180_tx{self.seq.par.tx_chan}.yaml'))

    # update plots with data
    async def update_plots(self, final, *args, **kwargs):
        data = await self.seq.fetch_data()
        if data is None:
            return
        
        data = np.reshape(data, (-1, self.seq.par.n_samples))
        sum_echo = np.sum(data, axis=0) # sum of all echos, sample by sample
        int_echo = np.mean(data, axis=1) # integrated echos, each echo has all its samples averaged

        # Auto phase correct
        phi = np.angle(np.sum(int_echo*np.abs(int_echo)))
        int_echo = int_echo * np.exp(-1j*phi)
        t_echo = self.seq.par.t_echo*np.arange(self.seq.par.n_echo)
        
        self.plots['echo_int'].update(dict(
            Real=dict(x=t_echo, y=int_echo.real),
            Imag=dict(x=t_echo, y=int_echo.imag)))
        
        self.plots['echo_sum'].update(
            seqdata=sum_echo,
            t_dw=self.seq.par.t_dw)

        if final:
            SNR = np.mean(int_echo.real[:2]).item() / np.sqrt(np.mean(int_echo.imag[self.seq.par.n_echo//2:] * int_echo.imag[self.seq.par.n_echo//2:])).item()
            self.log.info('SNR: %.2f', SNR)
            

def main(doc=None):
    cpmg = CPMG()
    return cpmg()
