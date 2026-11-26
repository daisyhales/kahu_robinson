import panel as pn
import sys
import os
import yaml
import numpy as np
import scipy as sp
from bokeh.palettes import Viridis
import asyncio
import itertools

from bokeh.plotting import figure, ColumnDataSource
from bokeh.models import HoverTool, Circle, Span, Label
from bokeh.palettes import Category10

from matipo import Sequence, Unit, CONFIG, DATA_DIR, SEQUENCE_DIR, GLOBALS_DIR
from matipo.util.decimation import decimate
from matipo.util.fft import get_freq_spectrum, fft_reconstruction
from matipo.util.etl import deinterlace
from matipo.util import ilt

from matipo.experiment.base_experiment import BaseExperiment, auto_inputs, PlotInterface
from matipo.experiment.models import PLOT_COLORS, SITickFormatter, SIHoverFormatter
from matipo.experiment.plots import SignalPlot, SpectrumPlot, ComplexDataLinePlot, WobblePlot
from matipo.experiment.line_plot import LinePlot

class FindFrequencyPlot(ComplexDataLinePlot):
    def __init__(self, **kwargs):
        # override to set title and axis labels
        figure_opts = dict(
            title="Processed Spectrum",
            x_axis_label="Relative Frequency (Hz)",
            y_axis_label="Spectral Density")
        if 'figure_opts' in kwargs:
            figure_opts.update(kwargs['figure_opts'])
        kwargs['figure_opts'] = figure_opts
        kwargs['show_magnitude'] = True
        kwargs['show_complex'] = False
        super().__init__(**kwargs)
        self._freq = None
        self._freq_label = Label(x=10, y=0, y_offset=-20, x_units='screen', y_units='screen', text='')
        self.fig.add_layout(self._freq_label)
        self.fig.js_link('inner_height', self._freq_label, 'y')  # align top
        self._freq_span = Span(location=0, dimension='height', line_dash='dashed', line_alpha=0.0)
        self.fig.add_layout(self._freq_span)
    
    def update(self, seqdata, t_dw, center_freq=0, apodization=0):
        """Update the plot with sequence data and dwell time"""
        t = t_dw*np.arange(len(seqdata))
        window = np.exp(-np.pi*apodization*t)
        freq, spectrum = get_freq_spectrum(window*seqdata, t_dw)
        spectrum_mag = np.abs(spectrum)
        peak_index = np.argmax(spectrum_mag)
        offset_freq = freq[peak_index]
        self._freq = center_freq + offset_freq
        self._freq_label.text = '%0.6f MHz' % (1e-6*self._freq)
        self._freq_span.location = offset_freq
        self._freq_span.line_alpha = 1.0
        super().update(freq, spectrum)

    @property
    def frequency(self):
        return self._freq


class FindFrequency(BaseExperiment):
    def setup(self):
        # self.log.setLevel('DEBUG')
        self.enable_partialplot=True
        self.default_par_files=[os.path.join(GLOBALS_DIR, 'shims.yaml')]
        
        # load the pulse sequence 
        self.seq = Sequence(os.path.join(SEQUENCE_DIR, 'find_freq.py'))
        
        # set up the signal and spectrum plots
        self.plots = {
            'signal': SignalPlot(),
            'findfreq': FindFrequencyPlot()
        }
        
        # Adding user inputs to modify parameters that are already defined in the pulse sequence
        # These parameters will be automatically updated when the inputs are modified
        self.inputs = auto_inputs(self.seq, {
            'tx_chan': 0,
            'rx_chan': 0,
            'n_scans': 4,
            't_end': 0.1*Unit('s'),
            't_acqdelay': 100*Unit('us'),
            'a_90': 1.0, # use max amplitude short pulse for maximum bandwidth excitation
            't_90': 5*Unit('us'),
            't_dw': 0.2*Unit('us'), # use minimum dwell time for max bandwidth acquisition
            'n_samples': 10000,
            'apodization': 10000*Unit('Hz')
        })

        # set bounds
        self.inputs['apodization'].start=0
        
        # Adding display names for the user inputs
        self.inputs['n_scans'].name = 'Scans'
        self.inputs['t_end'].name = 'End Time (s)'
        self.inputs['t_acqdelay'].name = 'Acq. Delay (μs)'

    def update_par(self):
        # load appropriate frequency file
        self.seq.loadpar(os.path.join(GLOBALS_DIR, f'frequency_tx{self.seq.par.tx_chan}.yaml'))

    async def update_plots(self, final=False, *args, **kwargs):
        self.plots['signal'].update(seqdata=self.seq.data, t_dw=self.seq.par.t_dw)
        self.plots['findfreq'].update(seqdata=self.seq.data, t_dw=self.seq.par.t_dw, center_freq=self.seq.par.f,
                                      apodization=(self.inputs['apodization'].value*self.inputs['apodization'].unit).m_as('Hz'))
        if final:
            freq = float(self.plots['findfreq'].frequency)
            with open(os.path.join(GLOBALS_DIR, f'frequency_tx{self.seq.par.tx_chan}.yaml'), 'r') as file:
                freq_data = yaml.full_load(file)
            freq_data['f'] = freq
            
            with open(os.path.join(GLOBALS_DIR, f'frequency_tx{self.seq.par.tx_chan}.yaml'), 'w') as file:
                yaml.dump(freq_data, file, default_flow_style=False)


class SweepPlot(PlotInterface):
    def __init__(self, figure_opts={}):
        
        self.source = None
        
        _figure_opts = dict(
            title="Reflected Signal",
            x_axis_label="frequency (Hz)",
            y_axis_label="dB",
            y_range=(-100, 0),
            toolbar_location="above",
            tools='pan,wheel_zoom,box_zoom,crosshair,reset,save',
            sizing_mode="stretch_both",
            min_height=400,
            min_width=400
        )
        _figure_opts.update(figure_opts) # allow options to be overriden
        

        self.fig = figure(**_figure_opts)
        self.fig.axis.formatter = SITickFormatter()
        self.fig.toolbar.logo = None

    def update(self, seqdata, f, f_bw, f_mark, n_sweep, n_samples):
        ys = np.abs(np.mean(seqdata.reshape(-1, n_sweep, n_samples), axis=2))
        # ys = np.abs(seqdata.reshape(-1, n_sweep*n_samples))
        ys = 20*np.log10(ys/np.max(ys)) # convert to dB relative to largest value
        x = np.linspace(f - f_bw/2, f + f_bw/2, ys.shape[1], endpoint=True)
        self.state = dict(xs=[x for i in range(len(ys))], ys=[y for y in ys], f_mark=f_mark)
    
    @property
    def state(self):
        data = {'f_mark': self.freq_mark.location}
        data.update(self.source.data)
        return data
    
    @state.setter
    def state(self, data):
        if self.source is None:
            color_cycle = itertools.cycle(Category10[10])
            self.colors=[next(color_cycle) for i in range(len(data['ys']))]
            self.labels=[f'Channel {i}' for i in range(len(data['ys']))]
            self.source = ColumnDataSource(dict(
                xs=data['xs'], 
                ys=data['ys'], 
                colors=self.colors,
                labels=self.labels
            ))
            self.fig.multi_line(xs='xs', ys='ys', source=self.source, color='colors', legend_field='labels')
            self.freq_mark = Span(location=data['f_mark'], dimension='height', line_dash='dashed', line_alpha=1.0)
            self.fig.add_layout(self.freq_mark)
        else:
            self.source.data = dict(xs=data['xs'], ys=data['ys'], colors=self.colors, labels=self.labels)
            self.freq_mark.location = data['f_mark']
        
    def __call__(self):
        return self.fig

class TuneMatch(BaseExperiment):
    def setup(self):
        self.log.info("trying to run")
        self.seq = Sequence(SEQUENCE_DIR+'sweep.py')

        self.default_par_files=[]
        self.enable_runloop=True
        self.plots = {
            'wobble': SweepPlot()
        }
        
        self.inputs = auto_inputs(self.seq, {
            'tx_chan': 0,
            'f_bw': 1.0*Unit('MHz'),
            'a_sweep': 1.0,
        })
        
        # set custom label
        self.inputs['f_bw'].name = 'Bandwidth (MHz)'
        
        self.f_mark = 0

    def update_par(self):
        # load appropriate frequency file
        self.seq.loadpar(os.path.join(GLOBALS_DIR, f'frequency_tx{self.seq.par.tx_chan}.yaml'))
        
        self.seq.setpar(
            t_dw=4e-6,
            n_samples=16,
            n_sweep=1001
        )
        
        # if the frequency range would overlap 0, shift the centre frequency to avoid negative frequencies
        self.f_mark = self.seq.par.f
        if self.seq.par.f < self.seq.par.f_bw/2:
            self.seq.setpar(f=self.seq.par.f_bw/2)
    
    async def update_plots(self, *args, **kwargs):
        await self.seq.fetch_data()
        self.plots['wobble'].update(
            seqdata=self.seq.data,
            f=self.seq.par.f, 
            f_bw=self.seq.par.f_bw,
            f_mark=self.f_mark,
            n_sweep=self.seq.par.n_sweep,
            n_samples=self.seq.par.n_samples)


class NoiseCheck(BaseExperiment):
    def setup(self):
        self.default_par_files=[os.path.join(GLOBALS_DIR, 'shims.yaml')]
        self.seq = Sequence(SEQUENCE_DIR+'noise_single.py')
        self.enable_runloop=True
        self.plots = {
            'signal': SignalPlot(show_magnitude=False),
            'spectrum': SpectrumPlot(show_magnitude=True, show_complex=False)
        }
        
        self.inputs = auto_inputs(self.seq, {
            'tx_chan': 0, # only used to select the frequency file
            'rx_chan': 0,
            't_dw': 1.0*Unit('us'),
            'n_samples': 1000
        })

        # set bounds since tx_chan isn't a real parameter
        self.inputs['tx_chan'].start = 0
        self.inputs['tx_chan'].end = 1
        self.inputs['tx_chan'].name = "tx_chan (for centre freq)"

    def update_par(self):
        # load appropriate frequency file
        self.seq.loadpar(os.path.join(GLOBALS_DIR, f'frequency_tx{self.inputs['tx_chan'].value}.yaml'))

    # update plots with data
    async def update_plots(self, *args, **kwargs):
        data = await self.seq.fetch_data()
        RMS = np.sqrt(np.mean(np.abs(data)**2))
        data_windowed = data*sp.signal.windows.hamming(len(data))
        freq, fft = get_freq_spectrum(data_windowed, self.seq.par.t_dw)
        fft_pctile = np.percentile(np.abs(fft), [99])[0]
        self.log.info('Signal RMS: %.2f µV, Spectrum 99th percentile: %.2f nV/sqrt(Hz)', 1e6*RMS, 1e9*fft_pctile)
        self.plots['signal'].update(seqdata=data, t_dw=self.seq.par.t_dw)
        self.plots['spectrum'].update(seqdata=data_windowed, t_dw=self.seq.par.t_dw, t0=0)


class PulseCalibrationPlot(LinePlot):
    def __init__(self, figure_opts={}, **kwargs):
        _figure_opts = dict(
            title='Calibration',
            x_axis_label='Pulse Amplitude (%)',
            y_axis_label='Signal Sum of Squares (V²)',
            tooltips=[
                ('x', '@x'),
                ('y', '@y')
            ]
        )
        _figure_opts.update(figure_opts) # allow options to be overriden
        super().__init__(figure_opts=_figure_opts, **kwargs)

class PulseCalibration(BaseExperiment):
    def setup(self):
        self.log.setLevel('INFO')
        self.default_par_files=[os.path.join(GLOBALS_DIR, 'shims.yaml')]
        self.seq = Sequence(SEQUENCE_DIR+'pulse_calibration.py')
        self.enable_partialplot=True
        self.plots = {
            'signal': SignalPlot(show_magnitude=True),
            'cal': PulseCalibrationPlot()
        }
        
        self.inputs = auto_inputs(self.seq, {
            'tx_chan': 0,
            'rx_chan': 0,
            't_pulse': 40*Unit('us'),
            'n_pulse': 50,
            'n_scans': 2,
            't_end': 0.5*Unit('s'),
            't_acqdelay': 100*Unit('us'),
            't_dw': 1.0*Unit('us'),
            'n_samples': 1000,
            'apodization': 1000*Unit('Hz')
        })

        # set bounds
        self.inputs['n_pulse'].start=0
        self.inputs['apodization'].start=0

        self.amp_90 = 0
        self.amp_180 = 0
        
        # Adding display names for the user inputs
        # self.inputs['n_scans'].name = 'Scans'
        # self.inputs['t_end'].name = 'End Time (s)'
        # self.inputs['t_acqdelay'].name = 'Acq. Delay (μs)'

    def update_par(self):
        # load appropriate frequency file
        self.seq.loadpar(os.path.join(GLOBALS_DIR, f'frequency_tx{self.seq.par.tx_chan}.yaml'))
        self.seq.setpar(
            a_pulse = (1.0/self.inputs['n_pulse'].value)*(np.arange(self.inputs['n_pulse'].value)+1)
        )

    async def update_plots(self, final, *args, **kwargs):
        self.log.debug(final)
        await self.seq.fetch_data()
        ys = np.reshape(self.seq.data, (-1, self.seq.par.n_samples))
        t = self.seq.par.t_dw*np.arange(self.seq.par.n_samples)
        apodization = (self.inputs['apodization'].value*self.inputs['apodization'].unit).m_as('Hz')
        ys *= np.exp(-np.pi*apodization*t)[np.newaxis,:]
        sumsq = np.abs(np.sum(ys*ys, axis=1))
        self.plots['signal'].update(seqdata=ys[-1], t_dw=self.seq.par.t_dw)
        self.plots['cal'].update(data=dict(sumsq=dict(x=100*self.seq.par.a_pulse[:len(sumsq)], y=sumsq)))
        if final:
            amp_90_i = 0
            y_sumsq_90 = 0
            amp_180_i = 0
            y_sumsq_180 = 0
            found_90 = False
            found_180 = False
            for amp_i, (amp, y_sumsq) in enumerate(zip(self.seq.par.a_pulse, sumsq)):
                if not found_90:
                    if y_sumsq>y_sumsq_90:
                        y_sumsq_90 = y_sumsq
                        amp_90_i = amp_i
                    elif y_sumsq<0.9*y_sumsq_90 and amp_i>amp_90_i+2:
                        found_90 = True
                        y_sumsq_180 = y_sumsq_90
                elif not found_180:
                    if y_sumsq<y_sumsq_180:
                        y_sumsq_180 = y_sumsq
                        amp_180_i = amp_i
                    elif y_sumsq>0.05*y_sumsq_90 and amp_i>amp_180_i+2:
                        found_180 = True
                if found_90 and found_180:
                    break
            if found_90:
                p_90 = np.polyfit(
                    self.seq.par.a_pulse[amp_90_i-2:amp_90_i+3],
                    sumsq[amp_90_i-2:amp_90_i+3],
                    2)
                self.amp_90 = float(-p_90[1]/(2*p_90[0]))
                self.log.info(f'90 pulse amplitude: {self.amp_90}')
                with open(os.path.join(GLOBALS_DIR, f'hardpulse_90_tx{self.seq.par.tx_chan}.yaml'), 'r') as file:
                    pulse_data = yaml.full_load(file)
                pulse_data['a_90'] = self.amp_90
                pulse_data['t_90'] = self.seq.par.t_pulse
                with open(os.path.join(GLOBALS_DIR, f'hardpulse_90_tx{self.seq.par.tx_chan}.yaml'), 'w') as file:
                    yaml.dump(pulse_data, file, default_flow_style=False)
            if found_180:
                p_180 = np.polyfit(
                    self.seq.par.a_pulse[amp_180_i-2:amp_180_i+3],
                    sumsq[amp_180_i-2:amp_180_i+3],
                    2)
                self.amp_180 = float(-p_180[1]/(2*p_180[0]))
                self.log.info(f'180 pulse amplitude: {self.amp_180}')
                with open(os.path.join(GLOBALS_DIR, f'hardpulse_180_tx{self.seq.par.tx_chan}.yaml'), 'r') as file:
                    pulse_data = yaml.full_load(file)
                pulse_data['a_180'] = self.amp_180
                pulse_data['t_180'] = self.seq.par.t_pulse
                with open(os.path.join(GLOBALS_DIR, f'hardpulse_180_tx{self.seq.par.tx_chan}.yaml'), 'w') as file:
                    yaml.dump(pulse_data, file, default_flow_style=False)


from matipo.util.optimisation import nelder_mead_async

class Autoshim(BaseExperiment):
    def setup(self):
        self.default_par_files=[]
        self.shim_file = os.path.join(GLOBALS_DIR, 'shims.yaml')
        self.seq = Sequence(SEQUENCE_DIR+'FID.py')
        self.enable_partialplot=True
        self.plots = {
            'signal': SignalPlot(show_magnitude=True),
            'shims': LinePlot(
                legend_opts=dict(
                    location='bottom_left',
                    orientation='horizontal'
                ),
                figure_opts=dict(
                    title='Shim Values',
                    x_axis_label='Iteration',
                    y_axis_label='Value (%)'
                )
            )
        }
        
        self.inputs = auto_inputs(self.seq, {
            'tx_chan': 0,
            'rx_chan': 0,
            'n_scans': 2,
            't_end': 0.5*Unit('s'),
            't_acqdelay': 100*Unit('us'),
            't_dw': 10.0*Unit('us'),
            'n_samples': 10000,
            'apodization': 10*Unit('Hz')
        })

        self.OFFSET_LABELS = ['x', 'y', 'z']
        self.N_OFFSETS = len(self.OFFSET_LABELS)
        self.OFFSET_STEP = 0.1 # initial step size
        self.OFFSET_PRECISION = 0.0005
        self.SHIM_LABELS = ['z2', 'zx', 'zy', 'xy', 'x2y2']
        self.N_SHIMS = len(self.SHIM_LABELS)
        self.SHIM_STEP = 0.1 # initial step size
        self.SHIM_PRECISION = 0.0005
        self.MAX_ITERATIONS = 500
    
    def saveshims(self, shims):
        # load existing shim data
        try:
            with open(self.shim_file, 'r') as f:
                data = yaml.full_load(f)
        except:
            data = {}
        
        # only write new values without changing existing values at higher indexes
        if 'offsets' in data:
            for i in range(self.N_OFFSETS):
                data['offsets'][i] = shims[i]
        else:
            data['offsets'] = shims[:self.N_OFFSETS]
        
        if 'shims' in data:
            for i in range(self.N_SHIMS):
                data['shims'][i] = shims[self.N_OFFSETS + i]
        else:
            data['shims'] = shims[:self.N_OFFSETS]
        
        with open(self.shim_file, 'w') as f:
            yaml.dump(data, f)
    
    async def update_plots(self, final, *args, **kwargs):
        self.plots['signal'].update(seqdata=self.y_latest, t_dw=self.seq.par.t_dw)
        shim_vals_x_axis = np.arange(len(self.shim_vals[self.labels[0]]))
        self.plots['shims'].update(data={str(key): dict(x=shim_vals_x_axis, y=self.shim_vals[key]) for key in self.labels})
    
    async def run(self, progress_handler):
        self.labels = self.OFFSET_LABELS + self.SHIM_LABELS
        
        # load default parameter files
        self.seq.loadpar(os.path.join(GLOBALS_DIR, f'frequency_tx{self.seq.par.tx_chan}.yaml'))
        self.seq.loadpar(os.path.join(GLOBALS_DIR, f'hardpulse_90_tx{self.seq.par.tx_chan}.yaml'))
        
        # calculate window once
        t = self.seq.par.t_dw*np.arange(self.seq.par.n_samples)
        apodization = (self.inputs['apodization'].value*self.inputs['apodization'].unit).m_as('Hz')
        self.window = np.exp(-np.pi*apodization*t)
        
        self.y_latest = None # keep copy of latest signal data for plotting
        self.shim_vals = {key: [] for key in self.labels}
        self.sumsqs = []
        
        init_step = [self.OFFSET_STEP]*self.N_OFFSETS + [self.SHIM_STEP]*self.N_SHIMS
        precision = [self.OFFSET_PRECISION]*self.N_OFFSETS + [self.SHIM_PRECISION]*self.N_SHIMS
        lower_bounds = [-1]*self.N_OFFSETS + [-1]*self.N_SHIMS
        upper_bounds = [1]*self.N_OFFSETS + [1]*self.N_SHIMS
        shims = np.zeros(self.N_OFFSETS+self.N_SHIMS)
        
        # do a dummy execution so that the first actual run has similar T1 relaxation effects
        await self.evalfunc(shims, dummy=True)
        
        i = 0
        best_shims = shims
        async for r in nelder_mead_async(self.evalfunc, shims, x_lb=lower_bounds, x_ub=upper_bounds, max_iter=self.MAX_ITERATIONS, step=init_step, x_precision=precision):
            if not np.array_equal(r[0], best_shims):
                best_shims = r[0].tolist()
                self.saveshims(best_shims)
            progress_handler(i, self.MAX_ITERATIONS)
            i += 1
    
    async def evalfunc(self, try_shims, dummy=False):
        # Shim evaluation function that runs an FID pulse sequence and returns the signal sum of squares
        self.seq.setpar(offsets=try_shims[:self.N_OFFSETS], shims=try_shims[self.N_OFFSETS:])
        
        self.y_latest = await self.seq.run()
        
        # return early if it's a dummy run
        if dummy:
            return 0

        sumsq = np.sum(np.abs(self.window*self.y_latest) ** 2)
        self.sumsqs.append(sumsq)
        for i, key in enumerate(self.labels):
            self.shim_vals[key].append(100*try_shims[i])
        
        return -sumsq


def main(doc=None):
    findfreq = FindFrequency()
    tunematch = TuneMatch()
    noisecheck = NoiseCheck()
    pulse_calibration = PulseCalibration()
    autoshim = Autoshim()
    return pn.Tabs(
        (findfreq.title, findfreq()),
        (noisecheck.title, noisecheck()),
        (tunematch.title, tunematch()),
        (pulse_calibration.title, pulse_calibration()),
        (autoshim.title, autoshim())
    )
