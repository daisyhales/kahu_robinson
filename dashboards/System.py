import panel as pn
import sys
import os
import yaml
import time
import logging

from bokeh.models import DatetimeTickFormatter

from matipo.system_api import SystemAPI, StatusMonitor
from matipo import Sequence, SEQUENCE_DIR
from matipo.experiment.base_experiment import CardLogHandler, logerrors

import numpy as np

from bokeh.models import HoverTool
from bokeh.plotting import figure, show, ColumnDataSource

from matipo.experiment.plot_interface import PlotInterface
from matipo.experiment.models import PLOT_COLORS, SITickFormatter, SIHoverFormatter

class StreamingLinePlot(PlotInterface):
    def __init__(self, rollover=1000, figure_opts={}, line_opts={}, color_palette=PLOT_COLORS, legend_opts={}):
        self._rollover = rollover
        self.scatters = dict()
        self.sources = dict()
        
        _figure_opts = dict(
            toolbar_location="above",
            tools='pan,wheel_zoom,box_zoom,crosshair,reset,save',
            sizing_mode="stretch_both",
            min_height=400,
            min_width=400,
            tooltips=[
                ('x', '@x'),
                ('y', '@y')
            ]
        )
        _figure_opts.update(figure_opts) # allow options to be overriden
        
        self._line_opts = dict()
        self._line_opts.update(line_opts) # allow options to be overriden
        
        self._color_palette = color_palette
        self.legend_opts = legend_opts
        
        self._count = 0
        self.lines = {}

        self.fig = figure(**_figure_opts)
        self.fig.axis.formatter = SITickFormatter()
        self.fig.toolbar.logo = None

    def update(self, data):
        """ Update the plot with a data dict, e.g.:

        data = dict(
            label_1=dict(x=[0], y=[0]),
            label_2=dict(x=[0], y=[0])
        )
        """
        self.state = data

    def stream(self, data):
        """ Append to the plot with a data dict, e.g.:

        data = dict(
            label_1=dict(x=[0], y=[0]),
            label_2=dict(x=[0], y=[0])
        )
        """
        for key in data:
            if key in self.sources:
                self.sources[key].stream(data[key], rollover=self._rollover)
    
    @property
    def state(self):
        return {key: self.sources[key].data for key in self.sources}
    
    @state.setter
    def state(self, data):
        for key in data:
            if key in self.sources:
                self.sources[key].data = data[key]
            else:
                self.sources[key] = ColumnDataSource(data = data[key])
                self.lines[key] = self.fig.line(x='x', y='y', legend_label=key, color=self._color_palette[self._count%len(self._color_palette)], source=self.sources[key], **self._line_opts)
                self._count+=1
        for key,val in self.legend_opts.items():
            setattr(self.fig.legend, key, val)
        
    def __call__(self):
        return self.fig

def enable_handler(t, e):
    t.disabled = not e.new

def disable_handler(t, e):
    t.disabled = e.new

class SystemDashboard():
    def __init__(self):
        self.log = logging.getLogger(f'SystemDashboard')
        self.log.setLevel(logging.INFO)
        self.log.propagate = False # don't propagate to notebook when logs will appear in the status box
        self.log_handler = CardLogHandler()
        self.log.addHandler(self.log_handler)
        
        self.system_api = SystemAPI()

        # Driver Reset
        self.reset_description = pn.pane.HTML('''
            <h4>System Reset</h4>
            <p>Reset the hardware driver, forcibly terminating any sequence that may be running.</p>
            ''', sizing_mode='stretch_width', max_width=400)
        self.reset_button = pn.widgets.Button(name='Reset Driver', button_type='danger')
        self.reset_button.on_click(self.reset_button_handler)

        # Poweroff
        self.poweroff_description = pn.pane.HTML('''
            <h4>System Shutdown</h4>
            <p>Initiate a shutdown, the system will power down by itself when complete.
            To avoid the possibility of files being corrupted, do not disconnect the power without shutting down,
            and wait until the front LED indicator turns off before disconnecting power.</p>
            ''', sizing_mode='stretch_width', max_width=400)
        self.poweroff_enable = pn.widgets.Switch(name='Enable', align='center')
        self.poweroff_button = pn.widgets.Button(name='Shutdown', button_type='danger', disabled=True)
        self.poweroff_enable.link(self.poweroff_button, callbacks={'value': enable_handler})
        self.poweroff_button.on_click(self.poweroff_button_handler)

        # RF module gain control
        self.set_gain_seq = Sequence(SEQUENCE_DIR+'rf_module_gain.py')
        self.gain_input = pn.widgets.DiscreteSlider(name='Rx Gain (dB)', value=21, options=list(range(0, 46, 3))) # 0 to 45 in 3 dB steps
        self.gain_set_button = pn.widgets.Button(name='Set', button_type='primary', align='end')
        self.gain_set_button.on_click(self.gain_set_button_handler)

        # temperature control
        self.N_PWM = 2
        self.N_SENSOR = 2
        temp_ctrl_config = self.system_api.tempctrl_get_config()
        self.setpoint_input = pn.widgets.FloatInput(name='Set Point (°C)', value=temp_ctrl_config['setpoint'], start=-20, end=60, step=1, sizing_mode='stretch_width')
        self.pwm_period_inputs = [pn.widgets.FloatInput(name=f'PWM {i_pwm+1} Period (ms)', value=1000*temp_ctrl_config['pwm_period'][i_pwm], start=1e-3, end=4000, step=1, sizing_mode='stretch_width') for i_pwm in range(self.N_PWM)]
        self.pwm_override_enables = [pn.widgets.Switch(name=f'PWM {i_pwm+1} Override', align='center') for i_pwm in range(self.N_PWM)]
        self.pwm_override_inputs = [pn.widgets.FloatInput(name=f'PWM {i_pwm+1} Duty (%)', value=100*temp_ctrl_config['pwm_duty'][i_pwm], start=0, end=100, step=1, sizing_mode='stretch_width', disabled=True) for i_pwm in range(self.N_PWM)]
        self.K_inputs = [] # indexing: PWM, SENSOR, PID
        for i_pwm in range(self.N_PWM):
            self.pwm_override_enables[i_pwm].link(self.pwm_override_inputs[i_pwm], callbacks={'value': enable_handler})
            self.K_inputs.append([])
            for i_sensor in range(self.N_SENSOR):
                self.K_inputs[-1].append([])
                for name in ['KP','KI','KD']:
                    self.K_inputs[-1][-1].append(
                        pn.widgets.FloatInput(name=f'PWM {i_pwm+1} Sensor {i_sensor+1} {name}', value=temp_ctrl_config[name][i_pwm][i_sensor], sizing_mode='stretch_width')
                    )
                    self.pwm_override_enables[i_pwm].link(self.K_inputs[i_pwm][-1][-1], callbacks={'value': disable_handler})
            self.pwm_override_enables[i_pwm].value = temp_ctrl_config['pwm_override'][i_pwm]
        self.temp_update_button = pn.widgets.Button(name='Update', button_type='primary', sizing_mode='stretch_width')
        self.temp_update_button.on_click(self.temp_ctrl_update_button_handler)
        self.temp_plot = StreamingLinePlot(
            figure_opts=dict(
                x_axis_type="datetime",
                title="Temperature",
                x_axis_label="time",
                y_axis_label="temperature (°C)",
                min_height=200
            ),
            legend_opts=dict(location='top_left'),
            rollover=3600 # ~1 hour
        )
        self.temp_plot.fig.axis[0].formatter = DatetimeTickFormatter()
        
        self.pwm_plot = StreamingLinePlot(
            figure_opts=dict(
                x_axis_type="datetime",
                title="PWM Duty Cycle",
                x_axis_label="time",
                x_range=self.temp_plot.fig.x_range, # link x axes
                y_range=(-5,105),
                y_axis_label="Duty (%)",
                min_height=200
            ),
            legend_opts=dict(location='top_left'),
            rollover=3600 # ~1 hour
        )
        self.pwm_plot.fig.axis[0].formatter = DatetimeTickFormatter()
        
        self.temp_pwm_plot_container = pn.Column(
            self.temp_plot(),
            self.pwm_plot(),
            sizing_mode='stretch_both'
        )

        # temperature/pwm plot automatic updating
        temp_history = {}
        for i in range(self.N_SENSOR):
            data = self.system_api.tempctrl_get_history(sensor=i)
            if len(data)!=0:
                temp_history[f'Sensor {i+1}'] = dict(x=1000*data[0], y=data[1])
        self.temp_plot.update(temp_history)
        pwm_history = {}
        for i in range(self.N_PWM):
            data = self.system_api.tempctrl_get_pwm_history(pwm=i)
            if len(data)!=0:
                pwm_history[f'PWM {i+1}'] = dict(x=1000*data[0], y=100*data[1])
        self.pwm_plot.update(pwm_history)
        def on_data(data):
            self.log.debug(data)
            if data['name'] == 'temperature':
                self._update_plots_wrapper(
                    lambda: self.temp_plot.stream({f'Sensor {data['index']+1}': dict(x=[1000*data['time']], y=[data['value']])})
                )
            if data['name'] == 'pwm_duty':
                self._update_plots_wrapper(
                    lambda: self.pwm_plot.stream({f'PWM {data['index']+1}': dict(x=[1000*data['time']], y=[100*data['value']])})
                )
            
        self.status_mon = StatusMonitor(on_data)

    @logerrors
    def _update_plots_wrapper(self, func):
        # workaround for bokeh plots not updating when using "preview with panel"
        curdoc = pn.state.curdoc
        if curdoc is not None and curdoc.session_context is not None:
            curdoc.add_next_tick_callback(func)
        else:
            try:
                func()
                pn.io.push_notebook(self.temp_pwm_plot_container)
            except RuntimeError:
                pass
    
    @logerrors
    def reset_button_handler(self, e):
        self.system_api.reset()
        self.log.info('Driver Reset')

    @logerrors
    def poweroff_button_handler(self, e):
        self.log.info('Initiating shutdown...')
        self.system_api.poweroff()
        self.log.info('Done')

    @logerrors
    async def gain_set_button_handler(self, e):
        gain = self.gain_input.value
        self.set_gain_seq.setpar(gain=gain)
        await self.set_gain_seq.run()
        self.log.info('RF Module gain set to %d dB', gain)

    @logerrors
    def temp_ctrl_update_button_handler(self, e):
        self.system_api.tempctrl_setpoint(temp=self.setpoint_input.value)
        
        for i_pwm in range(self.N_PWM):
            self.system_api.tempctrl_set_pwm_period(pwm=i_pwm, period=0.001*self.pwm_period_inputs[i_pwm].value)
            if self.pwm_override_enables[i_pwm].value:
                # set PWM directly, disabling PID
                self.system_api.tempctrl_set_pwm_override(pwm=i_pwm, duty=0.01*self.pwm_override_inputs[i_pwm].value)
            else:
                # set PID constants and enable PID
                for i_sensor in range(self.N_SENSOR):
                    self.system_api.tempctrl_set_PID(
                        pwm=i_pwm, sensor=i_sensor,
                        KP=self.K_inputs[i_pwm][i_sensor][0].value,
                        KI=self.K_inputs[i_pwm][i_sensor][1].value,
                        KD=self.K_inputs[i_pwm][i_sensor][2].value)
                self.system_api.tempctrl_enable_PID(pwm=i_pwm)
        
        self.log.info('Updated temperature control settings')

    def __call__(self):
        return pn.Column(
            pn.Tabs(
                ('Temp. Monitor', self.temp_pwm_plot_container),
                ('Temp. Control', pn.Column(
                        self.setpoint_input,
                        pn.Row(
                            *[pn.Column(
                                pn.Row(pn.panel('Override', align='center'), self.pwm_override_enables[i_pwm], self.pwm_override_inputs[i_pwm]),
                                pn.Row(*[pn.Column(*[inp for inp in self.K_inputs[i_pwm][i_sensor]]) for i_sensor in range(self.N_SENSOR)]),
                                self.pwm_period_inputs[i_pwm]
                            ) for i_pwm in range(self.N_PWM)],
                            sizing_mode='stretch_both', margin=(0,0,10,0)
                        ),
                        self.temp_update_button,
                )),
                ('Admin', pn.Column(
                    self.reset_description,
                    self.reset_button,
                    self.poweroff_description,
                    pn.Row(
                        self.poweroff_enable,
                        self.poweroff_button,
                    ),
                    sizing_mode='stretch_both'
                )),
                ('RF Module', pn.Column(
                    pn.Row(self.gain_input, self.gain_set_button),
                    sizing_mode='stretch_both'
                )),
                sizing_mode='stretch_both', margin=(0,0,15,0)
            ),
            pn.Column(self.log_handler.view), # put in its own column to avoid it taking up too much space
            sizing_mode='stretch_both'
        )

def main(doc=None):
    system_dashboard = SystemDashboard()
    return system_dashboard()
