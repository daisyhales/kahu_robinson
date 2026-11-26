from matipo import SEQUENCE_DIR, GLOBALS_DIR
from matipo.sequence import Sequence

import panel as pn
import logging
from bokeh.models import ColumnDataSource
from bokeh.plotting import figure
from matipo import SEQUENCE_DIR, GLOBALS_DIR
from matipo.sequence import Sequence
import traceback

import numpy as np
import panel as pn
import yaml
import logging
from matipo.util.fft import fft_reconstruction
from matipo.util.fft import get_freq_spectrum
from scipy.signal import chirp, find_peaks, peak_widths
from bokeh.models import HoverTool
from bokeh.plotting import figure
from bokeh.models import TabPanel, Tabs
from bokeh.layouts import row, column
from bokeh.models import Span
import pandas as pd

from scipy.integrate import cumulative_trapezoid, trapezoid
from bokeh.models import LinearColorMapper, ColorBar
from bokeh.palettes import Viridis256
from bokeh.layouts import gridplot, column, row
from bokeh.palettes import Category10
from scipy.optimize import curve_fit

from scipy.optimize import nnls, lsq_linear
from scipy.signal import find_peaks, peak_widths
from bokeh.palettes import Turbo256, linear_palette, Bokeh




class panes:
    """
    Creates the panes that can be used in the panels made.
    """
    def __init__(self):

        height = 500
        lw = 2
        scatter_size = 8

        # ------------ logging and status panes ------------

        self.log_pane = pn.pane.Str("", height=200, sizing_mode="stretch_width", styles={"overflow-y": "auto"})
        self.plotlog_pane = pn.pane.Str("", height=200, sizing_mode="stretch_width", styles={"overflow-y": "auto"})
        
        self.softlog_pane = pn.pane.Str("", height=200, sizing_mode="stretch_width", styles={"overflow-y": "auto"})
        
        self.status_pane = pn.pane.Markdown("", sizing_mode="stretch_width")
        self.plotstatus_pane = pn.pane.Markdown("", sizing_mode="stretch_width")
        self.softstatus_pane = pn.pane.Markdown("", sizing_mode="stretch_width")
        
        self.info_pane = pn.pane.Markdown("", sizing_mode="stretch_width")
        self.progress_bar = pn.indicators.Progress(name="Acquisition Progress",value=0,max=100,width=150)

        
        self.plot_pane = pn.pane.Bokeh()
        self.img_pane = pn.pane.Bokeh()
        self.tabs_b0 = pn.Tabs()

        

        self.soft_pulse_plot = pn.pane.Bokeh(height=400, sizing_mode="stretch_width")

        self.twoimagetabs = pn.Tabs()



        # -------------- compiler decay plots --------------
        data_E_sources = {}
        n_PE = 8
        for i in range(1, n_PE+1): 
            
            data_E_sources[i]= ColumnDataSource(data=dict(x=[], y=[]))

        self.dict_E_withPE = data_E_sources
        self.data_E_noPE = ColumnDataSource(data=dict(x=[], y=[]))

        self.decay_E_compiler = pn.pane.Bokeh()
        # reuse data from eddy decay
        p_de = figure(title="Field Gradient Decay", height = height, sizing_mode="stretch_width", tools="pan,wheel_zoom,box_zoom,reset,save")
        colours = Bokeh[8]



        self.renderers_E = {}
        self.legend_items_E = {}
        
        for i in range(1, n_PE + 1):
        
            r = p_de.line("x", "y", source=self.dict_E_withPE[i], color=colours[i-1], legend_label=f"PE {i}", line_width=lw)
        
            self.renderers_E[i] = r
            self.legend_items_E[i] = p_de.legend[0].items[-1]

        p_de.line("x", "y", source=self.data_E_noPE, color="black", legend_label="No PE", line_width = lw) # initialise empty then add data later 
        
        p_de.add_tools(HoverTool(tooltips=[("Time (s)", "@x{0.00}"), ("Value", "@y{0.00}")]))
        p_de.legend.click_policy = "hide"
        p_de.xaxis.axis_label = "Time (s)"
        p_de.yaxis.axis_label = "Field Gradient (kHz/m)"
        self.decay_E_compiler.object = p_de


        
        data_B0_sources = {}
        n_PE = 8
        for i in range(1, n_PE+1): 
            
            data_B0_sources[i]= ColumnDataSource(data=dict(x=[], y=[]))

        self.dict_B0_withPE = data_B0_sources
        self.data_B0_noPE = ColumnDataSource(data=dict(x=[], y=[]))

        self.decay_B0_compiler = pn.pane.Bokeh()
        # reuse data from eddy decay
        p_db = figure(title="Field Offset Decay", height = height, sizing_mode="stretch_width", tools="pan,wheel_zoom,box_zoom,reset,save")
        # self.renderers_B0 = {}
        
        # for i in range(1, n_PE+1):
        #     self.renderers_B0[i] = p_db.line("x", "y", source=self.dict_B0_withPE[i], color=colours[i-1], legend_label=f"PE {i}", line_width = lw) # initialise empty then add data later 


        self.renderers_B0 = {}
        self.legend_items_B0 = {}
        
        for i in range(1, n_PE + 1):
        
            rb = p_db.line("x", "y", source=self.dict_B0_withPE[i], color=colours[i-1], legend_label=f"PE {i}", line_width = lw)
        
            self.renderers_B0[i] = rb
            self.legend_items_B0[i] = p_db.legend[0].items[-1]
           
        # p_db.line("x", "y", source=self.dict_B0_withPE[i], color=colours[i-1], legend_label=f"PE {i}", line_width = lw) # initialise empty then add data later 
        
        p_db.line("x", "y", source=self.data_B0_noPE, color="black", legend_label="No PE", line_width = lw) # initialise empty then add data later 
        
        p_db.add_tools(HoverTool(tooltips=[("Time (s)", "@x{0.00}"), ("Value", "@y{0.00}")]))
        p_db.legend.click_policy = "hide"
        p_db.xaxis.axis_label = "Time (s)"
        p_db.yaxis.axis_label = "Field Offset (kHz)"
        self.decay_B0_compiler.object = p_db

        self.decay_col_compiler = pn.Row(self.decay_E_compiler, self.decay_B0_compiler, sizing_mode="stretch_width",)


        

        # eddy measure GUI

        


        # -------------- decay_plots --------------

        self.data_E_source = ColumnDataSource(data=dict(x=[], y=[]))

        self.decay_E = pn.pane.Bokeh()
        # reuse data from eddy decay
        p_de = figure(title="Field Gradient Decay", height = height, sizing_mode="stretch_width", tools="pan,wheel_zoom,box_zoom,reset,save")
        p_de.scatter("x", "y", source=self.data_E_source, color="blue", size = scatter_size, legend_label="Data") # initialise empty then add data later 
        p_de.add_tools(HoverTool(tooltips=[("Time (s)", "@x{0.00}"), ("Value", "@y{0.00}")]))
        p_de.legend.click_policy = "hide"
        p_de.xaxis.axis_label = "Time (s)"
        p_de.yaxis.axis_label = "Field Gradient (kHz/m)"
        self.decay_E.object = p_de

        self.data_B0_source = ColumnDataSource(data=dict(x=[], y=[]))

        self.decay_B0 = pn.pane.Bokeh()
        # reuse data from eddy decay
        p_db = figure(title="Field Offset Decay", height = height, sizing_mode="stretch_width", tools="pan,wheel_zoom,box_zoom,reset,save")
        p_db.scatter("x", "y", source=self.data_B0_source, color="orange", size = scatter_size, legend_label="Data") # initialise empty then add data later 
        p_db.add_tools(HoverTool(tooltips=[("Time (s)", "@x{0.00}"), ("Value", "@y{0.00}")]))
        p_db.legend.click_policy = "hide"
        p_db.xaxis.axis_label = "Time (s)"
        p_db.yaxis.axis_label = "Field Offset (kHz)"
        self.decay_B0.object = p_db

        self.decay_col = self.decay_col = pn.Row(self.decay_E, self.decay_B0, sizing_mode="stretch_width",)

        

        self.analysis_plot = pn.Column(sizing_mode="stretch_width")
        self.spectra_plot = pn.Column(sizing_mode="stretch_width")

        
        
        self.tabs_measure = pn.Tabs(
            ("EC Measurement", self.decay_col),
            ("Analysis Plots", self.analysis_plot),
            ("Spectra Plots", self.spectra_plot),
            sizing_mode="stretch_width",
        )


        self.tabs_measure_compiler = pn.Tabs(
            ("EC Measurement", self.decay_col_compiler),
            ("Analysis Plots", self.analysis_plot),
            ("Spectra Plots", self.spectra_plot),
            sizing_mode="stretch_width",
        )

        
        self.ideal_source = ColumnDataSource(data=dict(x=[0, 1], y=[0, 0]))

        # -------------- tau plot E ------------------
        self.tau_E_source = ColumnDataSource(data=dict(x=[], y=[]))
        
        self.tau_E = pn.pane.Bokeh()
        p_tauE = figure(title="Least Squares Regression - Curve Fit Initial Guess", x_axis_type="log", height=height, sizing_mode="stretch_width", tools="pan,wheel_zoom,box_zoom,reset,save")

        p_tauE.line("x", "y", source=self.tau_E_source, color="black", line_width=lw, legend_label="Fit") # initialise empty then add data later 
        p_tauE.add_tools(HoverTool(tooltips=[("Tau (s)", "@x{0.00}"), ("Amplitude", "@y{0.00}")]))
        p_tauE.legend.click_policy = "hide"
        p_tauE.xaxis.axis_label = "Tau (s)"
        p_tauE.yaxis.axis_label = "Amplitude"
        p_tauE.legend.location = "top_right"

        self.tau_E.object = p_tauE

        
        # -------------- fitting E plot --------------
                
        self.fit_E_source = ColumnDataSource(data=dict(x=[], y=[]))

        self.fit_E = pn.pane.Bokeh()
        # reuse data from eddy decay
        p_g = figure(title="Field Gradient Decay", height = height, sizing_mode="stretch_width", tools="pan,wheel_zoom,box_zoom,reset,save")
        p_g.scatter("x", "y", source=self.data_E_source, color="blue", size = scatter_size, legend_label="Data") # initialise empty then add data later 
        p_g.line("x", "y", source=self.fit_E_source, color="red", line_width=lw, legend_label="Fit") # initialise empty then add data later 
        p_g.add_tools(HoverTool(tooltips=[("Time (s)", "@x{0.00}"), ("Value", "@y{0.00}")]))
        p_g.legend.click_policy = "hide"
        p_g.xaxis.axis_label = "Time (s)"
        p_g.yaxis.axis_label = "Field Gradient (kHz/m)"
        
        self.fit_E.object = p_g

        
        # -------------- residuals E plot --------------
        self.res_E_source = ColumnDataSource(data=dict(x=[], y=[]))

        self.res_E = pn.pane.Bokeh()
        # reuse data from eddy decay
        p_rE = figure(title="Residuals", height=300, width = 300, tools="pan,wheel_zoom,box_zoom,reset,save")
        p_rE.scatter("x", "y", source=self.res_E_source, color="green", size = scatter_size, legend_label="Data") # initialise empty then add data later 
        p_rE.line("x","y", source=self.ideal_source, color="black", line_dash="dashed", legend_label="Ideal")
        p_rE.legend.click_policy = "hide"
        p_rE.add_tools(HoverTool(tooltips=[("x", "$x{0.00}"), ("y", "$y{0.00}")]))
        p_rE.yaxis.axis_label = "Difference (Hz/m)"
        p_rE.xaxis.axis_label = "Time (s)"
        self.res_E.object = p_rE    
        
        # -------------- tau plot B0 ------------------
        self.tau_B0_source = ColumnDataSource(data=dict(x=[], y=[]))
        
        self.tau_B0 = pn.pane.Bokeh()
        p_tauB0 = figure(title="Least Squares Regression - Curve Fit Initial Guess", x_axis_type="log", height=height, sizing_mode="stretch_width", tools="pan,wheel_zoom,box_zoom,reset,save")

        p_tauB0.line("x", "y", source=self.tau_B0_source, color="black", line_width=lw, legend_label="Fit") # initialise empty then add data later 
        p_tauB0.add_tools(HoverTool(tooltips=[("Tau (s)", "@x{0.00}"), ("Amplitude", "@y{0.00}")]))
        p_tauB0.legend.click_policy = "hide"
        p_tauB0.xaxis.axis_label = "Tau (s)"
        p_tauB0.yaxis.axis_label = "Amplitude"
        p_tauB0.legend.location = "top_right"

        self.tau_B0.object = p_tauB0


        # -------------- fitting B0 plot --------------

        self.fit_B0_source = ColumnDataSource(data=dict(x=[], y=[]))

        self.fit_B0 = pn.pane.Bokeh()
        # reuse data from eddy decay
        p_b0 = figure(title="Field Offest Decay", height = height, sizing_mode="stretch_width", tools="pan,wheel_zoom,box_zoom,reset,save")
        p_b0.scatter("x", "y", source=self.data_B0_source, color="orange", size = scatter_size, legend_label="Data") # initialise empty then add data later 
        p_b0.line("x", "y", source=self.fit_B0_source, color="red", line_width=lw, legend_label="Fit") # initialise empty then add data later 
        p_b0.add_tools(HoverTool(tooltips=[("Time (s)", "@x{0.00}"), ("Value", "@y{0.00}")]))
        p_b0.legend.click_policy = "hide"
        p_b0.xaxis.axis_label = "Time (s)"
        p_b0.yaxis.axis_label = "Field Offset (kHz)"
        self.fit_B0.object = p_b0
        
        # -------------- residuals B0 plot --------------
        self.res_B0_source = ColumnDataSource(data=dict(x=[], y=[]))
        

        
        self.res_B0 = pn.pane.Bokeh()
        # reuse data from eddy decay
        p_rb0 = figure(title="Residuals", height=300, width = 300, tools="pan,wheel_zoom,box_zoom,reset,save")
        p_rb0.scatter("x", "y", source=self.res_B0_source, color="green", size = scatter_size, legend_label="Data") # initialise empty then add data later 
        p_rb0.line("x","y", source=self.ideal_source, color="black", line_dash="dashed", legend_label="Ideal")
        p_rb0.legend.click_policy = "hide"
        p_rb0.add_tools(HoverTool(tooltips=[("x", "$x{0.00}"), ("y", "$y{0.00}")]))
        p_rb0.yaxis.axis_label = "Difference (Hz)"
        p_rb0.xaxis.axis_label = "Time (s)"
        self.res_B0.object = p_rb0
     

        # soft pulse calibration panes
        self.soft_kplot = pn.pane.Bokeh()
        self.soft_imgplot = pn.pane.Bokeh()
        self.soft_diffplot = pn.pane.Bokeh()

        self.mesh_plot = pn.pane.Bokeh()
        self.seq_plot = pn.pane.Bokeh(sizing_mode="stretch_width")

    def info_pane_input(self, string):
        """
        string to put in the info pane
        """
        self.info_pane = pn.pane.Markdown(string, sizing_mode="stretch_width")

