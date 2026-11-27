import asyncio
import logging
import os
import traceback
from datetime import date, datetime
from pathlib import Path

import matplotlib.pyplot as plt
import napari
import numpy as np
import panel as pn
import yaml

from bokeh.layouts import column, row
from bokeh.models import HoverTool, Span, TabPanel, Tabs
from bokeh.plotting import figure

from panel.io.state import state
from bokeh.models import LinearColorMapper, ColorBar
from bokeh.palettes import Greys256
from bokeh.plotting import figure
from scipy.signal import chirp, find_peaks, peak_widths

from matipo import GLOBALS_DIR, SEQUENCE_DIR, DATA_DIR
from matipo.sequence import Sequence
from matipo.util.fft import fft_reconstruction, get_freq_spectrum

from user_interface.general import general_functions
from user_interface.oneDfunctions import oneDfunctions
from user_interface.panes import panes
from user_interface.updating import updating
from user_interface.twoDfunctions import twoDfunctions
from user_interface.threeDfunctions import threeDfunctions
from qtpy.QtWidgets import QApplication
import threading
import sys




class ViewerButtons:

    def __init__(self, general_functions, panes):

        self.g = general_functions
        self.panes = panes

        self.kdata = None
        self.scan_info = None
        self.dimension = None

        self.u = updating
        self.threeDfunctions = threeDfunctions(self.g, self.panes)
        self.twoDfunctions = twoDfunctions(self.g, self.panes)

        
    def wire(self, widget_dict, file, dimension):
        self.widget_dict = widget_dict
        self.file = file # where the UIfunctions is kept
        self.dimension = dimension
        
    @staticmethod
    def freq_FOV(g_cal, g_val, t_dw):
        """
        g_cal: Hz/m
        g_val: T/m
        t_dw: secs
        """
        return 1/(g_cal * g_val * t_dw) # m
    
    @staticmethod
    
    def phase_FOV(g_cal, g_val, n_phase, t_phase, t_ramp):
        A = g_val * (t_phase + t_ramp)
        delta_k = g_cal * (2 * A / n_phase)
        return 1 / delta_k

    @staticmethod
    def widget_get_files(folder_name):
        data_path = DATA_DIR
        folder_path = Path(data_path) / folder_name
        files = []

        if folder_path.exists():
            files.extend(
                [f.name for f in folder_path.iterdir() if f.is_file()]
            )
        else: 
            self.g.log_msg("Path doesn't exist.")
            raise ValueError("doesnt exist for some reason.")
        
        return sorted(files)

        
    def load_files_on_click(self, event):

        if self.dimension == 1:
            self.widget_dict["select_file1"].options = self.widget_get_files(self.widget_dict["select_folder1"].value)
        
        if self.dimension == 2:
            self.widget_dict["select_file2"].options = self.widget_get_files(self.widget_dict["select_folder2"].value)
            
        if self.dimension == 3:
            self.widget_dict["select_file3"].options = self.widget_get_files(self.widget_dict["select_folder3"].value)
        
        
    def scale_voxels(self, scan_parameters, file = None):
        """
        calculates the FOV from the scan parameters for 3D scans to scale voxels in the image to mm
        """
        
        t_preread = self.get_t_preread(scan_parameters)
        scale = {"g_read": None, "g_phase1": None, "g_phase2": None}
        axis_g = {"g_read": None, "g_phase1": None, "g_phase2": None}
        for k, v in scan_parameters['img_settings'].items():
            if k == "g_read":
                g_strength = self.g.get_g_strength(v["g"], v['axis'], file = file)

                freq_F = self.freq_FOV(g_cal = g_strength, g_val = v["g"], t_dw = scan_parameters["t_dw"])
                scale[k] = freq_F / v["n"]
                axis_g[k] = v["axis"]
            
            elif k in ["g_phase1", "g_phase2"]:
            
                g_strength = self.g.get_g_strength(v["g"], v['axis'], file = file)
                phase_F = self.phase_FOV(g_cal = g_strength, g_val = v["g"], n_phase = v["n"], t_phase = scan_parameters['t_phase'], t_ramp = scan_parameters['t_ramp'])
                scale[k] = phase_F / v["n"]
                axis_g[k] = v["axis"]
        self.scale_voxel = (scale["g_phase2"]*1000, scale["g_phase1"]*1000, scale["g_read"]*1000) # in mm
        self.axes_g = (axis_g["g_phase2"], axis_g["g_phase1"], axis_g["g_read"])
        self.g.log_msg(f"Voxel size ({self.axes_g}): {self.scale_voxel} mm")


    def scale_voxels2D(self, scan_parameters, file = None):
        """
        calculates the FOV from the scan parameters for 2D scans to scale the pixels to mm
        """
        
        self.g.log_msg(f"{scan_parameters}")
        scale = {"g_read": None, "g_phase": None}
        axis_g = {"g_read": None, "g_phase": None}
        for k, v in scan_parameters["img_settings"].items():
            if k == "g_read":
                g_strength = self.g.get_g_strength(v["g"], v['axis'], file = file)

                freq_F = self.freq_FOV(g_cal = g_strength, g_val = v["g"], t_dw = scan_parameters["t_dw"])
                scale[k] = freq_F / v["n"]
                axis_g[k] = v["axis"]
            
            elif k in ["g_phase"]:
            
                g_strength = self.g.get_g_strength(v["g"], v['axis'], file = file)
                phase_F = self.phase_FOV(g_cal = g_strength, g_val = v["g"], n_phase = v["n"], t_phase = scan_parameters['t_phase'], t_ramp = scan_parameters['t_ramp'])
                scale[k] = phase_F / v["n"]
                axis_g[k] = v["axis"]
                
        self.scale_voxel = (scale["g_phase"]*1000, scale["g_read"]*1000) # in mm
        self.axes_g = (axis_g["g_phase"], axis_g["g_read"])
        self.g.log_msg(f"Voxel size ({self.axes_g}): {self.scale_voxel} mm")

    def get_file(self):
        """
        finds the file requested and loads the scan information and kdata into global variables
        """
        start_path = DATA_DIR
        if self.dimension == 1:
            fileheader = self.widget_dict["select_folder1"].value
            file_title = self.widget_dict["select_file1"].value
        elif self.dimension == 2:
            fileheader = self.widget_dict["select_folder2"].value
            file_title = self.widget_dict["select_file2"].value           
        elif self.dimension == 3:
            fileheader = self.widget_dict["select_folder3"].value
            file_title = self.widget_dict["select_file3"].value
            
        filename = Path(start_path) / fileheader / file_title
        self.file_title = file_title

        
        if filename.suffix != ".npz":
            filename = filename.with_suffix(".npz")
            
        
        try:
            # self.g.log_msg(f"filename: {filename}")
            data = np.load(filename, allow_pickle=True)
            self.g.log_msg(f"File found: {filename}")
        except Exception:
            self.g.log_msg(f"!!! Error !!! No data_file was given or filepath does not exist. Filepath tried: {filename}")
            return

        kdata = data["kdata"] # will be complex, 1D array
        self.kdata = kdata
        scan_info = data["scan_info"].item()
        self.g.log_msg(f"Loaded Scan Information: {scan_info}")

        return kdata, scan_info

    @staticmethod
    def img_plot1D(freq, img, n_samples):
        """
        For a 1D spin echo pulse sequence, this uses the bokeh.plotting panel function to display the 
        real image data in the frequency plot from the pulse sequence acquired. 
        
        The figure is definied with the title,
        sizing mode that sets to use the space in the window open and the interactive tools. The freq (kHz) is then 
        plotted against the fourier transformed signal in the real, imaginary and magnitude.
    
        Inputs:
            freq (np.array): array of frequency values
            img (np.array): array of fourier transformed signal values
            n_samples (int): number of samples acquired for the title only
    
        Returns:
            p (bokeh object): interactable figure 
        """

        hover = HoverTool(tooltips=[("x", "$x{0.000}"), ("y", "$y{0.000}")])
        p = figure(
            title=f"Real Space - {n_samples} Samples",
            sizing_mode="stretch_both",
            tools="pan,wheel_zoom,box_zoom,reset,save"
        )
 
        p.add_tools(hover)

        p.line(freq*1e-3,img.real*1e6, color='blue',  legend_label="real")
        p.line(freq*1e-3,img.imag*1e6,  color='red',   legend_label="imag")
        p.line(freq*1e-3,np.abs(img)*1e6,  color='green', legend_label="mag")
        
        p.xaxis.axis_label = "Frequency (kHz)"
        p.title.text_font_size = "18pt"
        
        p.xaxis.axis_label_text_font_size = "16pt"
        p.yaxis.axis_label_text_font_size = "16pt"
        
        p.xaxis.major_label_text_font_size = "14pt"
        p.yaxis.major_label_text_font_size = "14pt"
        
        p.legend.label_text_font_size = "14pt"
        
        return p 
        
    @staticmethod
    def signal_plot1D(time, data, n_samples, title = None):
        """
        For a 1D spin echo pulse sequence, this uses the bokeh.plotting panel function to display the 
        data in the signal plot from the pulse sequence acquired. 
        
        The figure is definied with the title,
        sizing mode that sets to use the space in the window open and the interactive tools. The time (ms) is then 
        plotted against the signal (mV) in the real, imaginary and magnitude.
    
        Inputs:
            time (np.array): array of acquisition times separated by t_dw in seconds
            data (np.array): voltage data that is retrieved from the kahu in volts
            n_samples (int): number of samples acquired for the title only
    
        Returns:
            p (bokeh object): interactable figure 
        """
        hover = HoverTool(tooltips=[("x", "$x{0.000}"), ("y", "$y{0.000}")])
        if title == None:
            p = figure(
                title=f"K Space - {n_samples} Samples",
                sizing_mode="stretch_both",
                tools="pan,wheel_zoom,box_zoom,reset,save"
            )
        else:
            p = figure(
                title=f"{title} - {n_samples} Samples",
                sizing_mode="stretch_both",
                tools="pan,wheel_zoom,box_zoom,reset,save"
            )

        p.add_tools(hover)

        p.line(time*1e3, data.real*1e3, color="blue",  legend_label="real")
        p.line(time*1e3, data.imag*1e3, color="red",   legend_label="imag")
        p.line(time*1e3, abs(data)*1e3,  color="green", legend_label="mag")
    
        p.xaxis.axis_label = "Time (ms)"
        p.yaxis.axis_label = "Signal (mV)"
        p.legend.click_policy = "hide"

        p.title.text_font_size = "18pt"
        
        p.xaxis.axis_label_text_font_size = "16pt"
        p.yaxis.axis_label_text_font_size = "16pt"
        
        p.xaxis.major_label_text_font_size = "14pt"
        p.yaxis.major_label_text_font_size = "14pt"
        
        p.legend.label_text_font_size = "14pt"
    
        return p
            
    def img_run_on_click(self, event):
        """
        opens a scaled napari image depending on the scan_type and the number of dimensions to get the kdata and image in napari. 
        The viewer will open with all images hidden so click the eye icon on the image label to view it 
        """
        try:
            self.g.log_msg("Clicked...")
            kdata, scan_info = self.get_file()
        except:
            ValueError("Doesn't exist")
            return
        self.scan_info = scan_info
        if scan_info['dimensions'] != 1:
            g_blur = self.widget_dict["gaussian_blur"].value

        t_dw = scan_info['t_dw']
        time = np.linspace(t_dw, len(kdata)*t_dw, len(kdata))
        
        if scan_info['dimensions'] == 1:
            # self.g.log_msg("Plotting...")
            self.panes.plot_pane.object = self.signal_plot1D(time, kdata, scan_info['n_samples'])
            
            freq, img = get_freq_spectrum(signal = kdata, dwell_time= t_dw)
            self.panes.img_pane.object = self.img_plot1D(freq, img, scan_info['n_samples'])
            
            
        elif scan_info['dimensions'] == 2:
            self.g.log_msg("in dimension 2")
            self.panes.twoimagetabs.clear()
            viewer = napari.Viewer(ndisplay=2)
            array_kdata = kdata
            for i, k in enumerate(array_kdata):

            # self.twoimagetabs
                
                size = 700
                
                k = k[: scan_info["n_phase"] * scan_info["n_samples"]]
                k = np.reshape(k, (scan_info["n_phase"], scan_info["n_samples"]))
                
                k_img = np.abs(k)
                k_mapper = LinearColorMapper(palette=Greys256, low=k_img.min(), high=k_img.max())
                p_k = figure(title="Raw K-space", width=size, height=size, match_aspect=True,)
                p_k.image(image=[k_img], x=0, y=0, dw=k_img.shape[1], dh=k_img.shape[0], color_mapper=k_mapper,)
                
                
                # --------------------------
                # create image from k-space
                # --------------------------
                image = fft_reconstruction(k, gaussian_blur=0, upscale_factor=1)
                
                img = np.abs(image)
                img_mapper = LinearColorMapper(palette=Greys256, low=img.min(), high=img.max())
                
                p_img = figure(title="Reconstructed Image", width=size, height=size, match_aspect=True,)
                
                p_img.image(image=[img], x=0, y=0, dw=img.shape[1], dh=img.shape[0], color_mapper=img_mapper,)
                

                plots_two = pn.Column(pn.Row(p_k, p_img))
                self.panes.twoimagetabs.append((f"Slice {scan_info['slices_list'][i]}mm", plots_two))

                # OPEN NAPARI IMAGE
                self.twoDfunctions.signal_2D(scan_info, None, k, time, cams = False, viewer = "y")
                self.twoDfunctions.open_img_napari(viewer, self.widget_dict, scan_parameters = scan_info, scan_type = scan_info["scan_type"], var = i, file = self.file)
                
            napari.run()
            
        elif scan_info['dimensions'] == 3:
            
            self.g.log_msg("in dimension 3")
            self.threeDfunctions.signal_3D(scan_info, None, kdata, time, cams = False, viewer = "y")
            self.threeDfunctions.open_img_napari(self.widget_dict, scan_parameters = scan_info, file = self.file)
            napari.run()