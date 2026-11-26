import napari 
from bokeh.models import HoverTool
import matplotlib.pyplot as plt

from matipo import SEQUENCE_DIR, GLOBALS_DIR
from matipo.sequence import Sequence
import numpy as np
import panel as pn
import yaml
import logging
from matipo.util.fft import fft_reconstruction
from matipo.util.fft import get_freq_spectrum
from scipy.signal import chirp, find_peaks, peak_widths
from bokeh.models import HoverTool
from bokeh.plotting import figure
import traceback

from .panes import panes
from .general import general_functions


class twoDfunctions:
    
    def __init__(self, general_functions, panes):
        self.g = general_functions
        self.all_pns = panes
        self.viewer2 = None

    @staticmethod
    def get_g_val(grad, axis):
        """
        gets index of axis and then grabs the gradient value there.
        """
        
        mapper = {"X": 0, "Y": 1, "Z": 2}
        return grad[mapper[axis]]

    
    def signal_2D(self, scan_parameters, widget_dict, data, time, cams = False, slices = 1, viewer = "n"):
        """
        take in scan_parameters
        get the x : g?, y : g?, z : g? -> relate to n_? 
        reshape the k data with n?, fft for image 
        take in slicing asked for and apply to img and kdata
        convert the freq to mm for each axis based on g?
        open images and kdata in napari

        """
        if cams == False:
            if slices == 1:
                if viewer == "n":
                    self.image2D_info = {"g_read": {}, "g_phase": {}}
                    
                    s_list = ["select_g_read", "select_g_phase1"]
                    g_list = ["g_read", "g_phase"]
                    n_list = ["n_samples", "n_phase"]
                    
                    for i in range(len(s_list)):
                        g = g_list[i]
                        if g == 'g_read':
                            add_dict = {"n": scan_parameters[n_list[i]], "axis": widget_dict[s_list[i]].value, "g": self.get_g_val(scan_parameters[g], widget_dict[s_list[i]].value)}
                        elif g == 'g_phase':
                            add_dict = {"n": scan_parameters[n_list[i]], "axis": widget_dict[s_list[i]].value, "g": self.get_g_val(scan_parameters[g], widget_dict[s_list[i]].value)}
    
                        self.image2D_info[g].update(add_dict)
            
                    self.g.log_msg(f"Image construction info: {self.image2D_info}")
                else:
                    self.image2D_info = scan_parameters['img_settings']
                
                try: 
                    self.g.log_msg(f"data shape: {np.shape(data)}")
                    self.kdata = np.reshape(data, (self.image2D_info["g_phase"]["n"], self.image2D_info["g_read"]["n"]))
         
                    # self.kdata = np.fft.ifftshift(self.kdata)
                    self.g.log_msg(f"kdata shape: {np.shape(self.kdata)}")
                    self.g.twoD_info = {"img_settings": self.image2D_info}

                except ValueError: 
                    if "n_ETL" in scan_parameters:
                        if scan_parameters["n_ETL"]%2 != 0:
                            self.g.log_msg(f"data shape: {np.shape(data)}")
                            self.kdata = np.reshape(data[:int(self.image2D_info["g_phase"]["n"]*self.image2D_info["g_read"]["n"])], (self.image2D_info["g_phase"]["n"], self.image2D_info["g_read"]["n"]))
                 
                            # self.kdata = np.fft.ifftshift(self.kdata)
                            self.g.log_msg(f"kdata shape: {np.shape(self.kdata)}")
                            self.g.twoD_info = {"img_settings": self.image2D_info}
        
                except Exception: 
                    tb = traceback.format_exc()
                    self.g.log_msg("Error in Fourier Transform Plot:")
                    self.g.log_msg(tb)
                
                return
 
    @staticmethod
    def freq_FOV(g_cal, g_val, t_dw):
        """
        g_cal: Hz/m
        g_val: T/m
        t_dw: secs

        calculates the read/frequency FOV from gradients and scan parameters
        """
        return 1/(g_cal * g_val * t_dw) # m
    
    @staticmethod
    def phase_FOV(g_cal, g_val, n_phase, t_phase, t_ramp):
        """
        Calculates the phase FOV from gradients and scan parameters
        """
        A = g_val * (t_phase + t_ramp)
        delta_k = g_cal * (2 * A / n_phase)
        return 1 / delta_k

    def scale_voxels(self, scan_parameters, file = None):
        """
        creates the axis tuple and scale voxel tuple (for the 2D scans will scale the pixels) that can be put into napari 
        scaling the pixels will convert their size into mm from the gradient strengths to the FOV
        """
        
        # t_preread = self.get_t_preread(scan_parameters)
        scale = {"g_read": None, "g_phase": None}
        axis_g = {"g_read": None, "g_phase": None}
        for k, v in self.image2D_info.items():
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

            
    def open_img_napari(self, viewer2, slice_dict, scan_parameters, scan_type = None, var = None, file = None):
        """
        Opens the imported 2D images and k-space data in napari 
        """

        g_blur = slice_dict["gaussian_blur"].value

        self.scale_voxels(scan_parameters, file)

        viewer2.dims.axis_labels = self.axes_g
        try:
            
            self.g.log_msg(f"kdata shape before open napari: {np.shape(self.kdata)}")

            if scan_type in ["2DSE_multi", "rare",'rare_pe', "2DSE_multiSLR"]:
                k_layer = viewer2.add_image(np.abs(self.kdata), name=f"Kdata {var+1} Slice: {scan_parameters["slices_list"][var]}mm", scale = (1,1))
            else:
                k_layer = viewer2.add_image(np.abs(self.kdata), name=f"k_space", scale = (1,1))
            k_layer.visible = False    
            # k_layer.scale = (1, 1)
            self.g.log_msg(f"Viewer Opened with Gaussian Values Selected: {g_blur}")


            for i in g_blur:
                image = fft_reconstruction(self.kdata, gaussian_blur=i, upscale_factor=1)
                self.g.log_msg(f"image shape: {np.shape(image)}")
                if scan_type in ["2DSE_multi", "rare",'rare_pe',"2DSE_multiSLR"]:
                    img_layer = viewer2.add_image(np.abs(image), name=f"Image {var + 1} Slice: {scan_parameters["slices_list"][var]}mm")
                else:    
                    img_layer = viewer2.add_image(np.abs(image), name=f"Img_guassian{i}")#, units = "mm", scale = self.scale_voxel)
                img_layer.scale = self.scale_voxel
                img_layer.units = ("mm", "mm")
                img_layer.visible = False    
                


            # Make image layer active so scale bar shows mm
            viewer2.layers.selection.clear()
            viewer2.layers.selection.add(img_layer)

            
            viewer2.scale_bar.visible = True
            viewer2.scale_bar.unit = "mm"
            k_layer.gamma = 1
            viewer2.axes.visible = True
            k_layer.interpolation2d = 'nearest'
            k_layer.interpolation3d = 'nearest'
            k_layer.rendering = 'mip'   # or 'mip'
            k_layer.projection_mode = 'none'

            img_layer.gamma = 1
            img_layer.interpolation2d = 'nearest'
            img_layer.interpolation3d = 'nearest'
            img_layer.rendering = 'mip'   # or 'mip'
            img_layer.projection_mode = 'none'      


        except Exception:
            tb = traceback.format_exc()
            self.g.log_msg("Error in opening viewer:")
            self.g.log_msg(tb)
            return
        