from matipo import SEQUENCE_DIR, GLOBALS_DIR
from matipo.sequence import Sequence

from matipo.util.fft import fft_reconstruction
from matipo.util.fft import get_freq_spectrum
from bokeh.models import PrintfTickFormatter
import numpy as np
from bokeh.plotting import figure
import panel as pn
import asyncio
import traceback
import os
from scipy.signal import chirp, find_peaks, peak_widths

from pathlib import Path
from datetime import date, datetime
import yaml

import logging

from .updating import updating
from .general import general_functions
from .panes import panes
from .panel import panel_functions
from .buttons.btn_control import ButtonController
from .log import PanelLogHandler


class run_pulse_sequence:
    """
    Sets up to run the pulse sequence.
    """
    def __init__(self, updating, general_functions, panes, panel_functions, ButtonController):
        """
        Loads the classes on initialisation of the class.
        """

        self.u = updating
        self.g = general_functions
        self.panes = panes
        self.p = panel_functions
        self.btn_ctrl = ButtonController
        self.p_last = 0
        self.panes.progress_bar.value = 0
        self.l_listed = []
        self.end = 0
        self.same_l = 0
        self.done = False


    @staticmethod
    def filter_select_grad(value, gradient_selected):
        """
        puts the gradient value into the correct axis for the gradient vector assuming the setup of (X,Y,Z) in how the Kahu is connected to the amplifiers.
        """
        mapper = {'X': np.array([value, 0,0]), 'Y': np.array([0,value, 0]), 'Z': np.array([0,0,value])}
        return mapper[gradient_selected]

    @staticmethod
    def get_g_phase(t_phase, t_ramp, n_phase, FOV, g_cal):
        """
        Calculates the g_phase from the FOV GUI inputs
        """
        A = 1 / (t_phase + t_ramp)
        B = n_phase / (2 * FOV * g_cal)
        return A*B

    @staticmethod
    def get_g_val(grad, axis):
        """
        gives index for the gradients based on the axis
        """
        mapper = {"X": 0, "Y": 1, "Z": 2}
        return grad[mapper[axis]]


    def change_pars(self, e_dict, b0_dict, x_e_factor, x_b0_factor, y_e_factor, y_b0_factor, z_e_factor, z_b0_factor):
        """
        factors are in % when they enter the function
        """
        scale_dict = {"X": [x_e_factor, x_b0_factor], "Y": [y_e_factor, y_b0_factor], "Z": [z_e_factor, z_b0_factor]}
        
        for ax, dic in e_dict.items():
            for k, v in dic.items():
                if k[0] == "A":
                    if k == "A0":
                        continue
                    else:
                        s = 1 + (scale_dict[ax][0]/100)
                        if v == 0:
                            dic[k] = 0
                        else:
                            dic[k] = v * s
                            if v <0: 
                                self.g.status_msg(f"{ax}, {k} is negative")
                                self.g.log_msg(f"{ax}, {k} is negative")
                                raise ValueError(f"{ax}, {k} is negative")
                            else:
                                continue
                            
            e_dict[ax] = dic
    
    
        for ax, dic in b0_dict.items():
            for k, v in dic.items():
                if k[0] == "A":
                    s = 1 + (scale_dict[ax][1]/100)
                    if v == 0:
                        dic[k] = 0
                    else:
                        dic[k] = v * s
            b0_dict[ax] = dic
    
            
        return e_dict, b0_dict
                
        
    def create_scan_parameters(self, widget_dict, non_input_params = None, scan_type=None, file = None):
        """
        Creates the scan parameters dict which can be loaded into the pulse sequence - the factor set
        in the input_dict in panel_functions is used to put the input values into the pulse seq format.

        filters for the 1DSE to combine each spearate gradient input into one g_read parameter to upload
        ADD the filters as you go to tranform the input values into the pulse seq parameter format.

        Inputs:
            widget_dict (dict): dictionary of all the widgets that will be made for the panel inputs 
            scan_type (str): name of the scan type
            non_input_params (dict): parameters that are set not variable frm the inputs
        Output:
            scan_parameters (dict): dictionary of parameters for the pulse sequence
        """
        if scan_type == "FID":
            shims_dict = {"X":None, "Y":None, "Z":None}
            for k, v in widget_dict.items():
                if k == "x_offset":
                    x = widget_dict[k].value
                    shims_dict["X"] = (x/100) * self.p.shims_limit[0]
                if k == "y_offset":
                    y = widget_dict[k].value
                    shims_dict["Y"] = (y/100) * self.p.shims_limit[1]
                if k == "z_offset":
                    z = widget_dict[k].value
                    shims_dict["Z"] = (z/100) * self.p.shims_limit[2]
                   
            self.g.gradient_boundaries(shims_dict["X"], shims_dict["Y"], shims_dict["Z"], shim = "y", file = file)
            scan_parameters = {"g_shims": np.array([shims_dict["X"], shims_dict["Y"], shims_dict["Z"]])} 
            self.g.log_msg(f"---- Amps available for shimming: X = {self.p.amps_limit[0]}A, Y = {self.p.amps_limit[1]}A, Z = {self.p.amps_limit[2]}A ----")
            self.g.log_msg(f"Shim Percentages: X = {round(x, 0)}%, Y = {round(y, 0)}%, Z = {round(z, 0)}%")

        else:
            scan_parameters = {"offsets": self.g.get_yaml_info(info_name = 'shims', file = file)["offsets"]}

        if scan_type in ['measure_ec']:
            
            scan_parameters["a_pulse"], p_time = self.g.get_yaml_info(info_name = "hardpulse_90_tx0", file = file).values()
            
            scan_parameters["t_pulse"] = float(p_time) * (widget_dict["flip_angle"].value/90)
            scan_parameters["t_dw"] = 1 / (widget_dict["spectral_width"].value * 1e3)
            scan_parameters["n_samples"] = widget_dict["resolution"].value
            # scan_parameters["t_acq_delay"] = widget_dict["t_acq_delay"].value * 1e-6

            scan_parameters["g_read"] = self.filter_select_grad(widget_dict["g_val"].value, widget_dict["select_g"].value)

            scan_parameters["rx_chan_list"] = (widget_dict["rx_l"].value, widget_dict["rx_c"].value, widget_dict["rx_r"].value)
            scan_parameters["pre_emphasis"] = False
            scan_parameters["b0"] = False

        if scan_type in ['compile_ec']:
            
            scan_parameters["a_pulse"], p_time = self.g.get_yaml_info(info_name = "hardpulse_90_tx0", file = file).values()
            
            scan_parameters["t_pulse"] = float(p_time) * (widget_dict["flip_angle"].value/90)
            scan_parameters["t_dw"] = 1 / (widget_dict["spectral_width"].value * 1e3)
            scan_parameters["n_samples"] = widget_dict["resolution"].value
            # scan_parameters["t_acq_delay"] = widget_dict["t_acq_delay"].value * 1e-6

            scan_parameters["g_read"] = self.filter_select_grad(widget_dict["g_val"].value, widget_dict["select_g"].value)

            scan_parameters["rx_chan_list"] = (widget_dict["rx_l"].value, widget_dict["rx_c"].value, widget_dict["rx_r"].value)
            
            if self.g.pe == True:
                scan_parameters["pre_emphasis"] = True
                scan_parameters["b0"] = True

                e_dict = self.g.get_yaml_info(info_name = "eddy_E_fit", file = file)
                b0_dict = self.g.get_yaml_info(info_name = "eddy_B0_fit", file = file)

                e_dict, b0_dict = self.change_pars(e_dict, b0_dict, widget_dict["x_e_factor"].value, widget_dict["x_b0_factor"].value, widget_dict["y_e_factor"].value, widget_dict["y_b0_factor"].value, widget_dict["z_e_factor"].value, widget_dict["z_b0_factor"].value) 
                scan_parameters["PE_E_pars"] = e_dict
                scan_parameters["PE_B0_pars"] = b0_dict
                
                
                save_path = Path(file)
                with open(save_path / "eddy_E.yaml", "w") as f:
                    yaml.safe_dump(e_dict, f, sort_keys=False)
                self.g.log_msg(f"Saved into {file}/ eddy_E.yaml \n {e_dict}")
                with open(save_path / "eddy_B0.yaml", "w") as f:
                    yaml.safe_dump(b0_dict, f, sort_keys=False)
                self.g.log_msg(f"Saved into {file}/ eddy_B0.yaml \n {b0_dict}")
                
                    
            else:

                
                scan_parameters["pre_emphasis"] = False
                scan_parameters["b0"] = False

        if scan_type in ["rare",'rare_pe']:
            if scan_type == "rare_pe":
                scan_parameters["file_parsE"] = widget_dict["filepath_E"].value
                scan_parameters["file_parsB"] = widget_dict["filepath_B"].value

            #     if widget_dict["soft_pulse_type"].value == "SLR":
            #         scan_parameters["a_90"], scan_parameters["t_90"], scan_parameters["bw_90"] = self.g.get_yaml_info(info_name = "SLR", file = file).values()
            # else:
                if widget_dict["soft_pulse_type"].value == "3-lobe sinc":
                    scan_parameters["a_90"], scan_parameters["t_90"], scan_parameters["bw_90"] = self.g.get_yaml_info(info_name = "3_lobe", file = file).values()
                if widget_dict["soft_pulse_type"].value == "5-lobe sinc":
                    scan_parameters["a_90"], scan_parameters["t_90"], scan_parameters["bw_90"] = self.g.get_yaml_info(info_name = "5_lobe", file = file).values()
                if widget_dict["soft_pulse_type"].value == "SLR":
                    scan_parameters["a_90"], scan_parameters["t_90"], scan_parameters["bw_90"] = self.g.get_yaml_info(info_name = "SLR", file = file).values()

            
            scan_parameters["t_90"] = float(scan_parameters["t_90"])
            scan_parameters["a_180"] = 2*scan_parameters["a_90"]
            scan_parameters["t_180"] = scan_parameters["t_90"]            

            scan_parameters["t_echo"] = widget_dict["ft_e"].value *1e-3

            
            scan_parameters["t_dw"] = 1 / (widget_dict["spectral_width"].value * 1e3)

            scan_parameters["n_samples"] = widget_dict["resolution_read"].value
            scan_parameters["n_phase"] = widget_dict["resolution_phase1"].value

            g_cal_read = self.g.get_g_strength(widget_dict["FOV_read"].value, widget_dict["select_g_read"].value, file = file)
            g_cal_p1 = self.g.get_g_strength(widget_dict["FOV_phase1"].value, widget_dict["select_g_phase1"].value, file = file)
            g_cal_s = self.g.get_g_strength(widget_dict["slice_thickness"].value, widget_dict["select_g_slice"].value, file = file)

            g_read = 1 / (g_cal_read * (widget_dict["FOV_read"].value * 1e-3) * scan_parameters["t_dw"])
            g_phase1 = self.get_g_phase((widget_dict["t_phase"].value * 1e-6), widget_dict["t_ramp"].value*1e-6, scan_parameters["n_phase"], (widget_dict["FOV_phase1"].value * 1e-3), g_cal_p1)
            
            scan_parameters["g_read"] = self.filter_select_grad(g_read, widget_dict["select_g_read"].value)
            scan_parameters["g_phase"] = self.filter_select_grad(g_phase1, widget_dict["select_g_phase1"].value)


            scan_parameters["g_crush"] = [0,0,0.2]
            scan_parameters["g_spoil"] = [0,0,0.2]
            
            gss_90 = (scan_parameters["bw_90"]) / ((widget_dict["slice_thickness"].value * 1e-3) * g_cal_s) # 3000 Hz as the rf frequency bandwidth - set this from MR solutions
            scan_parameters["g_slice"] = self.filter_select_grad(gss_90, widget_dict["select_g_slice"].value)

            offset_freq = (widget_dict["slice_offset"].value * 1e-3) * g_cal_s * gss_90
            self.g.log_msg(f"{offset_freq} Hz, g_cal_s {g_cal_s}, gss_90 {gss_90}")

            slice_sep = (widget_dict["slice_separation"].value * 1e-3) * g_cal_s * gss_90
            
            
            if widget_dict["slice_offset"].value == 0:
                scan_parameters["f_slice_offset"] = 0
            else:                
                scan_parameters["f_slice_offset"] = offset_freq


            if widget_dict["num_slices"].value != 1:
                indices = np.arange(-(widget_dict["num_slices"].value/2) + 0.5, (widget_dict["num_slices"].value/2))
                slices_list = (widget_dict["slice_offset"].value + indices * widget_dict["slice_separation"].value)                    
                freq_offset_list = offset_freq + indices * slice_sep
            else:
                if scan_parameters["f_slice_offset"] == 0:
                    indices = np.array([0])
                else:
                    indices = np.array([1])
                slices_list = indices * widget_dict["slice_offset"].value
                freq_offset_list = indices * (scan_parameters["f_slice_offset"])
            scan_parameters["freq_offset_list"] = freq_offset_list
            scan_parameters["slices_list"] = slices_list

            if scan_type == "rare_pe":
                scan_parameters["pre_emphasis"] = widget_dict["use_pe"].value
                if "use_b0" in widget_dict.keys():
                    scan_parameters["b0"] = widget_dict["use_b0"].value
                          
        if scan_type in ['2DSE_FOV', '2DSE_multi', "2DSE_multiSLR"]:

            if widget_dict["soft_pulse_type"].value == "3-lobe sinc":
                scan_parameters["a_90"], scan_parameters["t_90"], scan_parameters["bw_90"] = self.g.get_yaml_info(info_name = "3_lobe", file = file).values()
            if widget_dict["soft_pulse_type"].value == "5-lobe sinc":
                scan_parameters["a_90"], scan_parameters["t_90"], scan_parameters["bw_90"] = self.g.get_yaml_info(info_name = "5_lobe", file = file).values()
            if widget_dict["soft_pulse_type"].value == "SLR":
                scan_parameters["a_90"], scan_parameters["t_90"], scan_parameters["bw_90"] = self.g.get_yaml_info(info_name = "SLR", file = file).values()
            
            scan_parameters["t_90"] = float(scan_parameters["t_90"])
            scan_parameters["a_180"] = 2*scan_parameters["a_90"]
            scan_parameters["t_180"] = scan_parameters["t_90"]
                
            scan_parameters["t_dw"] = 1 / (widget_dict["spectral_width"].value * 1e3)
            
            scan_parameters["n_samples"] = widget_dict["resolution_read"].value
            scan_parameters["n_phase"] = widget_dict["resolution_phase1"].value

            g_cal_read = self.g.get_g_strength(widget_dict["FOV_read"].value, widget_dict["select_g_read"].value, file = file) 
            g_cal_p1 = self.g.get_g_strength(widget_dict["FOV_phase1"].value, widget_dict["select_g_phase1"].value, file = file)
            g_cal_s = self.g.get_g_strength(widget_dict["slice_thickness"].value, widget_dict["select_g_slice"].value, file = file)
            
            t_phase = 0.001
            scan_parameters["t_phase"] = t_phase

            g_read = 1 / (g_cal_read * (widget_dict["FOV_read"].value * 1e-3) * scan_parameters["t_dw"])
            g_phase1 = self.get_g_phase(t_phase, widget_dict["t_ramp"].value*1e-6, scan_parameters["n_phase"], (widget_dict["FOV_phase1"].value * 1e-3), g_cal_p1)
            
            scan_parameters["g_read"] = self.filter_select_grad(g_read, widget_dict["select_g_read"].value)
            scan_parameters["g_phase"] = self.filter_select_grad(g_phase1, widget_dict["select_g_phase1"].value)


            gss_90 = (scan_parameters["bw_90"]) / ((widget_dict["slice_thickness"].value * 1e-3) * g_cal_s) # 3000 Hz as the rf frequency bandwidth - set this from MR solutions
            scan_parameters["g_slice"] = self.filter_select_grad(gss_90, widget_dict["select_g_slice"].value)

            offset_freq = (widget_dict["slice_offset"].value * 1e-3) * g_cal_s * gss_90
            self.g.log_msg(f"{offset_freq} Hz, g_cal_s {g_cal_s}, gss_90 {gss_90}")

            slice_sep = (widget_dict["slice_separation"].value * 1e-3) * g_cal_s * gss_90
            
            
            if widget_dict["slice_offset"].value == 0:
                scan_parameters["f_slice_offset"] = 0
            else:                
                scan_parameters["f_slice_offset"] = offset_freq

            if scan_type in ['2DSE_multi', "2DSE_multiSLR"]:

                if widget_dict["num_slices"].value != 1:
                    indices = np.arange(-(widget_dict["num_slices"].value/2) + 0.5, (widget_dict["num_slices"].value/2))
                    slices_list = (widget_dict["slice_offset"].value + indices * widget_dict["slice_separation"].value)                    
                    freq_offset_list = offset_freq + indices * slice_sep
                else:
                    if scan_parameters["f_slice_offset"] == 0:
                        indices = np.array([0])
                    else:
                        indices = np.array([1])
                    slices_list = indices * widget_dict["slice_offset"].value
                    freq_offset_list = indices * (scan_parameters["f_slice_offset"])
                scan_parameters["freq_offset_list"] = freq_offset_list
                scan_parameters["slices_list"] = slices_list
            
        if scan_type in ["pulse_cal_SLR", "pulse_cal_SLR_tab"]:
            

            scan_parameters["t_dw"] = 1 / (widget_dict["spectral_width"].value * 1e3)
            
            if widget_dict["soft_pulse_type"].value == "3-lobe sinc":
                widget_dict["t_pulse"].value = 1332
                widget_dict["bw_90"].value = 3000
                
            if widget_dict["soft_pulse_type"].value == "5-lobe sinc":
                
                widget_dict["t_pulse"].value = 2000
                widget_dict["bw_90"].value = 3000
                
                
            scan_parameters["a_90"] = widget_dict["a_rf_slider"].value/100
            scan_parameters["t_90"] = widget_dict["t_pulse"].value * 1e-6

            
            if widget_dict["slice_thickness"].value != 0:
                g_cal_s = self.g.get_g_strength(widget_dict["slice_thickness"].value, widget_dict["select_g_slice"].value, file = file)
                gss_90 = (widget_dict["bw_90"].value) / ((widget_dict["slice_thickness"].value * 1e-3) * g_cal_s)
            else: 
                gss_90 = 0

            scan_parameters["g_slice_select"] = self.filter_select_grad(gss_90, widget_dict["select_g_slice"].value)
        

        
        if scan_type in ["3DSE_FOV"]:
            scan_parameters["a_90"], scan_parameters["t_90"] = self.g.get_yaml_info(info_name = "hardpulse_90_tx0", file = file).values()
            scan_parameters["a_180"], scan_parameters["t_180"] = self.g.get_yaml_info(info_name = "hardpulse_180_tx0", file = file).values()
            scan_parameters["t_90"] = float(scan_parameters["t_90"])
            scan_parameters["t_180"] = float(scan_parameters["t_180"])
            
            scan_parameters["t_dw"] = 1 / (widget_dict["spectral_width"].value * 1e3)
            scan_parameters["n_samples"] = widget_dict["resolution_read"].value
            scan_parameters["n_phase1"] = widget_dict["resolution_phase1"].value
            scan_parameters["n_phase2"] = widget_dict["resolution_phase2"].value
            scan_parameters["t_phase"] = widget_dict["t_phase"].value * 1e-6
            self.g.log_msg(f"scan_parameters[t_phase], {scan_parameters["t_phase"]}")
            
            rf_freq = 3000

            g_cal_read = self.g.get_g_strength(1, widget_dict["select_g_read"].value, file = file) # 1 as needs to be positive
            g_cal_p1 = self.g.get_g_strength(1, widget_dict["select_g_phase1"].value, file = file) # 1 as needs to be positive
            g_cal_p2 = self.g.get_g_strength(1, widget_dict["select_g_phase2"].value, file = file) # 1 as needs to be positive

            self.g.log_msg(f"{g_cal_read}, {g_cal_p1}, {g_cal_p2}")
            g_read = (1) / (g_cal_read * (widget_dict["FOV_read"].value * 1e-3) * scan_parameters["t_dw"])
            g_phase1 = self.get_g_phase(scan_parameters["t_phase"], widget_dict["t_ramp"].value*1e-6, scan_parameters["n_phase1"], (widget_dict["FOV_phase1"].value * 1e-3), g_cal_p1)
            g_phase2 = self.get_g_phase(scan_parameters["t_phase"], widget_dict["t_ramp"].value*1e-6, scan_parameters["n_phase2"], (widget_dict["FOV_phase2"].value * 1e-3), g_cal_p2)
            
            scan_parameters["g_read"] = self.filter_select_grad(g_read, widget_dict["select_g_read"].value)
            scan_parameters["g_phaseread"] = self.filter_select_grad(-g_read/2, widget_dict["select_g_read"].value)

            scan_parameters["g_phase1"] = self.filter_select_grad(g_phase1, widget_dict["select_g_phase1"].value)
            scan_parameters["g_phase2"] = self.filter_select_grad(g_phase2, widget_dict["select_g_phase2"].value)
            
            scan_parameters["g_crush"] = [0,0,0.2]
            scan_parameters["g_spoil"] = [0,0,0.2]

        if scan_type in ["1DSE_FOV"]:
            scan_parameters["a_90"], scan_parameters["t_90"] = self.g.get_yaml_info(info_name = "hardpulse_90_tx0", file = file).values()
            scan_parameters["a_180"], scan_parameters["t_180"] = self.g.get_yaml_info(info_name = "hardpulse_180_tx0", file = file).values()

            scan_parameters["t_dw"] = 1 / (widget_dict["spectral_width"].value * 1e3)
            scan_parameters["n_samples"] = widget_dict["resolution"].value

            rf_freq = 3000
            
            g_cal = self.g.get_g_strength(widget_dict["FOV_read"].value, widget_dict["select_xyz"].value, file = file) 
            g_val = (1) / (g_cal * (widget_dict["FOV_read"].value * 1e-3) * scan_parameters["t_dw"])
            
            scan_parameters["g_read"] = self.filter_select_grad(g_val, widget_dict["select_xyz"].value)

        if scan_type == "B0_shim":
            B0_max = self.g.get_yaml_info(info_name = "current_cals", file = file)["b0grad"]
            scan_parameters["b0_list"] = np.linspace(0, B0_max, widget_dict["data_points"].value)
            scan_parameters["t_dw"] = 1 / (widget_dict["spectral_width"].value * 1e3)
            scan_parameters["a_90"], scan_parameters["t_90"] = self.g.get_yaml_info(info_name = "hardpulse_90_tx0", file = file).values()
            
        # elif "select_xyz" in widget_dict.keys():
        #     select_exists = True

        #     scan_parameters["g_read"] = self.filter_select_grad(widget_dict["grad_select"].value, widget_dict["select_xyz"].value)

        #     if "g_crusher" in widget_dict.keys():
        #         scan_parameters["g_crusher"] = self.filter_select_grad(widget_dict["g_crusher"].value, widget_dict["select_xyz"].value)

        # -------- get parameters from widget inputs --------
        for key, widget in widget_dict.items():
            if key in ["select_xyz", "grad_select", "width_threshold", "x_offset", "y_offset", "z_offset", "g_read", "g_phase1", "g_phase2", "select_g_read", "select_g_phase1", "select_g_phase2", "n_phase2", "n_phase1", "g_phaseread", "a_rf_slider", "g_phase", "g_slice_90", "g_slice_180", "spectral_width", "resolution", "FOV_read", "t_pulse", "use_pe", "data_points", "rx_l", "rx_c", "rx_r", ]:
                # "x_e_factor", "y_e_factor", "z_e_factor", "x_b0_factor", "y_b0_factor", "z_b0_factor"
                continue
            if key == "g_crusher" and select_exists:
                continue
            if key == "autosave":
                self.g.autosave = widget_dict["autosave"].value
                self.g.log_msg("AUTOSAVE ON")

            else:    
                value = widget.value # gets the widget value from the input box
            
                if key in self.p.input_dict:  # check if this parameter has a conversion factor
                    if 'factor' in self.p.input_dict[key]:
                        value = value * self.p.input_dict[key]['factor']
    
                scan_parameters[key] = value
                
        if non_input_params != None:
            self.g.log_msg(f"non_input_params = {non_input_params}, {type(non_input_params)}")
            scan_parameters.update(non_input_params) # adds on the set parameters

        scan_parameters["scan_type"] = scan_type
        self.g.scan_parameters = scan_parameters
        self.g.log_msg(f"{scan_parameters}")
        return scan_parameters

    
    def print_progress(self, p, l):
        """
        takes the progress info from the run seq and updates if the progress has changed 
        """
        
        self.l_listed.append(l)
        total = len(self.l_listed)
        if total > 2:
            if self.l_listed[-2] == self.l_listed[-1]:
                self.same_l += 1
                if self.same_l > 10 and self.done == False:
                    self.g.log_msg(f"This run will take: {round(self.l_listed[-1], 2)} seconds (= {round(self.l_listed[-1]/60, 2)} minutes).")
                    self.done = True
            else:
                self.same_l = 0
                self.done = False
        
        percent = int((p / l) * 100)
    
        if percent != self.p_last:
            self.panes.progress_bar.value = percent
            self.p_last = percent

        if percent == 100:
            self.p_last = 0
            self.panes.progress_bar.value = 0
            self.l_listed = []
            self.end = 0
            self.same_l = 0

    def plot_seq(self, pulse_file, widget_dict, scan_type, dimensions, slice_dict = None,
                      non_input_params = None, add_amplitudes = 'y', 
                      title_signal = None, title_image = None, pre_file = None, add_scan_param = False, cams = False, file = None):
        """
        PLOTS THE SEQUENCE 
        
        Inputs:
            pulse_file (str): pulse sequence file name (use .py)
            widget_dict (dict): dictionary of all the widgets that will be made for the panel inputs 
            scan_type (str): name of scan
            dimensions (int): number of dimensions of the image
            add_amplitudes (str): tell whether to give the RF pulse amplitudes in the parameters to be printed
                "y" will give it and any other str will not
        
        """
        if self.g.seq_on == True:
            self.g.status_msg("Error: Seq in Use")
            self.g.log_msg("Error: Running Sequence, retry plots after run.")
            return
            
        
        if file == None:
            file = GLOBALS_DIR
        
        try:
            scan_parameters = self.create_scan_parameters(widget_dict, non_input_params, scan_type, file = file)
        except Exception:
            tb = traceback.format_exc()
            self.g.log_msg("Error in scan parameters:")
            self.g.log_msg(tb)
            self.g.status_msg("Error in scan parameters")
            return
        # ---------- check the gradients are within the limits ----------
        try:

            if scan_type in ['1DSE']:
                g_read = self.filter_select_grad(widget_dict["grad_select"].value, widget_dict["select_xyz"].value)
                self.g.gradient_boundaries(*g_read, file = file)
                # add the types in here as it goes to check the gradients
            elif scan_type in ['measure_ec', "compile_ec"]:
                g_read = self.filter_select_grad(widget_dict["g_val"].value, widget_dict["select_g"].value)
                self.g.gradient_boundaries(*g_read, file = file)
                
            elif scan_type == "1DSEsoft":
                self.g.gradient_boundaries(*scan_parameters["g_read"], file = file)
            elif scan_type == '1DSE_FOV':
                self.g.gradient_boundaries(*scan_parameters["g_read"], file = file)
            elif scan_type == 'FID':
                # gradients checked in scan_parameters as not accounted for in widget_dict
                pass

            elif scan_type in ['rare', 'rare_pe']:
                self.g.gradient_boundaries(*scan_parameters["g_read"], file = file)
                self.g.gradient_boundaries(*scan_parameters["g_slice"], file = file)
                self.g.gradient_boundaries(*scan_parameters["g_phase"], file = file)
                self.g.gradient_boundaries(*scan_parameters["g_crush"], file = file)
                self.g.gradient_boundaries(*scan_parameters["g_spoil"], file = file)

            elif scan_type in ["B0_shim"]:
                pass # B0 limit is handled in the pulse sequence -- can write it in here if I have time


            elif scan_type in ["3DSE_FOV"]: 

                self.g.gradient_boundaries(*scan_parameters["g_read"], file = file)
                self.g.gradient_boundaries(*scan_parameters["g_phaseread"], file = file)
                self.g.gradient_boundaries(*scan_parameters["g_phase1"], file = file)
                self.g.gradient_boundaries(*scan_parameters["g_phase2"], file = file)
                self.g.gradient_boundaries(*scan_parameters["g_crush"], file = file)
                self.g.gradient_boundaries(*scan_parameters["g_spoil"], file = file)

            elif scan_type in ["2DSE_FOV", "2DSE_multi", "2DSE_multiSLR"]:
                self.g.gradient_boundaries(*scan_parameters["g_read"], file = file)
                self.g.gradient_boundaries(*scan_parameters["g_phase"], file = file)
                if "g_slice" in scan_parameters.keys():
                    self.g.gradient_boundaries(*scan_parameters["g_slice"], file = file)

                # ADD SPOILER IN IF REQ
            elif scan_type.lower() == 'wobble':
                pass # no gradients used
                
            elif scan_type == 'grad_cal':
                g_read = self.filter_select_grad(widget_dict["grad_select"].value, widget_dict["select_xyz"].value)
                g_crusher = self.filter_select_grad(widget_dict["g_crusher"].value, widget_dict["select_xyz"].value)
                self.g.gradient_boundaries(*g_read, file = file)
                self.g.gradient_boundaries(*g_crusher, file = file)
            elif scan_type in ["soft_pulse_cal", "pulse_cal", "pulse_cal_SLR", "pulse_cal_SLR_tab"]:
                self.g.gradient_boundaries(*scan_parameters["g_slice_select"], file = file)
                
            else: 
                raise ValueError("Add gradient boundaries to run_seq in UIfunctions.py for gradient safety.")
            
        except Exception:
            tb = traceback.format_exc()
            self.g.log_msg("Error in Gradients:")
            self.g.log_msg(tb)
            self.g.status_msg("Error in Gradients")
            return


        # # ---------- attach panel logging handler for DEBUG in pulse seq file ----------

        logger_name = os.path.splitext(os.path.basename(pulse_file))[0]
        logger = logging.getLogger(logger_name)
        logger.setLevel(logging.DEBUG)
        
        panel_handler = PanelLogHandler(self.g)
        panel_handler.setLevel(logging.DEBUG)
        panel_handler.setFormatter(logging.Formatter(
            "[%(levelname)s] %(name)s: %(message)s"
        ))
        
        if not any(isinstance(h, PanelLogHandler) for h in logger.handlers):
            logger.addHandler(panel_handler)
        
        logger.propagate = False

        
        # ---------- load in the pulse seq file and parameters ---------
        try:

            self.g.plotlog_msg(f"file: {GLOBALS_DIR+'../'+pre_file+pulse_file}")
            seq = Sequence(GLOBALS_DIR+'../'+pre_file + pulse_file)
            seq.loadpar(GLOBALS_DIR+'frequency_tx0.yaml')
            if scan_type != "FID":
                seq.loadpar(GLOBALS_DIR+'shims.yaml')
            if scan_type in ["FID", "grad_cal"]:
                seq.loadpar(GLOBALS_DIR+'hardpulse_180_tx0.yaml')

                seq.loadpar(GLOBALS_DIR+'hardpulse_90_tx0.yaml')
                
            
            if add_scan_param != False:
                for k,v in add_scan_param.items():
                    scan_parameters[k] = v
            self.g.scan_parameters = scan_parameters
            self.g.plotlog_msg(f"scan parameters: {scan_parameters}")


            seq.setpar(**scan_parameters) # ** unpacks the scan parameters dict for the input to the pulse seq file
            scan_info_dict = self.p.scan_info_maker(scan_type, dimensions, self.g.scan_parameters, add_amplitudes, file = file)
            self.g.scan_info_dict = scan_info_dict
            self.g.log_msg("PLOTTING !!!!!!!!!!")

            self.panes.seq_plot = seq.plot()

            
        except Exception:
            tb = traceback.format_exc()
            self.g.log_msg("Error in Plotting:")
            self.g.status_msg("Error in Plotting")
            self.g.log_msg(tb)
            return


    
   
    async def run_seq(self, pulse_file, widget_dict, scan_type, dimensions, slice_dict = None,
                      non_input_params = None, add_amplitudes = 'y', 
                      title_signal = None, title_image = None, pre_file = None, add_scan_param = False, cams = False, file = None):
        """
        Calls the python code with the pulse sequence in, loads the global parameters, sets the variable 
        parameters from the inputted values in the panel
        and then runs the pulse sequence. The plots will be created using the update_plots function and
        will update with each press of the run button.

        Inputs:
            pulse_file (str): pulse sequence file name (use .py)
            widget_dict (dict): dictionary of all the widgets that will be made for the panel inputs 
            scan_type (str): name of scan
            dimensions (int): number of dimensions of the image
            add_amplitudes (str): tell whether to give the RF pulse amplitudes in the parameters to be printed
                "y" will give it and any other str will not
        
        """
        if file == None:
            file = GLOBALS_DIR
        
        try:
            scan_parameters = self.create_scan_parameters(widget_dict, non_input_params, scan_type, file = file)
        except Exception:
            tb = traceback.format_exc()
            self.g.log_msg("Error in scan parameters:")
            self.g.log_msg(tb)
            self.g.status_msg("Error in scan parameters")
            return
        # ---------- check the gradients are within the limits ----------
        try:

            if scan_type in ['1DSE']:
                g_read = self.filter_select_grad(widget_dict["grad_select"].value, widget_dict["select_xyz"].value)
                self.g.gradient_boundaries(*g_read, file = file)
                # add the types in here as it goes to check the gradients
            elif scan_type in ['measure_ec', "compile_ec"]:
                g_read = self.filter_select_grad(widget_dict["g_val"].value, widget_dict["select_g"].value)
                self.g.gradient_boundaries(*g_read, file = file)
                
            elif scan_type == "1DSEsoft":
                self.g.gradient_boundaries(*scan_parameters["g_read"], file = file)
            elif scan_type == '1DSE_FOV':
                self.g.gradient_boundaries(*scan_parameters["g_read"], file = file)
            elif scan_type == 'FID':
                # gradients checked in scan_parameters as not accounted for in widget_dict
                pass
            elif scan_type in ["B0_shim"]:
                pass # B0 limit is handled in the pulse sequence -- can write it in here if I have time

            elif scan_type in ['rare', 'rare_pe']:
                self.g.gradient_boundaries(*scan_parameters["g_read"], file = file)
                self.g.gradient_boundaries(*scan_parameters["g_slice"], file = file)
                self.g.gradient_boundaries(*scan_parameters["g_phase"], file = file)
                self.g.gradient_boundaries(*scan_parameters["g_crush"], file = file)
                self.g.gradient_boundaries(*scan_parameters["g_spoil"], file = file)


            elif scan_type in ["3DSE_FOV"]: 

                self.g.gradient_boundaries(*scan_parameters["g_read"], file = file)
                self.g.gradient_boundaries(*scan_parameters["g_phaseread"], file = file)
                self.g.gradient_boundaries(*scan_parameters["g_phase1"], file = file)
                self.g.gradient_boundaries(*scan_parameters["g_phase2"], file = file)
                self.g.gradient_boundaries(*scan_parameters["g_crush"], file = file)
                self.g.gradient_boundaries(*scan_parameters["g_spoil"], file = file)

            elif scan_type in ["2DSE_FOV", "2DSE_multi", "2DSE_multiSLR"]:
                self.g.gradient_boundaries(*scan_parameters["g_read"], file = file)
                self.g.gradient_boundaries(*scan_parameters["g_phase"], file = file)
                if "g_slice" in scan_parameters.keys():
                    self.g.gradient_boundaries(*scan_parameters["g_slice"], file = file)

                # ADD SPOILER IN IF REQ
            elif scan_type.lower() == 'wobble':
                pass # no gradients used
                
            elif scan_type == 'grad_cal':
                g_read = self.filter_select_grad(widget_dict["grad_select"].value, widget_dict["select_xyz"].value)
                g_crusher = self.filter_select_grad(widget_dict["g_crusher"].value, widget_dict["select_xyz"].value)
                self.g.gradient_boundaries(*g_read, file = file)
                self.g.gradient_boundaries(*g_crusher, file = file)
            elif scan_type in ["soft_pulse_cal", "pulse_cal", "pulse_cal_SLR", "pulse_cal_SLR_tab"]:
                self.g.gradient_boundaries(*scan_parameters["g_slice_select"], file = file)
                
            else: 
                raise ValueError("Add gradient boundaries to run_seq in UIfunctions.py for gradient safety.")
            
        except Exception:
            tb = traceback.format_exc()
            self.g.log_msg("Error in Gradients:")
            self.g.log_msg(tb)
            self.g.status_msg("Error in Gradients")
            return


        # # ---------- attach panel logging handler for DEBUG in pulse seq file ----------

        logger_name = os.path.splitext(os.path.basename(pulse_file))[0]
        logger = logging.getLogger(logger_name)
        logger.setLevel(logging.DEBUG)
        
        panel_handler = PanelLogHandler(self.g)
        panel_handler.setLevel(logging.DEBUG)
        panel_handler.setFormatter(logging.Formatter(
            "[%(levelname)s] %(name)s: %(message)s"
        ))
        
        if not any(isinstance(h, PanelLogHandler) for h in logger.handlers):
            logger.addHandler(panel_handler)
        
        logger.propagate = False

        
        # ---------- load in the pulse seq file and parameters ---------
        try:
            self.g.seq_on = True
            self.g.log_msg(f"file: {GLOBALS_DIR+'../'+pre_file+pulse_file}")
            seq = Sequence(GLOBALS_DIR+'../'+pre_file + pulse_file)
            seq.loadpar(GLOBALS_DIR+'frequency_tx0.yaml')
            if scan_type != "FID":
                seq.loadpar(GLOBALS_DIR+'shims.yaml')
            if scan_type in ["FID", "grad_cal"]:
                seq.loadpar(GLOBALS_DIR+'hardpulse_180_tx0.yaml')

                seq.loadpar(GLOBALS_DIR+'hardpulse_90_tx0.yaml')
            
            
            
            if add_scan_param != False:
                for k,v in add_scan_param.items():
                    scan_parameters[k] = v
            self.g.scan_parameters = scan_parameters
            self.g.log_msg(f"scan parameters: {scan_parameters}")


            seq.setpar(**scan_parameters) # ** unpacks the scan parameters dict for the input to the pulse seq file
            scan_info_dict = self.p.scan_info_maker(scan_type, dimensions, self.g.scan_parameters, add_amplitudes, file = file)
            self.g.scan_info_dict = scan_info_dict
    
            # self.panes.seq_plot = seq.plot()
            
        except Exception:
            tb = traceback.format_exc()
            self.g.log_msg("Error in Inputs:")
            self.g.status_msg("Error in Inputs")
            self.g.log_msg(tb)
            return
        # ----------- run the pulse sequence --------------
        try:
            if scan_type in ['2DSE_multi', 'rare','rare_pe', "2DSE_multiSLR"]:
                array_kdata = []
                for i, foff in enumerate(scan_parameters["freq_offset_list"]):

                    seq.setpar(f_slice_offset = foff)
                    
                    self.g.status_msg(f"#### Running No. {self.btn_ctrl.total_runs}..., Slice {i+1}") # +1 as 0 starting number system
                    self.g.log_msg(f"*********** Slice {i+1} at offset of {scan_parameters["slices_list"][i]}mm ***********")
    
                    self.btn_ctrl.run_task = asyncio.create_task(seq.run(progress_handler=self.print_progress))
                    self.g.log_msg(f"assigned asyncio, {self.btn_ctrl.run_task}")
    
                    try:
                        data = await self.btn_ctrl.run_task
                    except Exception as e:
                        self.g.log_msg(f"ERROR in seq.run(): {e}")
                        raise
                    array_kdata.append(data)

                    if self.btn_ctrl.stop_requested:
                        self.g.log_msg("Stopping Slice Run (abort detected)")
                        self.g.status_msg(f"Aborted {scan_type} Slicing")
                        break
        
                    
                        
                    self.g.log_msg(f"Data Shape:{np.shape(data)}")
                    self.btn_ctrl.run_task = None
                    self.g.log_msg("After seq.run()")
                    
                    self.g.log_msg("Pulse Sequence Finished.")
                    self.g.status_msg(f"#### Finished Run No. {self.btn_ctrl.total_runs}")


            elif scan_type == "B0_shim":
                array_kdata = []
                for i, B0 in enumerate(scan_parameters["b0_list"]):
                    seq.setpar(B0 = B0)
                    
                    self.g.status_msg(f"#### Running No. {self.btn_ctrl.total_runs}..., B0: {round(B0, 2)}") # +1 as 0 starting number system
                    self.g.log_msg(f"*********** B0: {round(B0, 2)} ***********")
    
                    self.btn_ctrl.run_task = asyncio.create_task(seq.run(progress_handler=self.print_progress))
                    self.g.log_msg(f"assigned asyncio, {self.btn_ctrl.run_task}")
    
                    try:
                        data = await self.btn_ctrl.run_task
                    except Exception as e:
                        self.g.log_msg(f"ERROR in seq.run(): {e}")
                        raise
                    array_kdata.append(data)

                    if self.btn_ctrl.stop_requested:
                        self.g.log_msg("Stopping B0 Run (abort detected)")
                        self.g.status_msg(f"Aborted {scan_type} Slicing")
                        break
        
                    
                        
                    self.g.log_msg(f"Data Shape:{np.shape(data)}")
                    self.btn_ctrl.run_task = None
                    self.g.log_msg("After seq.run()")
                    
                    self.g.log_msg("Pulse Sequence Finished.")
                    self.g.status_msg(f"#### Finished Run No. {self.btn_ctrl.total_runs}")        
                    
                
            elif scan_type != "soft_pulse_cal":
                array_kdata = None
                self.g.status_msg(f"#### Running No. {self.btn_ctrl.total_runs}...")

                
                self.g.log_msg("Before seq.run()")

                self.btn_ctrl.run_task = asyncio.create_task(seq.run(progress_handler=self.print_progress))
                self.g.log_msg(f"assigned asyncio, {self.btn_ctrl.run_task}")
                # data = await self.btn_ctrl.run_task              

                try:
                    data = await self.btn_ctrl.run_task
                except Exception as e:
                    self.g.log_msg(f"ERROR in seq.run(): {e}")
                    raise
                    
                self.g.log_msg(f"Data Shape:{np.shape(data)}")
                self.btn_ctrl.run_task = None
                self.g.log_msg("After seq.run()")
                
                self.g.log_msg("Pulse Sequence Finished.")
                self.g.status_msg(f"#### Finished Run No. {self.btn_ctrl.total_runs}")

                
            else: 
                try:
                    self.btn_ctrl.run_task = asyncio.create_task(
                        seq.run(progress_handler=self.print_progress)
                    )
                    data = await self.btn_ctrl.run_task
                except asyncio.CancelledError:
                    self.g.log_msg("Pulse run cancelled.")
                                    
                
                
                
    
        except asyncio.CancelledError:
            self.g.log_msg("Run sequence cancelled. INSIDE")
            raise



        except Exception as e:
            tb = traceback.format_exc()
            self.g.status_msg("Error in Run Sequence")
            self.g.log_msg("Error in Run Sequence:")
            self.g.log_msg(tb)
            return

        finally:
            self.g.seq_on = False
            self.btn_ctrl.run_task = None
            self.btn_ctrl.stop_requested = False
            self.panes.progress_bar.value = 0
                    
        # ---------- save the data output and load into the plotting function -----------

        self.btn_ctrl.kdata_newest = data
        self.btn_ctrl.array_kdata = array_kdata
        if self.btn_ctrl.array_kdata != None:
            self.g.log_msg("array in kdata cntrl")
        try:
            self.g.seq_on = False
            if scan_type in ["measure_ec", "compile_ec"]:
                self.u.update_plot(seq.data_map, widget_dict, dimensions, scan_type, slice_dict, non_input_params, title_signal, title_image, freq_mm=self.btn_ctrl.freq_mm, calc_width = self.btn_ctrl.calc_width, cams = cams, array_kdata = array_kdata, file = file)
            else:
                self.u.update_plot(data, widget_dict, dimensions, scan_type, slice_dict, non_input_params, title_signal, title_image, freq_mm=self.btn_ctrl.freq_mm, calc_width = self.btn_ctrl.calc_width, cams = cams, array_kdata = array_kdata, file = file)
            if self.g.autosave == True:
                self.g.log_msg("Autosaving ...")
                self.btn_ctrl.save_on_click(None)
        except Exception as e:
            tb = traceback.format_exc()
            self.g.status_msg("Error in Rendering Plots")
            self.g.log_msg("Error in Rendering Plots:")
            self.g.log_msg(tb)
            return
        await asyncio.sleep(0)

    async def run_loop(self, pulse_file, widget_dict, scan_type, dimensions, slice_dict = None,non_input_params=None, add_amplitudes='y',title_signal=None, title_image=None, pre_file=None, file = None):
        try:
            while self.btn_ctrl.run_loop:
                await self.run_seq(pulse_file, widget_dict, scan_type, dimensions, slice_dict = slice_dict,
                    non_input_params = non_input_params, add_amplitudes = add_amplitudes,
                    title_signal = title_signal, title_image = title_image, pre_file = pre_file, file = file)

                await asyncio.sleep(self.btn_ctrl.update_secs)
    
        except asyncio.CancelledError:
            # task was cancelled via abort
            self.g.log_msg("Run Loop Cancelled. Inside log statement.")
            raise   # re-raise for proper task cleanup
    
        finally:
            self.btn_ctrl.run_loop = False


