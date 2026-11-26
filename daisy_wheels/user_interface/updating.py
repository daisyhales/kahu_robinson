from matipo import SEQUENCE_DIR, GLOBALS_DIR
from matipo.sequence import Sequence
import traceback
import numpy as np
import panel as pn
import yaml
import napari 
from matipo.util.fft import fft_reconstruction
from matipo.util.fft import get_freq_spectrum
import logging
from bokeh.models import LinearColorMapper, ColorBar
from bokeh.palettes import Greys256
from bokeh.plotting import figure
import pandas as pd
from .general import general_functions
from .panes import panes
from .oneDfunctions import oneDfunctions
from .twoDfunctions import twoDfunctions
from .threeDfunctions import threeDfunctions
import os
from pathlib import Path


class updating:
    """
    Updates the plots to work with each run of the pulse sequence.
    """
    def __init__(self, general_functions, panes, oneDfunctions, threeDfunctions):
        """
        
        """
        self.g = general_functions
        self.panes = panes
        self.oneDfunctions = oneDfunctions(self.g, self.panes)
        self.threeDfunctions = threeDfunctions(self.g, self.panes)
        self.twoDfunctions = twoDfunctions(self.g, self.panes)

        self.freq_arr = []
        self.img_arr = []
        self.a_rf_arr = []

    @staticmethod
    def get_peak_freq(freq, img):
        """
        finds the max data point of the frequency and img
        """
        idx = np.argmax(np.abs(img))
        return freq[idx], np.abs(img[idx])

    @staticmethod
    def get_g_val(grad, axis):
        """
        gets index of axis and then grabs the gradient value there.
        """
        mapper = {"X": 0, "Y": 1, "Z": 2}
        return grad[mapper[axis]]
        

    def update_plot(self, data, widget_dict, dimensions, scan_type, slice_dict = None, non_input_params = None, title_signal = None, title_image = None, file = None, freq_mm = None, calc_width = 'n', cams = False, array_kdata = None):
        """
        Plots the k space (signal vs time) and real space images from the acquired data and fourier transform of that data 
        - will filter based on dimensions of the acquired image to fourier transform and display the data 
        in the required method from the dimension.
    
        Inputs:
            data (1D np.array): signal data of the pulse sequence.
            widget_dict (dict): dictionary of all the widgets that will be made for the panel inputs 
            dimensions (int): number of dimensions of the image
        """

        scan_parameters = self.g.scan_parameters
        t_dw = scan_parameters['t_dw']
        time = np.linspace(t_dw, len(data)*t_dw, len(data))

        if dimensions == 1:
            try: 
                if scan_type not in ['measure_ec', 'compile_ec']:
                    self.panes.plot_pane.object = self.oneDfunctions.signal_plot1D(time, data, scan_parameters['n_samples'], title_signal)
            except Exception:
                tb = traceback.format_exc()
                self.g.log_msg("Error in Signal Plot:")
                self.g.log_msg(tb)
                return
            if scan_type not in ['measure_ec', 'compile_ec']:        
                freq, img = get_freq_spectrum(signal = data, dwell_time= t_dw)
                
            try: 
                if scan_type == "B0_shim":
                    self.panes.img_pane.object, p_tabs, B0_cal =  self.oneDfunctions.B0_plot(array_kdata, scan_parameters["t_dw"], scan_parameters["apodisation"], scan_parameters["b0_list"])
                    self.g.B0_cal = B0_cal
                    # tabs = pn.Tabs()
                    self.panes.tabs_b0.clear()
                    
                    for name, panel_obj in p_tabs.items():
                        self.panes.tabs_b0.append((name, panel_obj))
                    # self.panes.tabs_b0 = tabs
                    self.g.log_msg(f"tabs b0: {len(self.panes.tabs_b0)}")
                    # self.panes.extras_pane -- put the B0 signal plots into this part

                elif scan_type == "measure_ec":
                    data_map = data.copy()
                    ZERO_FILLING = 200000
                    times_idx = np.geomspace(scan_parameters["t_wait"], scan_parameters["t_end"], num=scan_parameters["n_pulses"])

                    times_idx = np.concatenate(([0], times_idx))
                    list_ = []
                    n_acq = scan_parameters["n_pulses"]+1
                    
                    for i_pulse in range(n_acq):
                        list_.append([(i_pulse+1 + n_acq * i_chan) for i_chan, rx in enumerate(scan_parameters["rx_chan_list"])]) # +1 as started on 1 index not zero
                    df_idx = pd.DataFrame(list_, columns = ["rx_l", "rx_c", "rx_r"])
                    df_idx["times (s)"] = times_idx
                    
                    df_freq = pd.DataFrame(df_idx["times (s)"], columns =["times (s)"]) # skip first one as is for gradient calibration
                    df_rms = pd.DataFrame(df_idx["times (s)"],columns = ["times (s)"])
                    
                    # CALCULATING THE FREQ AND RMS
                    for rx in df_idx.columns[0:3]: # dont do times column
                        df = df_idx[rx]
                        freq_c  = []
                        rms_list  = []
                        for key in df:        

                            data = data_map[key]
                            time = np.linspace(t_dw,len(data)*t_dw, len(data))
                            window = np.exp(-np.pi*scan_parameters["apodisation"]*time)
                            processed_signal = np.concatenate((data_map[key]*window, np.zeros(ZERO_FILLING)))
                            freq, img = get_freq_spectrum(signal = processed_signal, dwell_time= t_dw)
                                
                            RMS = np.sqrt(np.mean(np.abs(data)**2)) # plot this against time for each acq to see how the amplitude changes over time and checks that the flip angle leads to steady-state
                            
                            spectrum_mag = np.abs(img)
                            peak_index = np.argmax(spectrum_mag)
                            centre_freq = freq[peak_index]
                            freq_c.append(centre_freq)
                            rms_list.append(RMS)
                            
                        df_freq[rx] = freq_c
                        df_rms[rx] = rms_list




                    cal_data = self.g.get_yaml_info("g_hzpm_slope.yaml", file)
                    dist = self.g.get_yaml_info("probe_distances.yaml", file)
                    g_hzpm = cal_data[scan_parameters['probe_axis']] * scan_parameters["g_val"]
                    df_freq["g_eddy"] = (df_freq["rx_r"] - df_freq["rx_l"]) / (float(dist["d_r"]) - float(dist["d_l"]))
                    df_freq["E(t)"] = df_freq["g_eddy"]/ g_hzpm 
                    
                    df_freq["B0(t)"] = df_freq["rx_c"]/ g_hzpm    

                    dict_ = {
                        "g_hzpm_measured": float((df_freq["rx_r"][0] - df_freq["rx_l"][0]) / (float(dist["d_r"]) - float(dist["d_l"]))),
                        "g_hzpm_loaded": g_hzpm
                    }
                    save_folder = general_functions.get_yaml_info(info_name = 'folder_names.yaml', file = GLOBALS_DIR)[scan_type]
                    name = f"{scan_parameters['select_g']}grad_{scan_parameters['probe_axis']}eddy"
                    folder = f"data/{save_folder}/{name}"
                    os.makedirs(folder, exist_ok=True)
                    
                    df_freq.to_csv(f"{folder}/df_freq.csv", index=False)
                    df_rms.to_csv(f"{folder}/df_rms.csv", index=False)
                    with open(f"{folder}/g_hzpm.yaml", "w") as f:
                        yaml.safe_dump(dict_, f, sort_keys=False)
                        
                    with open(f"{folder}/seq_par.txt", "w") as f:
                        f.write(str(scan_parameters))

                    height = 600
                    self.panes.analysis_plot[:] = [self.oneDfunctions.eddy_current_plot(df_freq, df_rms, height = height)]

                    self.panes.spectra_plot[:] = [self.oneDfunctions.eddy_spectra_plot(data_map, scan_parameters, df_idx, height = height)]

                    pn.state.curdoc.add_next_tick_callback(lambda: self.oneDfunctions.eddy_decay_plot(df_freq, height))

                    yaml_file = f"{folder}/fitting_pars.yaml"
                    
                    pars = {}
                    if Path(yaml_file).exists():
                        with open(yaml_file) as f:
                            pars = yaml.safe_load(f) or {}
                    
                    pars["folder_name"] = folder
                    
                    with open(yaml_file, "w") as f:
                        yaml.safe_dump(pars, f, sort_keys=False)
                        

                        


                elif scan_type == "compile_ec":
                    data_map = data.copy()
                   
                    ZERO_FILLING = 200000
                    times_idx = np.geomspace(scan_parameters["t_wait"], scan_parameters["t_end"], num=scan_parameters["n_pulses"])

                    times_idx = np.concatenate(([0], times_idx))
                    list_ = []
                    n_acq = scan_parameters["n_pulses"]+1
                    
                    for i_pulse in range(n_acq):
                        list_.append([(i_pulse+1 + n_acq * i_chan) for i_chan, rx in enumerate(scan_parameters["rx_chan_list"])]) # +1 as started on 1 index not zero
                    df_idx = pd.DataFrame(list_, columns = ["rx_l", "rx_c", "rx_r"])
                    df_idx["times (s)"] = times_idx
                    
                    df_freq = pd.DataFrame(df_idx["times (s)"], columns =["times (s)"]) # skip first one as is for gradient calibration
                    df_rms = pd.DataFrame(df_idx["times (s)"],columns = ["times (s)"])
                    
                    # CALCULATING THE FREQ AND RMS
                    for rx in df_idx.columns[0:3]: # dont do times column
                        df = df_idx[rx]
                        freq_c  = []
                        rms_list  = []
                        for key in df:        

                            data = data_map[key]
                            time = np.linspace(t_dw,len(data)*t_dw, len(data))
                            window = np.exp(-np.pi*scan_parameters["apodisation"]*time)
                            processed_signal = np.concatenate((data_map[key]*window, np.zeros(ZERO_FILLING)))
                            freq, img = get_freq_spectrum(signal = processed_signal, dwell_time= t_dw)
                                
                            RMS = np.sqrt(np.mean(np.abs(data)**2)) # plot this against time for each acq to see how the amplitude changes over time and checks that the flip angle leads to steady-state
                            
                            spectrum_mag = np.abs(img)
                            peak_index = np.argmax(spectrum_mag)
                            centre_freq = freq[peak_index]
                            freq_c.append(centre_freq)
                            rms_list.append(RMS)
                            
                        df_freq[rx] = freq_c
                        df_rms[rx] = rms_list

                        
                    cal_data = self.g.get_yaml_info("g_hzpm_slope.yaml", file)
                    dist = self.g.get_yaml_info("probe_distances.yaml", file)
                    g_hzpm = cal_data[scan_parameters['probe_axis']] * scan_parameters["g_val"]
                    df_freq["g_eddy"] = (df_freq["rx_r"] - df_freq["rx_l"]) / (float(dist["d_r"]) - float(dist["d_l"]))
                    df_freq["E(t)"] = df_freq["g_eddy"]/ g_hzpm 
                    
                    df_freq["B0(t)"] = df_freq["rx_c"]/ g_hzpm    

                    dict_ = {
                        "g_hzpm_measured": float((df_freq["rx_r"][0] - df_freq["rx_l"][0]) / (float(dist["d_r"]) - float(dist["d_l"]))),
                        "g_hzpm_loaded": g_hzpm
                    }

                    
                    height = 600
                    self.panes.analysis_plot[:] = [self.oneDfunctions.eddy_current_plot(df_freq, df_rms, height = height)]
                    self.panes.spectra_plot[:] = [self.oneDfunctions.eddy_spectra_plot(data_map, scan_parameters, df_idx, height = height)]
                    pn.state.curdoc.add_next_tick_callback(lambda: self.oneDfunctions.eddy_decay_plot(df_freq, height, scan_parameters["x_e_factor"], scan_parameters["y_e_factor"], scan_parameters["z_e_factor"], scan_parameters["x_b0_factor"], scan_parameters["y_b0_factor"], scan_parameters["z_b0_factor"], compiler = 'y'))



                    
                    
                elif scan_type == "grad_cal":
                    self.panes.img_pane.object, peak_width = self.oneDfunctions.grad_cal_img_plot(freq, img, scan_parameters['n_samples'], 
                                                                                                  widget_dict['select_xyz'].value, scan_parameters['n_samples'], title_image, 
                                                                                                  threshold =(widget_dict['width_threshold'].value/100), file = file)
                    if peak_width == None:
                        self.g.status_msg("Gradient calibration failed: no measurable peak width!!!")
                        return


                    sample_cal = self.g.get_yaml_info(info_name = "sample_calibration", file = file)
                    self.g.log_msg(f"Peak width: {peak_width} Hz -> {peak_width*1e-3} kHz")

                    self.peak_width = peak_width
    

                elif scan_type == "FID":
                    self.panes.img_pane.object, length = self.oneDfunctions.img_plot1D(freq, img, widget_dict['n_samples'].value, select_xyz=None, grad_value=None, title=title_image, scan_type=scan_type, freq_mm=None, g=self.g, file = file)

                elif scan_type in ["pulse_cal", "pulse_cal_SLR", "pulse_cal_SLR_tab"]:
            
                    n_samples = widget_dict['n_samples'].value
            
                    freq, img = get_freq_spectrum(signal = data[0:n_samples], dwell_time= t_dw)
                    freq2, img2 = get_freq_spectrum(signal = data[n_samples:], dwell_time= t_dw)
            
            
                    freq_peak, peak = self.get_peak_freq(freq, img)
                    freq2_peak, peak2 = self.get_peak_freq(freq2, img2)
                
                    var_peak = np.abs(peak - peak2)
                    var_freq = np.abs(freq_peak - freq2_peak)
                
                    self.freq_arr.append(var_freq)
                    self.img_arr.append(var_peak)

                    a_rf = widget_dict['a_rf_slider'].value
                    self.a_rf_arr.append(a_rf/100)
                    
                    p = self.oneDfunctions.signal_soft_plot(time, data, n_samples, a_rf)
                    if scan_type != "pulse_cal_SLR_tab":
                        self.g.status_msg(f"Slice Gradient: {np.round(scan_parameters['g_slice_select'], 2)}")
                    else:
                        self.g.softstatus_msg(f"Slice Gradient: {np.round(scan_parameters['g_slice_select'], 2)}")
                            
                    self.panes.soft_pulse_plot.object = self.oneDfunctions.img_soft_plot(freq, img, freq2, img2, n_samples, a_rf, freq_peak, peak, freq2_peak, peak2)
                    
                else:
                    if "width_threshold" in widget_dict.keys():
                        g_val = self.get_g_val(scan_parameters['g_read'], widget_dict['select_xyz'].value)
                        self.panes.img_pane.object, length = self.oneDfunctions.img_plot1D(freq, img, scan_parameters['n_samples'], widget_dict['select_xyz'].value, g_val, title_image, scan_type, threshold = widget_dict['width_threshold'].value/100, freq_mm = freq_mm, g = self.g, calc_width = calc_width, file = file)
                    else: 
                        self.panes.img_pane.object, length = self.oneDfunctions.img_plot1D(freq, img, widget_dict['n_samples'].value, widget_dict['select_xyz'].value, widget_dict['grad_select'].value, title_image, scan_type, threshold = 0.03, freq_mm = freq_mm, g = self.g, calc_width = calc_width, file = file)
                    
                    if length != None:
                        if scan_type != "pulse_cal_SLR_tab":
                            self.g.log_msg(f"Axis {widget_dict['select_xyz'].value} has length: {length*1e3} mm")
                        else:
                            self.g.softlog_msg(f"Axis {widget_dict['select_xyz'].value} has length: {length*1e3} mm")

            except Exception:
                tb = traceback.format_exc()
                self.g.log_msg("Error in Image Plot:")
                self.g.log_msg(tb)
                return
        if dimensions == "wobble":
            try:
                self.panes.plot_pane.object = self.oneDfunctions.wobble_plot(data, widget_dict, non_input_params, title_signal)
            except Exception:
                tb = traceback.format_exc()
                self.g.log_msg("Error in Signal Plot:")
                self.g.log_msg(tb)
                return

        if dimensions == 2:
            if scan_type in ['2DSE_multi', 'rare','rare_pe', "2DSE_multiSLR"]:
                self.panes.twoimagetabs.clear()
                viewer = napari.Viewer(ndisplay=2)
                
                for i, k in enumerate(array_kdata):

                # self.twoimagetabs
                    
                    size = 700
                    
                    k = k[: self.g.scan_parameters["n_phase"] * self.g.scan_parameters["n_samples"]]
                    k = np.reshape(k, (self.g.scan_parameters["n_phase"], self.g.scan_parameters["n_samples"]))
                    
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
                    self.panes.twoimagetabs.append((f"Slice {self.g.scan_parameters["slices_list"][i]}mm", plots_two))

                #     # OPEN NAPARI IMAGE
                    self.twoDfunctions.signal_2D(scan_parameters, widget_dict, k, time, cams = cams)
                    self.twoDfunctions.open_img_napari(viewer, slice_dict, scan_parameters = self.g.scan_parameters, scan_type = scan_type, var = i, file = file)

                napari.run()

            
            
        if dimensions == 3:
            self.threeDfunctions.signal_3D(self.g.scan_parameters, widget_dict, data, time, cams = cams)
            self.threeDfunctions.open_img_napari(slice_dict, scan_parameters = self.g.scan_parameters, file = file)
            napari.run()
            