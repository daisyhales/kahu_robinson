from matipo import GLOBALS_DIR

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yaml

from pathlib import Path

from scipy.optimize import curve_fit, nnls, lsq_linear
from scipy.signal import find_peaks

import pickle

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

class Compare:
    """
    Class for the creations of the panel elements that can be used with any scan.
    """
    def __init__(self, scan_type):
        self.file = GLOBALS_DIR

        with open(self.file + "fitting_pars.yaml", "r") as f:
            self.MAX_EXP = yaml.safe_load(f)["MAX_EXP"]
            
        with open(self.file + "prefiles.yaml", "r") as f:
            self.pre_file = yaml.safe_load(f)[scan_type]
        
        with open(self.file + "seqnames.yaml", "r") as f:
            self.seqname = yaml.safe_load(f)[scan_type]
            
    @staticmethod
    def get_g_read(axis, g_val):
        grad_dict = {"X": (g_val,0,0), "Y": (0,g_val,0), "Z": (0,0,g_val)}
        return grad_dict[axis]

    async def run_comparison(self, run_info, seq_params):
        """
        
        """

        g_val = seq_params["g_val"]
        dict_ = {}
        f_list = []
        axis = seq_params["OPT_AXIS"]
        N_scans = seq_params["n_scans"]
        for key, info in run_info.items():

            if info["E_pars"] == None:
                p = False
            else: 
                p = True
                
            if info["B_pars"] == None:
                b = False
            else: 
                b = True
            ZERO_FILLING = 200000
            
            
            seq = Sequence(self.pre_file + self.seqname)
            
            # load relevant global variables
            seq.loadpar(GLOBALS_DIR+'frequency_tx0.yaml')
            seq.loadpar(GLOBALS_DIR+'shims.yaml')
            # g_val = 0.1
            
            if axis == "X":
                g_val = g_val*2
                print("g_val doubled for X")
                
            gg_read = self.get_g_read(axis, seq_params["g_val"])
            
            seq.setpar(
                n_scans = 1,
                a_pulse = seq_params["a_pulse"], 
                t_pulse = seq_params["t_pulse"],
                t_dw = seq_params["t_dw"],
                n_samples = seq_params["n_samples"],
                t_acq_delay = seq_params["t_acq_delay"],
                t_grad = seq_params["t_grad"],
                t_ramp = seq_params["t_ramp"],
                t_wait = seq_params["t_wait"],
                n_pulses = seq_params["n_pulses"],
                t_end = seq_params["t_end"], 
                t_next = seq_params["t_next"], # wait between scans 
                g_read = gg_read,
        
                tx_chan = seq_params["tx_chan"],
                rx_chan_list = seq_params["rx_chan_list"],
                pre_emphasis = p,
                b0 = b,
                compare = "y",
                file_pars = GLOBALS_DIR,
                file_cals = GLOBALS_DIR,
                
                file_parsE = info["E_pars"],
                file_parsB = info["B_pars"],
                
            )
            
            pn.extension()
            print(seq.par)
            y_sum = None
        
            for i in range(N_scans):
                y = await seq.run()
                if y_sum is None:
                    y_sum = y
                else:
                    y_sum += y
            y = y_sum/N_scans
        
            # REMAKING THE INDEXING OF THE ACQs
            times = np.geomspace(seq.par.t_wait, seq.par.t_end, num=seq.par.n_pulses)
        
            times = np.concatenate(([0], times))
            list_ = []
            n_acq = seq.par.n_pulses+1
            
            for i_pulse in range(n_acq):
                list_.append([(i_pulse+1 + n_acq * i_chan) for i_chan, rx in enumerate(seq.par.rx_chan_list)]) # +1 as started on 1 index not zero
            
            # print(list_)
            
            df_idx = pd.DataFrame(list_, columns = ["rx_l", "rx_c", "rx_r"])
            df_idx["times (s)"] = times
            # display(df_idx[0:3])
            apodization = 100
        
            df_freq = pd.DataFrame(df_idx["times (s)"], columns =["times (s)"]) # skip first one as is for gradient calibration
            df_rms = pd.DataFrame(df_idx["times (s)"],columns = ["times (s)"])
            
            # CALCULATING THE FREQ AND RMS
            for rx in df_idx.columns[0:3]: # dont do times column
                df = df_idx[rx]
                freq_c  = []
                rms_list  = []
                for key in df:        
                    t_dw = seq.par.t_dw
                    data = seq.data_map[key]
                    time = np.linspace(t_dw,len(data)*t_dw, len(data))
                    window = np.exp(-np.pi*apodization*time)
                    processed_signal = np.concatenate((seq.data_map[key]*window, np.zeros(ZERO_FILLING)))
                    freq, img = get_freq_spectrum(signal = processed_signal, dwell_time= seq.par.t_dw)
                        
                    RMS = np.sqrt(np.mean(np.abs(data)**2)) # plot this against time for each acq to see how the amplitude changes over time and checks that the flip angle leads to steady-state
                    
                    spectrum_mag = np.abs(img)
                    peak_index = np.argmax(spectrum_mag)
                    centre_freq = freq[peak_index]
                    freq_c.append(centre_freq)
                    rms_list.append(RMS)
                    
                df_freq[rx] = freq_c
                df_rms[rx] = rms_list
        
            # PLOTTING TO SHOW THE FREQ AND RMS
            fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=True)
            
            cols = ["rx_l", "rx_c", "rx_r"]
            colors = {
                "rx_l": "tab:blue",
                "rx_c": "tab:orange",
                "rx_r": "tab:green",
            }
            
            for i, col in enumerate(cols):
            
                # Frequency row
                axes[0, i].scatter(
                    df_freq["times (s)"][1:],
                    df_freq[col][1:],
                    color=colors[col]
                )
                axes[0, i].set_xscale("log")
                axes[0, i].set_title(col)
                axes[0, i].set_ylabel("Frequency (Hz)")
            
                # RMS row
                
                axes[1, i].scatter(df_rms["times (s)"][1:], df_rms[col][1:], color=colors[col])
                axes[1, i].set_xscale("log")
                axes[1, i].set_xlabel("Time (s)")
                axes[1, i].set_ylabel("RMS")
        
            # fig.suptitle(f"Eddy Current Analysis, PE: {e}, BE: {b}, Half: {half}", fontsize=16)
            fig.suptitle(f"{axis} Eddy {axis} Axis, PE: {p}, BE: {b}, G_val: {g_val}", fontsize=16)
            plt.tight_layout()
            plt.show()
        
            fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=False)
        
            cols = ["rx_l", "rx_c", "rx_r"]
            
            for i, col in enumerate(cols):
            
                df = df_idx[col]
            
                signals = []
                spectra = []
            
                for key in df:
                    signal = seq.data_map[key]
            
                    freq, spectrum = get_freq_spectrum(signal=signal,dwell_time=seq.par.t_dw)
            
                    signals.append(signal)
                    spectra.append(spectrum)
            
                signals = np.array(signals)
                spectra = np.array(spectra)
            
                axes[0, i].imshow(np.abs(signals),aspect="auto",cmap="viridis")
                axes[0, i].set_title(f"{col} signals")
                axes[0, i].set_ylabel("Scan")
            
                axes[1, i].imshow(np.abs(spectra),aspect="auto",cmap="viridis")
                axes[1, i].set_title(f"{col} spectra")
                axes[1, i].set_ylabel("Scan")
                axes[1, i].set_xlabel("Sample")
            # fig.suptitle(f"Eddy Current Analysis, PE: {e}, BE: {b}, Half: {half}", fontsize=16)
            fig.suptitle(f"{axis} Eddy {axis} Axis, PE: {p}, BE: {b}, G_val: {g_val}", fontsize=16)
        
            plt.tight_layout()
            plt.show()
            
            with open("C:/Users/halesda/OneDrive - Victoria University of Wellington - STAFF/Desktop/kahu-workspace/notebooks/daisy/kahu_eddy/pre_emph_eddy/inv_laplace/g_hzpm_slope.yaml", "r") as f:
                cal_data = yaml.safe_load(f)

            g_hzpm = cal_data[axis]*seq_params["g_val"]

            
            df_freq["g_eddy"] = (df_freq["rx_r"] - df_freq["rx_l"]) / (float(seq_params["d_r"]) - float(seq_params["d_l"]))
            df_freq["E(t)"] = df_freq["g_eddy"]/ g_hzpm 
            
            df_freq["B0(t)"] = df_freq["rx_c"]/ g_hzpm
        
            f_list.append(df_freq)

        
            dict_[key] = {
                "seq_par":str(seq.par),
                "freq": df_freq.copy(),
                "rms": df_rms.copy(),
                "g_hzpm": g_hzpm,
            }
            
        return dict_, f_list



    def plot_comparison(self, run_info, f_list, axis):
        """
        """
        titles = []
        colours = []
        
        n = len(run_info)
        colours_extra = plt.cm.tab10(np.arange(n))
        count = 0
        for key, info in run_info.items():
            titles.append(info["title"])
            if "colour" in info.keys():
                colours.append(info["colour"])
            else:
                colours.append(colours_extra[count])
            
            count = count+1
                

        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        for df, label, colour in zip(f_list, titles, colours):

            N = 3
        
            baseline = df["g_eddy"].tail(N).mean()
            ged = df["g_eddy"] - baseline
        
            axes[0].plot(df["times (s)"][1:] ,ged[1:] * 1e-3, label=label, color=colour)
        
            baselineb0 = df["rx_c"].tail(N).mean()
            rxc = df["rx_c"] - baselineb0
            print(f"{label}, g_eddy max:", np.max(np.abs(ged[1:]))*1e-3, "kHz/m       offset max:", np.max(np.abs(rxc[1:]))*1e-3, "kHz")
        
            axes[1].plot(df["times (s)"][1:], rxc[1:] * 1e-3, label=label, color=colour)
        axes[0].axhline(y=0, color="black", linestyle="--", linewidth=1, label="Ideal")            
        axes[1].axhline(y=0, color="black", linestyle="--", linewidth=1, label="Ideal")            
        
        axes[0].set_xscale("log")
        axes[0].set_xlabel("Time (s)")
        axes[0].set_ylabel("Field Gradient (kHz/m)")
        axes[0].set_title("Field Gradient Decay")
        axes[0].legend(fontsize=8)
        
        axes[1].set_xscale("log")
        axes[1].set_xlabel("Time (s)")
        axes[1].set_ylabel("Field Offset (kHz)")
        axes[1].set_title("Field Offset Decay")
        axes[1].legend(fontsize=8)
        
        plt.show()


        
    
    @staticmethod
    def get_yaml_info(info_name = None, file = None):
        """
        Retrieves the infomation stored in a yaml file and returns it as a dictionary (val).
    
        Inputs:
            info_name (str): name of the yaml file to retrieve info from - can end with or without .yaml and work
                but needs to be a yaml file.
            file (str): name of the filepath to get the the info_name.yaml with the information in 
                (should be stored in a globals file on your kahu workspace)
            
        Output:
            val (dictionary): dictionary of the information stored in the yaml file asked to be retrieved from 
        
        """
        if file == None:
            file = "C:/Users/halesda/OneDrive - Victoria University of Wellington - STAFF/Desktop/kahu-workspace/globals"
    
        if info_name == None:
            info_name = 'shims'
        try:
            if ".yaml" in info_name:
                with open(f"{file}/{info_name}", "r") as f:
                    val = yaml.safe_load(f) # list
                return val
            else:
                with open(f"{file}/{info_name}.yaml", "r") as f:
                    val = yaml.safe_load(f) # list
                return val
        except Exception:
            print(f"The file: {info_name} does not exist in {file}.")

    @staticmethod
    def save_pkl(dict_, axis):
        name = f"{axis.upper()}grad_{axis.upper()}eddy"
        
        with open(f"{Path.cwd()}/{name}.pkl", "wb") as f:
            pickle.dump(dict_, f)