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
from sklearn.linear_model import LinearRegression

from pathlib import Path
from datetime import date, datetime
import yaml
import re
import pickle
import logging

class Calibrate:
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

    async def run_comparison(self, seq_params):
        """
        
        """

        dict_ = {}
        f_list = []
        
        axis = seq_params["OPT_AXIS"]
        N_scans = seq_params["n_scans"]
        pe = [False, True]
        be = [False]

        for p in pe:
            for b in be:
                for g_val in seq_params['g_list']:
        
        
                    ZERO_FILLING = 200000
                    
                    
                    seq = Sequence(self.pre_file + self.seqname)
                    
                    # load relevant global variables
                    seq.loadpar(GLOBALS_DIR+'frequency_tx0.yaml')
                    seq.loadpar(GLOBALS_DIR+'shims.yaml')
        
                    
                    gg_read = self.get_g_read(axis, g_val)
                    
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
                        # compare = "nn",
                        file_pars = GLOBALS_DIR,
                        file_cals = GLOBALS_DIR,
                        
                        # file_parsE = info["E_pars"],
                        # file_parsB = info["B_pars"],
                        
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
            
                    g_hzpm = (df_freq["rx_r"][0] - df_freq["rx_l"][0]) / (float(seq_params["d_r"]) - float(seq_params["d_l"]))
                    
                    df_freq["g_eddy"] = (df_freq["rx_r"] - df_freq["rx_l"]) / (float(seq_params["d_r"]) - float(seq_params["d_l"]))
                    df_freq["E(t)"] = df_freq["g_eddy"]/ g_hzpm 
                    
                    df_freq["B0(t)"] = df_freq["rx_c"]/ g_hzpm
                
                    f_list.append(df_freq)
            
                    key = f"{name}_PE{e}_BE{b}_G{g_val}"
        
                    dict_[key] = {
                        "freq": df_freq.copy(),
                        "rms": df_rms.copy(),
                        "g_hzpm": g_hzpm,
                    }


        # ------------------------
        # save the info into a pkl file 
        # ------------------------
        
        g_h = []
        g_vals = []
        pe_vals = []
        
        
        for key, value in dict_.items():
        
            match = re.search(r'G(-?\d*\.?\d+)$', key)
            if match is None:
                continue
        
            g = float(match.group(1))
            g_vals.append(g)
            g_h.append(dict_[key]["g_hzpm"])
            pe_vals.append("PETrue" in key)
            print(key, ": ", dict_[key]["g_hzpm"])
        
        df_ghzpm = pd.DataFrame({"G": g_vals, "g_hzpm (Hz/m)": g_h, "PE": pe_vals,})

        
        output_dir = Path(name)
        output_dir.mkdir(parents=True, exist_ok=True)

        
        dict_["df_ghzpm"] = df_ghzpm
        
        with open(output_dir / f"{name}.pkl", "wb") as f:
            pickle.dump(dict_, f)

        with open(output_dir / "seq_params.yaml", "w") as f:
            yaml.safe_dump(seq_params, f, sort_keys=False)
            
        return dict_, f_list, name
    
    @staticmethod
    def analyse_pkl(name):

        with open(f"{name}.pkl", "rb") as f:
            dict_ = pickle.load(f)
    
        g_h = []
        g_vals = []
        pe_vals = []
        be_vals = []
        f_list = []
        labels = []
        g_hzpm_list = []
        
        for key, value in dict_.items():
        
            match = re.search(r'G(-?\d*\.?\d+)$', key)
            if match is None:
                continue
                
            pe = "PETrue" in key
            be = "BETrue" in key
            g = float(match.group(1))
            
            g_vals.append(g)
            g_h.append(dict_[key]["g_hzpm"])
            pe_vals.append(pe)
            be_vals.append(be)
            f_list.append(value["freq"])
            labels.append(f"PE={pe}, G_val={g}")
            # print(key, ": ", dict_[key]["g_hzpm"])
        
        df_ghzpm = pd.DataFrame({"G": g_vals, "g_hzpm (Hz/m)": g_h, "PE": pe_vals, "BE": be_vals})
        # display(df_ghzpm)
        g_hzpm_list.append(df_ghzpm)
    
        # colours = [ "blue", "cornflowerblue", "orange", "gold", "green", "limegreen", "red", "salmon"]
        colours = [
            "blue", 
            "orange", 
            "green", 
            "red", "cornflowerblue","gold", "limegreen","salmon",
        
        ]
        # colours = [ "blue", "red"]
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig2 = plt.figure()
        
        for df, label in zip(f_list, labels):
            if "PE=True" in label:
                axes[0].plot(df["times (s)"][1:], df["E(t)"][1:], label=label, linestyle = "--")
        
            else:
                axes[0].plot(df["times (s)"][1:], df["E(t)"][1:], label=label)
        
        axes[0].set_xscale("log")
        axes[0].set_xlabel("Time (s)")
        axes[0].set_ylabel("E(t)")
        axes[0].set_title("Eddy Field Decay E(t)")
        # axes[0].legend(fontsize=8)
        
        for df, label in zip(f_list, labels):
            if "PE=True" in label:
                axes[1].plot(df["times (s)"][1:], df["B0(t)"][1:], label=label, linestyle = "--")
        
            else:
                axes[1].plot(df["times (s)"][1:], df["B0(t)"][1:], label=label)
        
        
        axes[1].set_xscale("log")
        axes[1].set_xlabel("Time (s)")
        axes[1].set_ylabel("B0(t)")
        axes[1].set_title("B0 Drift")
        axes[1].legend(loc = (1.05, 0.05),fontsize=8)
        
        plt.suptitle(name)
        plt.tight_layout()
        plt.show() 
    
        
        mask = np.array(pe_vals) # mask for PE on or off
        
        plt.figure(figsize=(6,4))
        plt.scatter(np.array(g_vals)[mask], np.array(g_h)[mask], color="blue", label="PE True")
        plt.scatter(np.array(g_vals)[~mask], np.array(g_h)[~mask], color="red", label="PE False")
        
        plt.xlabel("G (DAC units)")
        plt.ylabel("g_hzpm (Hz/m)")
        plt.title(name)
        plt.legend()
        plt.show()
        
        return g_hzpm_list, mask, g_vals, g_h

    @staticmethod
    def fit_ghzpm_slope(name, g_hzpm_list, mask, g_vals, g_h, g_fit_min, g_fit_max):
        
        df_ghzpm = pd.concat(g_hzpm_list, ignore_index=True)
        
        df_fit = df_ghzpm[(df_ghzpm["G"] > g_fit_min) & (df_ghzpm["G"] < g_fit_max)]
        X = df_fit[["G"]]
        Y = df_fit["g_hzpm (Hz/m)"]
        
        # Force intercept to 0 by setting fit_intercept=False
        model = LinearRegression(fit_intercept=False)
        model.fit(X, Y)
        
        # Get the slope
        slope = model.coef_[0]
        
        print(f"Slope: {slope}")           # e.g., ~2.0
        print(f"Intercept: {model.intercept_}") # Strictly 0.0
        
        x_fit = np.linspace(df_ghzpm[["G"]].min(), df_ghzpm[["G"]].max(), 100)
        
        plt.figure(figsize=(6,4))
        plt.scatter(np.array(g_vals)[mask], np.array(g_h)[mask], color="blue", label="PE True")
        plt.scatter(np.array(g_vals)[~mask], np.array(g_h)[~mask], color="red", label="PE False")
        plt.plot(x_fit, slope*x_fit + b, 'k--', label=f"Linear Fit")
        
        plt.xlabel("G (DAC units)")
        plt.ylabel("g_hzpm (Hz/m)")
        plt.title(name)
        plt.legend()
        plt.show()
        return slope, model.intercept_