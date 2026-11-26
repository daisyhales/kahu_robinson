import os
import copy
from matipo import SEQUENCE_DIR, GLOBALS_DIR
from matipo.sequence import Sequence
import sys
import asyncio
# sys.stdout.reconfigure(encoding="utf-8")
import numpy as np
import panel as pn
import yaml
from pathlib import Path
import logging
import traceback
import optuna
import pandas as pd
import matplotlib.pyplot as plt
from matipo.util.autophase import autophase
from matipo.util.decimation import decimate
from matipo.util.fft import fft_reconstruction
import numpy as np
# import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
from matipo.util.pulseshape import calc_soft_pulse
from matipo.util.fft import fft_reconstruction
from matipo.util.fft import get_freq_spectrum
import scipy as sp

class Optimiser:
    """
    Class for the creations of the panel elements that can be used with any scan.
    """
    def __init__(self):
        self.file = GLOBALS_DIR

        with open(self.file + "fitting_pars.yaml", "r") as f:
            self.MAX_EXP = yaml.safe_load(f)["MAX_EXP"]

        self.list_nums = []
        self.list_A = []
        self.list_tau = []
        for i in range(1, self.MAX_EXP):
            self.list_nums.append(i)
            self.list_A.append(f"A{i}")
            self.list_tau.append(f"tau{i}")
            
        self.best = None

        
    
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
    def get_g_read(axis, g_val):

        grad_dict = {"X": (g_val,0,0), "Y": (0,g_val,0), "Z": (0,0,g_val)}
        return grad_dict[axis]
    
    @staticmethod
    def g_ideal1(g_start, g_end, t_ramp, t_grad, t_rf_wait, t_next, t_acq_delay, t_acq, t_wait, waits_pulses, t_end, dt = 10e-6, pulse_time = None, amp = None, phase = None, acq = "n", p_acq = None, ):
        """
        """
        i_pulse = [0]
        N_rise = int(t_ramp/dt)
        N_flat = int(t_grad/dt)
        N_fall = int(t_ramp/dt)
        N_pad = int((t_end + t_next)/dt)
        
        N_rfwaitb = int((t_grad - t_rf_wait)/dt)
        N_rfwaitd = int(t_rf_wait/dt)
        N_rf = int(pulse_time/dt)
        N_acq_delay = int(t_acq_delay/dt)
        N_acq = int(t_acq/dt)
        N_wait = int(t_wait/dt)
        
        g_rise_array = np.linspace(g_start, g_end, N_rise+1)[:-1] # excluding the first 0 point, we want to be setting a non-zero value immediately
        g_flat_array = np.array([g_end]*N_flat)
        g_fall_array = np.linspace(g_end, g_start, N_fall+1)[1:] # excluding the first point, we want to start falling immediately and end at 0
        g_pad_array = np.linspace(g_start, g_start, N_pad+1)[:-1]
    
        
        g_ideal = np.concatenate((g_rise_array, g_flat_array, g_fall_array, g_pad_array))
    
        # --------------------------------------
        rf_arr1 = np.zeros(len(g_rise_array))
        freq_arr1 = np.zeros(len(g_rise_array))
        phase_arr1 = np.zeros(len(g_rise_array))
        acq_arr1 = np.zeros(len(g_rise_array))
        
        rf_arr2 = np.zeros(N_rfwaitb)
        freq_arr2 = np.zeros(N_rfwaitb)
        phase_arr2 = np.zeros(N_rfwaitb)
        acq_arr2 = np.zeros(N_rfwaitb)
    
        i_pulse.append(i_pulse[-1] + 1)
    
        # pulse rf (hard pulse) - pulse 1
        rf_arr3 = np.linspace(amp, amp, N_rf)
        freq_arr3 = np.ones(N_rf)
        phase_arr3 = np.linspace(phase, phase, N_rf)
        acq_arr3 = np.zeros(N_rf)
        
        # turn rf off, wait t_acq_delay then acquire -- this is the one with the g_read still on
        rf_arr4 = np.zeros(N_acq_delay + N_acq)
        freq_arr4 = np.zeros(N_acq_delay + N_acq)
        phase_arr4 = np.zeros(N_acq_delay + N_acq)
        acq_arr4 = np.concatenate((np.zeros(N_acq_delay), np.ones(N_acq)))
    
        # ramp off the g_read
        rf_arr5 = np.zeros(N_fall+N_wait)
        freq_arr5 = np.zeros(N_fall+N_wait)
        phase_arr5 = np.zeros(N_fall+N_wait)
        acq_arr5 = np.zeros(N_fall+N_wait)
    
        rf_arr6 = []
        freq_arr6 = []
        phase_arr6 = []
        acq_arr6 = []
        
        for j, t_pulse_wait in enumerate(waits_pulses):
            i_pulse.append(i_pulse[-1] + 1)
            # hard pulse
            rf_arrw = np.linspace(amp, amp, N_rf)
            freq_arrw = np.ones(N_rf)
            phase_arrw = np.linspace(phase, phase, N_rf)
            acq_arrw = np.zeros(N_rf)
    
            # wait acq delay then acquire
            rf_arrw1 = np.zeros(N_acq_delay + N_acq)
            freq_arrw1 = np.zeros(N_acq_delay + N_acq)
            phase_arrw1 = np.zeros(N_acq_delay + N_acq)
            acq_arrw1 = np.concatenate((np.zeros(N_acq_delay), np.ones(N_acq)))
    
            N_wait_pulse = int(t_pulse_wait / dt)
            # wait the exponential spacing required before next pulse
            
            rf_arrw2 = np.zeros(N_wait_pulse)
            freq_arrw2 = np.zeros(N_wait_pulse)
            phase_arrw2 = np.zeros(N_wait_pulse)
            acq_arrw2 = np.zeros(N_wait_pulse)
    
            rf_arr6.append(rf_arrw)
            rf_arr6.append(rf_arrw1)
            rf_arr6.append(rf_arrw2)
            freq_arr6.append(freq_arrw)
            freq_arr6.append(freq_arrw1)
            freq_arr6.append(freq_arrw2)
            phase_arr6.append(phase_arrw)
            phase_arr6.append(phase_arrw1)
            phase_arr6.append(phase_arrw2)
            acq_arr6.append(acq_arrw)
            acq_arr6.append(acq_arrw1)
            acq_arr6.append(acq_arrw2)
    
        
        rf_arr6 = np.concatenate(rf_arr6)
        freq_arr6 = np.concatenate(freq_arr6)
        phase_arr6 = np.concatenate(phase_arr6)
        acq_arr6 = np.concatenate(acq_arr6)
        
        # final pulse 
        i_pulse.append(i_pulse[-1] + 1)
    
        rf_arr7 = np.linspace(amp, amp, N_rf)
        freq_arr7 = np.ones(N_rf)
        phase_arr7 = np.linspace(phase, phase, N_rf)
        acq_arr7 = np.zeros(N_rf)
        
        # final acq delay and acquire
        rf_arr8 = np.zeros(N_acq_delay + N_acq)
        freq_arr8 = np.zeros(N_acq_delay + N_acq)
        phase_arr8 = np.zeros(N_acq_delay + N_acq)
        acq_arr8 = np.concatenate((np.zeros(N_acq_delay), np.ones(N_acq)))
    
        
        
        rf_arr = np.concatenate((rf_arr1, rf_arr2, rf_arr3, rf_arr4, rf_arr5, rf_arr6, rf_arr7, rf_arr7))
        freq_arr = np.concatenate((freq_arr1, freq_arr2, freq_arr3, freq_arr4, freq_arr5, freq_arr6, freq_arr7, freq_arr7))
        phase_arr = np.concatenate((phase_arr1, phase_arr2, phase_arr3, phase_arr4, phase_arr5, phase_arr6, phase_arr7, phase_arr7))
        acq_arr = np.concatenate((acq_arr1, acq_arr2, acq_arr3, acq_arr4, acq_arr5, acq_arr6, acq_arr7, acq_arr7))

        # generate time axis
        t = dt*np.arange(len(g_ideal))

        # pad to gradient length
        pad_len = len(g_ideal) - len(rf_arr)
        
        if pad_len > 0:
            rf_arr = np.concatenate((rf_arr, np.zeros(pad_len)))
            freq_arr = np.concatenate((freq_arr, np.zeros(pad_len)))
            phase_arr = np.concatenate((phase_arr, np.zeros(pad_len)))
            acq_arr = np.concatenate((acq_arr, np.zeros(pad_len)))
    
        return g_ideal, t, rf_arr, freq_arr, acq_arr, phase_arr, i_pulse

    @staticmethod
    def get_pulse_timings(t_pulse, t_acq, t_acq_delay, t_wait, t_end, n_pulses):
        """
        """
        #  calculate pulse spacing 
        vals = np.geomspace(t_wait, t_end, num=n_pulses)
        seps = np.diff(vals)
        minus = t_pulse + t_acq + t_acq_delay
        waits_pulses = []
        for i in seps:
            w = float(i-minus)
            if w < 0:
                raise ValueError("exp spacing is less than zero, change t_pulse or t_acq time")
            else:
                waits_pulses.append(w)
        return waits_pulses


    def check_preemph_limit(self, params, g_ideal, g_read, OPT_AXIS, dt):
        """
        pre-emphasis method used to check the g_preemph is not going over the gradient limits so it can be catched and given a 
        cost without breaking the optimisation loop
        """
    
        limits = {
            (0.1, 0, 0): 0.30,
            (0, 0.1, 0): 0.55,
            (0, 0, 0.1): 0.55,
        }
    
        limit = limits[tuple(g_read)]
    
        g_preemph = g_ideal.copy()
        for n, axis in enumerate(["X", "Y", "Z"]):
    
            g_preemph_axis = g_ideal[:, n].copy()
    
            amps = [params[axis][a] for a in self.list_A]
            taus = [params[axis][t] for t in self.list_tau]
    
            for j in range(len(amps)):
            
                coeff = amps[j]
                time_const = taus[j]
    
                if coeff == 0 or time_const == 0:
                    continue
            
                alpha = dt / time_const
                b_inv = [1, -1/(1+alpha)]
                a_inv = [1 - coeff, coeff - 1/(1+alpha)]
                    
                g_preemph_axis = sp.signal.lfilter(b_inv, a_inv, g_preemph_axis)
    
            
                if np.max(np.abs(g_preemph_axis)) > limit:
                    return False, g_ideal
    
            g_preemph[:, n] = g_preemph_axis
    
        return True, g_preemph

    @staticmethod
    def get_calibrations(file = GLOBALS_DIR):
        """
        """
        try:
            with open(f"{file}/B0_calibration.yaml", "r") as f:
                B0_cal = yaml.safe_load(f)
        
            with open(f"{file}/g_hzpm_slope.yaml", "r") as f:
                G_cal = yaml.safe_load(f)
    
        except Exception as e:
            ValueError(f"file in folder {file} doesnt exist")
    
        return B0_cal["m"], G_cal

   
    def check_b0_limit(self, B_params, g_preemph_g, dt = 10e-6):
        """
        pre-emphasis method used to check the g_preemph is not going over the gradient limits so it can be catched and given a 
        cost without breaking the optimisation loop
        """
    
        b0_limit = 1
        
        g_b0_all = []
        
        for i, axis in enumerate(["X", "Y", "Z"]):
            # go through the axes
            amps = [B_params[axis][k] for k in self.list_A]#, "A4", "A5", "A6"]]#, "A7", "A8", "A9", "A10", "A11", "A12"]]
            taus = [B_params[axis][k] for k in self.list_tau]#, "tau4", "tau5", "tau6"]]#, "tau7", "tau8", "tau9", "tau10", "tau11", "tau12"]]
            # apply eddy current and inverse filters to ideal gradient
            # to calculate the resulting gradient waveform and preemphisized waveform
    
            g_preemphB0 = np.zeros(len(g_preemph_g[:, i]))
            for coeff, time_const in zip(amps, taus):
    
                if coeff == 0 or time_const == 0:
                    continue
    
                alpha = dt / time_const
    
                b = [-coeff, coeff]
                a = [1, -1/(1+alpha)]
    
                g_preemphB0 += -sp.signal.lfilter(b, a, g_preemph_g[:, i])
    
            g_b0_all.append(g_preemphB0)
            
        g_b0_all = np.column_stack(g_b0_all) 
    
        B0_cal, G_cal = self.get_calibrations(self.file)
    
        for i, axis in enumerate(["X", "Y", "Z"]):
        
            g_cal = np.where(g_preemph_g[:, i] >= 0, G_cal[axis], G_cal[axis]) # creates an array of calibration values pos or neg depending on sign of g_ideal
        
            g_b0_all[:, i] = g_b0_all[:, i] * (g_cal / B0_cal) # scale the convolution (g_b0_all axis) by the g_cal and B0_cal 
        
        b0_sum = g_b0_all.sum(axis=1) # sums up all the values in each vector
        
        if np.max(np.abs(b0_sum)) > b0_limit:
            return False, b0_sum
    
        return True, b0_sum

    #  create objective function for optuna to minimise
    async def objective_async(self, trial, OPT_AXIS, metric, g_hzpm, E_START, B_START, start_params, percentage, seq_params, g_ideal):
        """
        builds the optimisation sequence
    
        1. check the g_preemph is valid 
        2. save the guess parameters into eddyE_opt for the seq to read
        3. run the seq and return the df_freq information to calculate the cost with
        4. calculate the cost of the run (and other analysis values used in the plots later)
        5. return the cost to be saved into the trials information
        
        """
        cost = 0

        if metric == "E":
            E_params = self.build_yaml(trial, E_START, OPT_AXIS, start_params, percentage)
            B_params = B_START
            
        if metric == "F":
            E_params = E_START
            B_params = self.build_yaml(trial, B_START, OPT_AXIS, start_params, percentage)
            
        g_read = self.get_g_read(OPT_AXIS, seq_params["g_val"])
        pe_check, g_preemph = self.check_preemph_limit(E_params, g_ideal, g_read, OPT_AXIS = OPT_AXIS, dt=10e-6)
        b0_check, b0_sum = self.check_b0_limit(B_params, g_preemph, dt = 10e-6)
        
        try:
            print("E max =", np.max(g_preemph))
            print("E nanmax =", np.nanmax(g_preemph))
            print("E contains nan =", np.any(np.isnan(g_preemph)))
            print("B max =", np.max(b0_sum))
            print("B nanmax =", np.nanmax(b0_sum))
            print("B contains nan =", np.any(np.isnan(b0_sum)))
        except Exception as e:
            print(e)
    
        # catch any issues and assign high cost value to stop the optimiser trying them again
        if np.any(np.isnan(b0_sum)):
            print("FAILED: B0 contains NaN") # sometimes the signal processing can lead to nan in divisions or in unstable inv filter
            g_pe_perc = np.nan
            max_b0var = np.nan
            cost += 1e20
            status = "NaN"
        elif np.any(np.isinf(b0_sum)):
            print("FAILED: B0 contains INF") # some params can blow up the signal processing 
            g_pe_perc = np.nan
            max_b0var = np.nan
            cost += 1e20
            status = "INF"
        elif np.any(np.isnan(g_preemph)):
            print("FAILED: Pre-emph contains NaN") # sometimes the signal processing can lead to nan in divisions or in unstable inv filter
            g_pe_perc = np.nan
            cost += 1e20
            status = "NaN"
        elif np.any(np.isinf(g_preemph)):
            print("FAILED: Pre-emph contains INF") # some params can blow up the signal processing 
            g_pe_perc = np.nan
            cost += 1e20
            status = "INF"
        
        elif not pe_check:
            print("FAILED: Pre-emph too HIGH") # params that make the gradient exceed the grad limits we put in place
            g_pe_perc = np.nan
            max_b0var = np.nan
            cost += 1e20
            status = "LIMIT"
        elif not b0_check:
            print("FAILED: B0 too HIGH") # params that make the gradient exceed the grad limits we put in place
            g_pe_perc = np.nan
            max_b0var = np.nan
            cost += 1e20
            status = "LIMIT"
        else:
            if metric == "E":
                self.save_yaml(E_params, "eddy_Eopt.yaml")
            else:
                self.save_yaml(B_params, "eddy_B0opt.yaml")
                
            print(f"Trial {trial.number}")
            df_freq, g_val = await self.run_measurement(trial, g_hzpm, OPT_AXIS, seq_params)
            g_pe_perc = np.max(np.abs(g_preemph))/g_val * 100
            max_b0var = np.max(np.abs(b0_sum))
            t = df_freq["times (s)"].values[1:]
    
            # calculate the cost of the run -- CONSTANT FREQUENCY METHOD
            if metric == "F":
    
                freq = df_freq["rx_c"].values[1:]
                rx_cost = self.zero_metric(freq, t)
                print(f"rx_c: {rx_cost:.3e}")
            
                cost += rx_cost
    
            elif metric == "E":
                E = df_freq["g_eddy"].values[1:]
    
                print("max |E| =", np.max(np.abs(E)))
                print("mean |E| =", np.mean(np.abs(E)))
                print("RMS E =", np.sqrt(np.mean(E**2)))
                cost += self.zero_metric(E, t)
                
            print(f"Total cost = {cost:.3e}")
            status = "PASS"
        trial.set_user_attr("g_pe_perc", float(g_pe_perc))
        trial.set_user_attr("max_b0var", float(max_b0var))
        trial.set_user_attr("status", status)
        
        return cost
    

    def objective(self, trial, OPT_AXIS, metric, g_hzpm, E_START, B_START, start_params, percentage, seq_params, g_ideal):
        """
        setup the optuna optimisation in a loop
        """
    
        loop = asyncio.get_event_loop()
    
        return loop.run_until_complete(
            self.objective_async(trial, OPT_AXIS, metric, g_hzpm, E_START, B_START, start_params, percentage, seq_params, g_ideal)
    )            

    @staticmethod
    def zero_metric(f, t):
        """
        cost function - gradient/b0 analysis over time - should be going to zero
        """
        baseline = np.mean(f[-3:])
        f = f - baseline
        f = np.abs(f)
        return np.trapezoid(f, x=t)

    @staticmethod        
    def build_yaml(trial, _START, OPT_AXIS, start_params, percentage):
        """
        Limitations of the parameters given for optuna to guess the next parameter from and saves into a dictionary to be used
        for the pre-emph
        """
        # start from existing calibration
        out = copy.deepcopy(_START)
    
        for i in [1, 2, 3, 4]:
            par = start_params[f"{OPT_AXIS}_A{i}"] 
            if par == 0:
                delta = 0.00000001
            else:
                delta = abs(par) * percentage/100 # allow % variation in the amplitudes
            out[OPT_AXIS][f"A{i}"] = trial.suggest_float(f"{OPT_AXIS}_A{i}", par-delta, par+delta)
    
        return out
    
    @staticmethod    
    def save_yaml(data, filename):
        """
        saves the current parameters into the eddy_Eopt yaml so that they can be read into the pulse sequence to generate 
        the gradient 
        """
        with open(filename, "w") as f:
            yaml.safe_dump(data, f, sort_keys=False)



    async def run_measurement(self, trial, g_hzpm, OPT_AXIS, seq_params):
        """
        Loads and runs the pulse sequence then plots the eddy current analysis from that run
        """
        ZERO_FILLING = 200000
        pre_file = self.get_yaml_info(info_name = 'prefiles.yaml', file = GLOBALS_DIR)["optimise_ec"] # where the pulse file lives
        file_name = self.get_yaml_info(info_name = 'seqnames.yaml', file = GLOBALS_DIR)["optimise_ec"] # folder to save data and protocols

        seq = Sequence(pre_file+file_name)
        
        # load relevant global variables
        seq.loadpar(GLOBALS_DIR+'frequency_tx0.yaml')
        seq.loadpar(GLOBALS_DIR+'shims.yaml')
        
        g_val = seq_params["g_val"]
        gg_read = self.get_g_read(OPT_AXIS, g_val)
        
        seq.setpar(
            n_scans = seq_params["n_scans"],
            a_pulse=seq_params["a_pulse"], 
            t_pulse=seq_params["t_pulse"],
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
            pre_emphasis = seq_params["pre_emphasis"],
            b0 = seq_params["b0"],
            file_pars = seq_params["file_pars"],
            file_cals = seq_params["file_cals"],
            opt = "y"
        )
        
        
        pn.extension()
        # print(seq.par)
        y = await seq.run()
        # progress_handler=print_progress
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
        
        if type(trial) is not str:  
            fig.suptitle(f"Frequency, Optimisation Round {trial.number}", fontsize=16)
        else:
            fig.suptitle(f"Frequency, Optimisation Round {trial}", fontsize=16)
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
        if type(trial) is not str:   
            fig.suptitle(f"Spectra, Optimisation Round {trial.number}", fontsize=16)
        else:
            fig.suptitle(f"Spectra, Optimisation Round {trial}", fontsize=16)
        
        plt.tight_layout()
        plt.show()
    
        # distances of coils in meters
        d_r = float(seq_params["d_r"])
        # d_c = int(seq_params["d_c"])
        d_l = float(seq_params["d_l"])    
        
        # calculate the eddy over time 
        df_freq["g_eddy"] = (df_freq["rx_r"] - df_freq["rx_l"]) / (d_r - d_l)
        df_freq["E(t)"] = df_freq["g_eddy"]/ g_hzpm 
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        N = 3
        baseline = df_freq["g_eddy"].tail(N).mean()
        ged = df_freq["g_eddy"] - baseline
        axes[0].axhline(y=0, color="black", linestyle="--", linewidth=1, label="Ideal")
        axes[0].plot(df_freq["times (s)"][1:], ged[1:]*1e-3, label = "Data", color="red")
        axes[0].set_xscale("log")
        axes[0].set_xlabel("Time (s)")
        axes[0].set_ylabel("Field Gradient (kHz/m)")
        axes[0].set_title("Field Gradient Decay")
        axes[0].legend(fontsize=8)
            
        baselineb0 = df_freq["rx_c"].tail(N).mean()
        rxc = df_freq["rx_c"] - baselineb0
        axes[1].plot(df_freq["times (s)"][1:], rxc[1:]*1e-3, label = "Data", color="blue")
        axes[1].axhline(y=0, color="black", linestyle="--", linewidth=1, label="Ideal")            
        axes[1].set_xscale("log")
        axes[1].set_xlabel("Time (s)")
        axes[1].set_ylabel("Field Offset (kHz)")
        axes[1].set_title("Field Offset Decay")
        axes[1].legend(fontsize=8)
        if type(trial) is not str:   
            fig.suptitle(f"{OPT_AXIS} Eddy {OPT_AXIS} Axis, Optimisation Round {trial.number}", fontsize=16)
        else:
            fig.suptitle(f"{OPT_AXIS} Eddy {OPT_AXIS} Axis, Optimisation Round {trial}", fontsize=16)
        plt.show() 
    
        return df_freq, g_val


    def plot_for_optimisation(self, df_trials, OPT_AXIS, metric):
        """
        """
        # PLOT 1 !
        # plot the passes and fails to check the optimiser worked with the cost over trials 
        plt.figure(figsize=(8,5))
        
        passed = df_trials["status"] == "PASS"
        
        plt.scatter(df_trials.loc[passed, "trial"], df_trials.loc[passed, "cost"], color="tab:blue", label="PASS")
        plt.scatter(df_trials.loc[~passed, "trial"], df_trials.loc[~passed, "cost"], color="red", marker="x", s=80, label="FAIL")
        plt.plot(df_trials["trial"], np.minimum.accumulate(df_trials["cost"]), lw=2, color = "orange", label="Best so far")
        
        plt.xlabel("Trial")
        plt.ylabel("Cost")
        plt.yscale("log")
        plt.legend()
        if metric == "E":
            plt.suptitle("Cost Minimisation - Gradient")
        else:
            plt.suptitle("Cost Minimisation - Offset")
        plt.show()

        # PLOT 2 !
        # plot the variation in the amplitudes over trials to see the spread the optimiser has created
        fig, axes = plt.subplots(1, self.MAX_EXP, figsize=(4 * self.MAX_EXP, 4))
        
        for col in range(self.MAX_EXP):
        
            ax = axes[col]
            i = col + 1
        
            name = f"{OPT_AXIS}_A{i}"
        
            ax.scatter(df_trials.loc[passed, "trial"], df_trials.loc[passed, name], color="green", label="PASS")
            ax.scatter(df_trials.loc[~passed, "trial"], df_trials.loc[~passed, name], color="red", marker="x", label="FAIL")
        
            ax.set_title(name)
            ax.set_xlabel("Trial")
            ax.set_ylabel("Amplitude")
        
        axes[-1].legend(loc="upper left", bbox_to_anchor=(1.05, 1))
        if metric == "E":
            plt.suptitle("Amplitude Optimisation - Gradient")
        else:
            plt.suptitle("Amplitude Optimisation - Offset")
            
        plt.tight_layout()
        plt.show()

        # PLOT 3 !
        # plot the gradient/B0 usage variation over the trials
        if metric == "E":
            plt.figure(figsize=(8,5))
            
            plt.scatter(df_trials.loc[passed, "trial"], df_trials.loc[passed, "g_pe_perc"], color="green", label="PASS")
            plt.scatter(df_trials.loc[~passed, "trial"], df_trials.loc[~passed, "g_pe_perc"], color="red", marker="x", s=80, label="FAIL")
            
            plt.xlabel("Trial")
            plt.ylabel("Max pre-emphasised gradient (% of g_read)")
            plt.title(f"{OPT_AXIS} Gradient Usage During Optimisation")
            plt.legend()
            plt.show()

        else:
            plt.figure(figsize=(8,5))
            
            plt.scatter(df_trials.loc[passed, "trial"], df_trials.loc[passed, "max_b0var"], color="green", label="PASS")
            plt.scatter(df_trials.loc[~passed, "trial"], df_trials.loc[~passed, "max_b0var"], color="red", marker="x", s=80, label="FAIL")
            
            plt.xlabel("Trial")
            plt.ylabel("Max B0 Value (DAC units)")
            plt.title(f"B0 Usage During Optimisation for {OPT_AXIS} Optimisation")
            plt.legend()
            plt.show()


    def save_to_global(self, _START, OPT_AXIS, filename):
        best_yaml = copy.deepcopy(_START)

        for i in self.list_nums:
            best_yaml[f"{OPT_AXIS}"][f"A{i}"] = self.best[f"{OPT_AXIS}_A{i}"]
            
        with open(GLOBALS_DIR+filename,"w") as f:
            yaml.safe_dump(best_yaml, f, sort_keys=False)