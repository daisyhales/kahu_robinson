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
from .panes import panes
from .general import general_functions

from scipy.integrate import cumulative_trapezoid, trapezoid
from bokeh.models import LinearColorMapper, ColorBar
from bokeh.palettes import Viridis256
from bokeh.layouts import gridplot, column, row
from bokeh.palettes import Category10
from scipy.optimize import curve_fit

from scipy.optimize import nnls, lsq_linear
from scipy.signal import find_peaks, peak_widths

from bokeh.palettes import Bokeh

class oneDfunctions:
    """
    1D plotting functions such as for spin echo and FIDs/shimming.

    uses @staticmethod uses the attributes defined in an __init__ function at the start of the class
    - as these just take inputs and dont call for variables set in the __init__ function, we 
    can use staticmethod instead which means that we dont have to pass self into each function defined in the 
    class

    Can access the functions through:
    
    from UIfunctions import oneDfunctions
    p = oneDfunctions.signal_plot1D(time, data, n_samples, title = None)
    """
    def __init__(self, general_functions, panes):
        self.g = general_functions
        self.panes = panes
        
        

        

    @staticmethod 
    def find_freq_centre(img, freq):
        """
        Takes in the frequency data and finds the width of the peak for the gradient_calibration.
        """
        mag_img = np.abs(img) *1e6

        total_area = trapezoid(mag_img, freq)

        half_area = total_area/2
        
        cumulative_area = cumulative_trapezoid(mag_img, freq, initial=0)
        idx = np.searchsorted(cumulative_area, half_area)

        half_freq = freq[idx]*1e-3
        return half_freq, idx
   

    
    
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

    @staticmethod
    def signal_soft_plot(time, data, n_samples, a_rf, title = None):
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
                title=f"Kdata RF = {round(a_rf, 2)} %",
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

        p.line(time*1e3, np.abs(data[0:n_samples])*1e3,color="blue", legend_label = "Echo")
        p.line(time*1e3, np.abs(data[n_samples:])*1e3,color="orange", legend_label = "Simulated Echo")
    
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


    @staticmethod
    def eddy_current_plot(df_freq, df_rms, height = 500):
        """
        Display eddy current frequency and RMS results.
    
        Inputs:
            df_freq (pd.DataFrame):
                Frequency data containing columns:
                ['times (s)', 'rx_l', 'rx_c', 'rx_r']
    
            df_rms (pd.DataFrame):
                RMS data containing columns:
                ['times (s)', 'rx_l', 'rx_c', 'rx_r']
    
            angle (float):
                Angle used for the acquisition.
    
        Returns:
            pn.Column:
                Panel layout containing six interactive Bokeh plots.
        """
    
        cols = ["rx_l", "rx_c", "rx_r"]
    
        colors = {
            "rx_l": "blue",
            "rx_c": "orange",
            "rx_r": "green",
        }
    
        freq_plots = []
        rms_plots = []
        height = int(height / 2)

        # self.g.log_msg(f"df__freq: {display(df_freq)}")
    
        for col in cols:
    
            hover = HoverTool(tooltips=[("Time (s)", "@x{0.000}"), ("Value", "@y{0.000}")])
    
            # Frequency plot
            p_freq = figure(title=f"{col} Frequency", height = height, sizing_mode="stretch_width", x_axis_type="log", tools="pan,wheel_zoom,box_zoom,reset,save")
    
            p_freq.add_tools(hover)
            p_freq.scatter(df_freq["times (s)"][1:], df_freq[col][1:], color=colors[col], size=8,)
            p_freq.xaxis.axis_label = "Time (s)"
            p_freq.yaxis.axis_label = "Frequency (Hz)"
            freq_plots.append(p_freq)
    
            # RMS plot
            p_rms = figure(title=f"{col} RMS", sizing_mode="stretch_both", x_axis_type="log", tools="pan,wheel_zoom,box_zoom,reset,save" )
            p_rms.add_tools(HoverTool(tooltips=[("Time (s)", "@x{0.000}"), ("RMS", "@y{0.000}")]))
            p_rms.scatter(df_rms["times (s)"][1:], df_rms[col][1:], color=colors[col], size=8,)
    
            p_rms.xaxis.axis_label = "Time (s)"
            p_rms.yaxis.axis_label = "RMS"
    
            rms_plots.append(p_rms)
    
            p_freq.legend.click_policy = "hide"
            p_rms.legend.click_policy = "hide"
            
        layout = pn.Column(pn.Row(*freq_plots, sizing_mode="stretch_width"),pn.Row(*rms_plots, sizing_mode="stretch_width"),sizing_mode="stretch_width",)

        return layout






    @staticmethod
    def eddy_spectra_plot(data_map, scan_parameters, df_idx, height = 500):
        """
        Display signal and spectrum heatmaps for rx_l, rx_c and rx_r.
    
        Inputs:
            seq : sequence object
                Contains data_map and acquisition parameters.
    
            df_idx : dict or DataFrame-like
                Contains the acquisition indices for each receiver:
                ['rx_l', 'rx_c', 'rx_r']
    
        Returns:
            pn.Column
                Panel layout containing six heatmaps.
        """
    
        cols = ["rx_l", "rx_c", "rx_r"]
    
        signal_plots = []
        spectrum_plots = []
        height = int(height / 2)
        for col in cols:
    
            df = df_idx[col]
    
            signals = []
            spectra = []
    
            for key in df:
    
                signal = data_map[key]
    
                freq, spectrum = get_freq_spectrum(signal=signal,dwell_time=scan_parameters["t_dw"])
    
                signals.append(np.abs(signal))
                spectra.append(np.abs(spectrum))
    
            signals = np.asarray(signals)
            spectra = np.asarray(spectra)

            # ---------------- Signal Heatmap ----------------
    
            signal_mapper = LinearColorMapper(palette=Viridis256, low=np.min(signals), high=np.max(signals),)
    
            p_signal = figure(title=f"{col} Signals", height = height, sizing_mode="stretch_width", tools="pan,wheel_zoom,box_zoom,reset,save", )
            p_signal.legend.click_policy = "hide"
            p_signal.image(image=[signals], x=0, y=0, dw=signals.shape[1], dh=signals.shape[0], color_mapper=signal_mapper,)
    
            p_signal.xaxis.axis_label = "Sample"
            p_signal.yaxis.axis_label = "Scan"
    
            p_signal.add_layout(ColorBar(color_mapper=signal_mapper), "right",)
    
            signal_plots.append(p_signal)
    
            # ---------------- Spectrum Heatmap ----------------
    
            spectrum_mapper = LinearColorMapper(palette=Viridis256, low=np.min(spectra), high=np.max(spectra))
    
            p_spectrum = figure(title=f"{col} Spectra", height = height, sizing_mode="stretch_width", tools="pan,wheel_zoom,box_zoom,reset,save")
            p_spectrum.legend.click_policy = "hide"
            p_spectrum.image(image=[spectra], x=0, y=0, dw=spectra.shape[1], dh=spectra.shape[0], color_mapper=spectrum_mapper)
    
            p_spectrum.xaxis.axis_label = "Sample"
            p_spectrum.yaxis.axis_label = "Scan"
    
            p_spectrum.add_layout(ColorBar(color_mapper=spectrum_mapper), "right",)
    
            spectrum_plots.append(p_spectrum)
    
        # grid = gridplot([signal_plots, spectrum_plots], sizing_mode="stretch_width")
    
        # return grid
        layout = pn.Column(pn.Row(*signal_plots, sizing_mode="stretch_width"), pn.Row(*spectrum_plots, sizing_mode="stretch_width"), sizing_mode="stretch_width",)
        
        return layout


  
    def eddy_decay_plot(self, df_freq, height = 500, x_e_factor = 0, y_e_factor = 0, z_e_factor = 0, x_b0_factor = 0, y_b0_factor = 0, z_b0_factor = 0, compiler = 'n'):
        """
        
        """

        N = 3
        baseline = df_freq["g_eddy"].tail(N).mean()
        baselineb0 = df_freq["rx_c"].tail(N).mean()


        df_freq["g_eddy"] = df_freq["g_eddy"] - baseline
        df_freq["rx_c"] = df_freq["rx_c"] - baselineb0

        
        self.e_data = df_freq["g_eddy"][1:]
        self.b0_data = df_freq["rx_c"][1:]

        self.g.log_msg(f"Inside eddy_decay_plot: {df_freq}")


        if compiler == "y": 
            self.g.log_msg("compiler on")
            if self.g.pe == False: # plot the no PE plot and remove the previous PE plot if it exists - only one can be there
                self.g.log_msg("PE not on for plots")
                self.panes.data_E_noPE.data = {
                    "x": df_freq["times (s)"][1:].to_numpy(),
                    "y": df_freq["g_eddy"][1:].to_numpy()*1e-3
                }
                
                
                self.panes.data_B0_noPE.data = {
                    "x": df_freq["times (s)"][1:].to_numpy(),
                    "y": df_freq["rx_c"][1:].to_numpy()*1e-3
                }

            else: # plot the PE plot and check that no more than 8 PE plots are loaded otherwise reset the plot
                # make the line follow the Bokeh colour palette

                # self.g.pe_counter
                self.panes.dict_E_withPE[self.g.pe_counter].data = {
                    "x": df_freq["times (s)"][1:].to_numpy(),
                    "y": df_freq["g_eddy"][1:].to_numpy()*1e-3
                }
                
                
                self.panes.dict_B0_withPE[self.g.pe_counter].data = {
                    "x": df_freq["times (s)"][1:].to_numpy(),
                    "y": df_freq["rx_c"][1:].to_numpy()*1e-3
                }

                self.panes.legend_items_E[self.g.pe_counter].label = {"value": f"X: {round(x_e_factor)}%, Y: {round(y_e_factor)}%, Z: {round(z_e_factor)}%"}

                self.panes.legend_items_B0[self.g.pe_counter].label = {"value": f"X: {round(x_b0_factor)}%, Y: {round(y_b0_factor)}%, Z: {round(z_b0_factor)}%"}

        else: 
        
            self.panes.data_E_source.data = {
                "x": df_freq["times (s)"][1:].to_numpy(),
                "y": df_freq["g_eddy"][1:].to_numpy()*1e-3
            }
            
            self.panes.data_B0_source.data = {
                "x": df_freq["times (s)"][1:].to_numpy(),
                "y": df_freq["rx_c"][1:].to_numpy()*1e-3
            }


    @staticmethod
    def img_soft_plot(freq, img, freq2, img2, n_samples, a_rf, freq_peak, peak, freq2_peak, peak2, title = None):
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
                title=f"RF = {round(a_rf, 2)} %",
                sizing_mode="stretch_width",
                tools="pan,wheel_zoom,box_zoom,reset,save"
            )
        else:
            p = figure(
                title=f"{title} - {n_samples} Samples",
                sizing_mode="stretch_width",
                tools="pan,wheel_zoom,box_zoom,reset,save"
            )

        p.add_tools(hover)


        p.line(freq*1e-3, np.abs(img)*1e6, color='red',  legend_label="Echo")
        p.line(freq2*1e-3, np.abs(img2)*1e6, color='green',  legend_label="Simulated Echo")
        p.scatter(freq_peak*1e-3, peak*1e6,  color='blue', marker = "x", size = 12, legend_label="Echo Peak")
        p.scatter(freq2_peak*1e-3, peak2*1e6,  color='orange', marker = "x", size = 12, legend_label="Simulated Echo Peak")
        
        p.xaxis.axis_label = "Frequency (kHz)"
        p.yaxis.axis_label = ""
        p.legend.click_policy = "hide"

        p.title.text_font_size = "18pt"
        
        p.xaxis.axis_label_text_font_size = "16pt"
        p.yaxis.axis_label_text_font_size = "16pt"
        
        p.xaxis.major_label_text_font_size = "14pt"
        p.yaxis.major_label_text_font_size = "14pt"
        
        p.legend.label_text_font_size = "14pt"

        return p

    @staticmethod
    def n_exp_log(t, *params):
        """
        params = (A0, A1, ..., An, logtau1, ..., logtaun)
        A0 is the offset; taus are parameterized as ln(tau).
        """
        n = (len(params) - 1) // 2
        A0 = params[0]
        amps = params[1:1 + n]
        logtaus = params[1 + n:]
        y = np.full_like(t, A0, dtype=float)
        for A, logtau in zip(amps, logtaus):
            y += A * np.exp(-t / np.exp(logtau))
        return y
        

    def fit(self, N_tau, lam, tau_min, tau_max, fitting_dataset, time_dataset, N_exp_max, type_ = "E", MAX_EXP = 4, g_hzpm = 0):
        """
        """

        self.g.log_msg(f"N_tau: {N_tau}, lam: {lam}, tau_min: {tau_min}, tau_max: {tau_max}, N_exp_max: {N_exp_max}, type_: {type_}, MAX_EXP: {MAX_EXP}")
        
        fit_params = {
            "N_tau": N_tau,
            "lam": lam,
            "tau_min": tau_min,
            "tau_max": tau_max,
            "N_exp_max": N_exp_max,
            "type": type_,
            "MAX_EXP": MAX_EXP,
        }

        time_dataset = np.asarray(time_dataset)
        
        taus = np.geomspace(tau_min, tau_max, N_tau)
        M = np.exp(-time_dataset[:, None]/taus) # matrix of exponential functions, indexed: (tau, t)
        reg_mat = np.sqrt(lam) * np.eye(N_tau)
        M = np.vstack([M, reg_mat])

        b = np.concatenate([fitting_dataset, np.zeros(N_tau)])
        if type_ == "B0":
            res = lsq_linear(M, b)
            x = res.x
            x_abs = np.abs(x)
            pk, _ = find_peaks(x_abs,prominence=0.01 * x_abs.max()) # use the zero crossing to find the left and right bases
            
            # find where x crosses zero
            zero_crossings = []
            
            for i in range(len(x) - 1):
                if np.sign(x[i]) != np.sign(x[i + 1]):
                    zero_crossings.append(i)
            
            zero_crossings = np.array(zero_crossings)

            left_bases = []
            right_bases = []
            
            for p in pk:
            
                left = zero_crossings[zero_crossings < p] # just get the zc on the left of peak
                right = zero_crossings[zero_crossings > p] # just get the zc on the right of peak
            
                # nearest zero crossing to the left of the peak
                if len(left) > 0:
                    lb = left[-1] + 1 # grabs the last item in left which is the val / crossing closest to the peak
                    # +1 as this is where the crossing is switching in between so want to pull the val inside the region
                else:
                    lb = 0
                
                # nearest zero crossing to the right of the peak
                if len(right) > 0:
                    rb = right[0] # already in the region so no +1
                else:
                    rb = len(x) - 1
            
                left_bases.append(lb)
                right_bases.append(rb)

            props = {
                "left_bases": np.array(left_bases),
                "right_bases": np.array(right_bases)
            }
                        
        else:
            x, rnorm = nnls(M, b)  

            # integrate peaks to get the correct A value
            pk, props = find_peaks(x, prominence=0.01*x.max())
        self.g.log_msg(f"{type_}: x.max={x.max()}")
        self.g.log_msg(f"{type_}: n_peaks={len(pk)}")
        pk_A = []
        pk_tau = []
        for lb, rb, tau in zip(props['left_bases'], props['right_bases'], taus[pk]):
            A = np.sum(x[lb:rb])
            pk_A.append(A)
        pk_A = np.array(pk_A)
        pk_tau = np.array(taus[pk])

        # Sort As/taus by decreasing A magnitude
        if type_ == "B0":
            idx = np.argsort(np.abs(pk_A))[::-1]
        else:
            idx = np.argsort(pk_A)[::-1]
        pk_A = pk_A[idx]
        pk_tau = pk_tau[idx]
        
        fit_A = pk_A[:N_exp_max]
        fit_tau = pk_tau[:N_exp_max]
        fit_n = len(fit_A)
        
        if fit_n == 1:
            fit_A = [fit_A[0], fit_A[0]]
            fit_tau = [fit_tau[0], fit_tau[0]]
            fit_n = 2

        lower_bounds = ([-np.inf] + [-1]*fit_n + [np.log(tau_min)]*fit_n)
        upper_bounds = ([np.inf] + [1]*fit_n + [np.log(tau_max)]*fit_n)
        
        guess = [0] + list(fit_A) + list(np.log(fit_tau))
        # self.g.log_msg(f"fit_A guess:, {fit_A}")
        # self.g.log_msg(f"fit_tau guess:, {fit_tau}")
        popt, pcov = curve_fit(self.n_exp_log, time_dataset, fitting_dataset, p0=guess, bounds=(lower_bounds, upper_bounds), method='trf', x_scale='jac', maxfev=10000)
        params = popt

        fit_A = params[1:1+fit_n]
        fit_logtau = params[1+fit_n:1+2*fit_n]
        fit_tau = np.exp(fit_logtau)
        
        rows = ["A0"]
        values = [params[0]]
        
        for i in range(MAX_EXP):
        
            if i < len(fit_A):
                A = fit_A[i]
                tau = fit_tau[i]
            else:
                A = 0.0
                tau = 0.0
        
            rows.extend([f"A{i+1}", f"tau{i+1}"])
            values.extend([A, tau])
        
        df_fit = pd.DataFrame({
            "Parameter": rows,
            "df_fit": values
        })
        fitted_values = self.n_exp_log(time_dataset, 0, *params[1:]) * g_hzpm
        self.g.log_msg(f"df_fit{type_}: \n{df_fit.to_string(index=False)}")

        return df_fit, fitted_values, taus, x, props, pk, fit_params 



    def fit_onto_E(self, fitted_dataset, time_dataset, taus, x, props, pk, N_exp_max = 3):
        #  make tau plot and attach to self.pane


        self.panes.tau_E_source.data = {
            "x": taus,
            "y": x
        }
        
        fig_tau = self.panes.tau_E.object
        fig_tau.renderers = [r for r in fig_tau.renderers if getattr(r, "name", None) not in ["peak_area", "peak_marker"]]

        # n_peaks = min(len(pk), N_exp_max)
        for legend in fig_tau.legend:
            legend.items.clear()

        n_peaks = min(len(pk), N_exp_max)
        
        colors = Bokeh[8][1:1+N_exp_max]
        
        for i, (color, lb, rb, p) in enumerate(zip(colors, props["left_bases"][:n_peaks], props["right_bases"][:n_peaks], pk[:n_peaks]), start=1):
            fig_tau.varea(x=taus[lb:rb+1], y1=0, y2=x[lb:rb+1], fill_color=color, fill_alpha=0.35, legend_label=f"A{i}", name="peak_area")
            fig_tau.scatter([taus[p]], [x[p]], marker="x", size=15, color=color, line_width=3, legend_label=f"τ{i}", name="peak_marker")
        
        self.panes.res_E_source.data = {
            "x": time_dataset,
            "y": (self.e_data - fitted_dataset)
        }       
        self.panes.ideal_source.data = {
            "x": time_dataset,
            "y": np.linspace(0, 0, len(time_dataset))     
        }
        self.panes.fit_E_source.data = {"x": time_dataset, "y": fitted_dataset * 1e-3}

    def fit_onto_B0(self, fitted_dataset, time_dataset, taus, x, props, pk, N_exp_max = 4):
        #  make tau plot and attach to self.pane

        self.panes.tau_B0_source.data = {
            "x": taus,
            "y": x
        }
        fig_tau = self.panes.tau_B0.object
        fig_tau.renderers = [r for r in fig_tau.renderers if getattr(r, "name", None) not in ["peak_area", "peak_marker"]]
        for legend in fig_tau.legend:
            legend.items.clear()
            
        colors = Bokeh[8][::-1][:N_exp_max]
        
        for i, (color, lb, rb, p) in enumerate(zip(colors, props["left_bases"][:N_exp_max], props["right_bases"][:N_exp_max], pk[:N_exp_max]), start=1):
            fig_tau.varea(x=taus[lb:rb+1], y1=0, y2=x[lb:rb+1], fill_color=color, fill_alpha=0.35, legend_label=f"A{i}", name="peak_area")
            fig_tau.scatter([taus[p]], [x[p]], marker="x", size=15, color=color, line_width=3, legend_label=f"τ{i}", name="peak_marker")

            
        # self.panes.tau_B0.object = p_tauB0
        
        self.panes.res_B0_source.data = {
            "x": time_dataset,
            "y": (self.b0_data - fitted_dataset)
        }       
        self.panes.ideal_source.data = {
            "x": time_dataset,
            "y": np.linspace(0, 0, len(time_dataset))
        }

        self.panes.fit_B0_source.data = {"x": time_dataset, "y": fitted_dataset * 1e-3}        


    
    def grad_cal_img_plot(self, freq, img, n_samples, select_xyz, grad_value, title = None, threshold = 0.01, file = None):
        """
        widths are given in mm
        """
        hover = HoverTool(tooltips=[("x", "$x{0.000}"), ("y", "$y{0.000}")])
        if title == None:
            p = figure(
                title=f"Real Space - {n_samples} Samples",
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
        
        p.line(freq*1e-3,img.real*1e6, color='blue',  legend_label="real")
        p.line(freq*1e-3,img.imag*1e6,  color='red',   legend_label="imag")
        p.line(freq*1e-3,np.abs(img)*1e6,  color='green', legend_label="mag")
        
        p.xaxis.axis_label = "Frequency (kHz)"
        p.yaxis.axis_label = ""
        p.legend.click_policy = "hide"

        
        print("freq type:", type(freq))
        print("img type:", type(img))
        print("freq shape:", np.shape(freq))
        print("img shape:", np.shape(img))
        print("freq first/last:", freq[0], freq[-1])

        try: 
            half_freq, idx_half = oneDfunctions.find_freq_centre(img, freq)
        except Exception:
            tb = traceback.format_exc()
            self.g.log_msg("Error in Signal Plot:")
            self.g.log_msg(tb)
            return
            
        mag_img = np.abs(img)*1e6
        peak_y_axis = np.max(mag_img)
        try:
            
            thresh_img = threshold * mag_img[idx_half]
            left_idxs = np.where(mag_img[:idx_half] < thresh_img)[0] # gets the idx of closest value to the peak where the threshold is crossed
            right_idxs = np.where(mag_img[idx_half:] < thresh_img)[0]

            if len(left_idxs) == 0 or len(right_idxs) == 0:
                self.g.log_msg(f"No peak found in img plot for threshold at {threshold * 100}% \n CHECK YOU TURNED ON RF AND GRAD AMPS (NO PEAK TO ANALYSE IN IMG PLOT)")
                return p, None, None
            else:
                left = left_idxs[-1]
                right = idx_half + right_idxs[0]
            
            peak_width = np.abs(freq[right] - freq[left])
            
            p.line([freq[left]*1e-3, freq[right]*1e-3], [thresh_img, thresh_img], color = "black", legend_label = "peak_width")
    
            # get select_xyz -> use to get the calibration m value
            sample_size_dict = self.g.get_yaml_info(info_name = "sample_calibration", file = file)
            length = sample_size_dict[select_xyz]

            p.title.text_font_size = "18pt"
            
            p.xaxis.axis_label_text_font_size = "16pt"
            p.yaxis.axis_label_text_font_size = "16pt"
            
            p.xaxis.major_label_text_font_size = "14pt"
            p.yaxis.major_label_text_font_size = "14pt"
            
            p.legend.label_text_font_size = "14pt"
    
            return p, peak_width
        except Exception:
            self.g.log_msg("Error in Image Plot: - CHECK YOU TURNED ON RF AND GRAD AMPS (NO PEAK TO ANALYSE IN IMG PLOT)")
            self.g.log_msg(tb)
            return None, None, None
    

    def img_plot1D(self, freq, img, n_samples, select_xyz = None, grad_value = None, title = None, scan_type = None, threshold = 0.03, file = None, freq_mm = None, g = None, calc_width = "n"):
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
        if title == None:
            p = figure(
                title=f"Real Space - {n_samples} Samples",
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

        if scan_type =="FID":
            self.g.log_msg("IN FID")

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
            
            return p, None 

        else:

            if file == None:
                file = GLOBALS_DIR
        
            if grad_value < 0:
                with open(f"{file}/neg_gstrength.yaml", "r") as f:
                    g_dict = yaml.safe_load(f)
            else: 
                g_str = self.g.get_g_strength(None, select_xyz)

            geff = g_str * grad_value
            position_m = freq/geff
            
            if freq_mm == 0:     
                p.line(position_m, img.real*1e6, color='blue',  legend_label="real")
                p.line(position_m, img.imag*1e6,  color='red',   legend_label="imag")
                p.line(position_m, np.abs(img)*1e6,  color='green', legend_label="mag")
                
                p.xaxis.axis_label = "Position (m)"
            elif freq_mm == 1:
    
                p.line(freq*1e-3,img.real*1e6, color='blue',  legend_label="real")
                p.line(freq*1e-3,img.imag*1e6,  color='red',   legend_label="imag")
                p.line(freq*1e-3,np.abs(img)*1e6,  color='green', legend_label="mag")
                
                p.xaxis.axis_label = "Frequency (kHz)"
      
            else:
                p.line(freq*1e-3,img.real*1e6, color='blue',  legend_label="real")
                p.line(freq*1e-3,img.imag*1e6,  color='red',   legend_label="imag")
                p.line(freq*1e-3,np.abs(img)*1e6,  color='green', legend_label="mag")
                
                p.xaxis.axis_label = "Frequency (kHz)"
            
                p.yaxis.axis_label = ""
                p.legend.click_policy = "hide"
        
            if calc_width == "y":
                if threshold == None:
                    self.g.log_msg("Calibrate with grad_cal.ipynb before you get the width.")
                    return p, None 
                else:
                    self.g.log_msg(f"Threshold = {threshold*100}%.")
                try: 
                    half_freq, idx_half = oneDfunctions.find_freq_centre(img, freq)
                except Exception:
                    tb = traceback.format_exc()
                    self.g.log_msg("Error in Signal Plot:")
                    self.g.log_msg(tb)
                    return p, None 
                    
                mag_img = np.abs(img)*1e6

                thresh_img = threshold * mag_img[idx_half]
                left_idxs = np.where(mag_img[:idx_half] < thresh_img)[0] # gets the idx of closest value to the peak where the threshold is crossed
                right_idxs = np.where(mag_img[idx_half:] < thresh_img)[0]
    
                if len(left_idxs) == 0 or len(right_idxs) == 0:
                    self.g.log_msg(f"No peak found in img plot for threshold at {threshold * 100}% \n CHECK YOU TURNED ON RF AND GRAD AMPS (NO PEAK TO ANALYSE IN IMG PLOT)")
                    self.g.status_msg("ERROR PEAK WIDTH")
                    return p, None
                else:
                    left = left_idxs[-1]
                    right = idx_half + right_idxs[0]
                    
                if freq_mm == 1:
                    p.line([freq[left]*1e-3, freq[right]*1e-3], [thresh_img, thresh_img], color = "black", legend_label = "peak_width")
                freq_width = np.abs(freq[right] - freq[left])

                length_m = np.abs(position_m[right] - position_m[left])
                length_mm = length_m * 1e3
                self.g.log_msg(f"Peak Width = {freq_width*1e-3:.3f} kHz")
                self.g.log_msg(f"Object Width = {length_mm:.3f} mm")

            else: 
                length_m = None
            from bokeh.models import SaveTool
            self.g.log_msg(f"{p.select(type=SaveTool)}") 

            p.title.text_font_size = "18pt"
            
            p.xaxis.axis_label_text_font_size = "16pt"
            p.yaxis.axis_label_text_font_size = "16pt"
            
            p.xaxis.major_label_text_font_size = "14pt"
            p.yaxis.major_label_text_font_size = "14pt"
            
            p.legend.label_text_font_size = "14pt"
            
            return p, length_m



   

    def B0_plot(self, kdata_array, t_dw,  apodisation, b0_list):
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
        
        freq_list = []
        ZERO_FILLING = 1000
        
        p_tabs = {}
        freq_list = []
        hover = HoverTool(tooltips=[("x", "$x{0.000}"), ("y", "$y{0.000}")])
        for i, data in enumerate(kdata_array):
        
            time = np.linspace(t_dw, len(data)*t_dw, len(data))
            window = np.exp(-np.pi * apodisation * time)
        
            processed_signal = np.concatenate((data * window, np.zeros(ZERO_FILLING)))
        
            freq, img = get_freq_spectrum(signal=processed_signal, dwell_time=t_dw)
        
            spectrum_mag = np.abs(img)
            peak_index = np.argmax(spectrum_mag)
            centre_freq = freq[peak_index]
        
            freq_list.append(centre_freq)
        
            # -------------------------
            # time
            # -------------------------

            p_time = figure(title=f"B0 = {b0_list[i]:.2f}", sizing_mode="stretch_width", tools="pan,wheel_zoom,box_zoom,reset,save")
        
            p_time.line(time*1e3, data.real.flatten()*1e3, color="blue", legend_label="real")
            p_time.line(time*1e3, data.imag.flatten()*1e3, color="red", legend_label="imag")
            p_time.line(time*1e3, np.abs(data.flatten())*1e3, color="green", legend_label="mag")
        
            p_time.xaxis.axis_label = "Time (ms)"
            p_time.yaxis.axis_label = "Signal (mV)"
        
            # -------------------------
            # freq
            # -------------------------
        
            p_freq = figure(title=f"B0 = {b0_list[i]:.2f}", sizing_mode="stretch_width", tools="pan,wheel_zoom,box_zoom,reset,save")
        
            p_freq.line(freq*1e-3, img.real*1e6, color="blue", legend_label="real")
            p_freq.line(freq*1e-3, img.imag*1e6, color="red", legend_label="imag")
            p_freq.line(freq*1e-3, np.abs(img)*1e6, color="green", legend_label="mag")
        
            p_freq.add_layout(
                Span(
                    location=centre_freq*1e-3,
                    dimension="height",
                    line_color="orange",
                    line_width=2,
                )
            )
        
            p_freq.xaxis.axis_label = "Frequency (kHz)"
            for p in [p_freq, p_time]:
                p.title.text_font_size = "18pt"
                
                p.xaxis.axis_label_text_font_size = "16pt"
                p.yaxis.axis_label_text_font_size = "16pt"
                
                p.xaxis.major_label_text_font_size = "14pt"
                p.yaxis.major_label_text_font_size = "14pt"
                
                p.legend.label_text_font_size = "14pt"
                p.add_tools(hover)
        
            freq_pane = pn.pane.Bokeh()
            freq_pane.object = p_freq
            
            time_pane = pn.pane.Bokeh()
            time_pane.object = p_time


            # row_for_tab = pn.Row(p_time, p_freq, sizing_mode="stretch_width")
            row_for_tab = pn.Row(time_pane, freq_pane, sizing_mode="stretch_width")
            p_tabs[f"{b0_list[i]:.2f}"] = row_for_tab


        m, c = np.polyfit(b0_list, np.array(freq_list), 1)

        linear_model = np.poly1d([m, c])
        
        p_cal = figure(title="B0 Calibration", sizing_mode="stretch_width", tools="pan,wheel_zoom,box_zoom,reset,save")
        
        p_cal.scatter(b0_list, np.array(freq_list)*1e-3, color="orange", legend_label="Data")
        
        x_fit = np.linspace(np.min(b0_list), np.max(b0_list), 200)
        
        p_cal.line(x_fit, linear_model(x_fit)*1e-3, color="blue", line_width=2, legend_label="Linear Fit")
        
        p_cal.xaxis.axis_label = "B0 (DAC units)"
        p_cal.yaxis.axis_label = "Frequency (kHz)"
        self.g.log_msg(f"B0 Calibration Value = {float(m):.3f} /Hz")



        p_cal.title.text_font_size = "18pt"
        
        p_cal.xaxis.axis_label_text_font_size = "16pt"
        p_cal.yaxis.axis_label_text_font_size = "16pt"
        
        p_cal.xaxis.major_label_text_font_size = "14pt"
        p_cal.yaxis.major_label_text_font_size = "14pt"
        
        p_cal.legend.label_text_font_size = "14pt"
        
        return p_cal, p_tabs, m
