from matipo import SEQUENCE_DIR, GLOBALS_DIR
from matipo.sequence import Sequence
import numpy as np
import panel as pn
import yaml
import logging

from .panes import panes


class panel_layout:
    """
    Contains the different panel layouts that you can use to display the panels.
    """

    def __init__(self, panes, ):
        self.panes = panes
        
    def p_layout_plotseq(self, seq_plot = None):
        """
        Creates a simple column based layout suitable for panels such as 1DSE with the two plots and the inputs underneath.
        use with: panel.servable() to initialise the panel in the notebook.

        Inputs:
            widget_dict (dict): dictionary of all the widgets that will be made for the panel inputs 
            save_dict (dict): dictionary containing the save k data input widget
            run (button widget): run button
            abort (button widget): abort button
            save (button widget): save button
            info (button widget): info button
            height (int): height of the plots pane
            
        Output:
            panel (obj): panel object
        """
        if seq_plot == None:
            panel = pn.Column(pn.Row(self.panes.seq_plot, width_policy = "max", sizing_mode="stretch_width"), pn.Row(pn.pane.Markdown("### Log:"), self.panes.plotstatus_pane), self.panes.plotlog_pane, sizing_mode="stretch_width")
        else:
            panel = pn.Column(pn.Row(seq_plot, sizing_mode="stretch_width"), pn.Row(pn.pane.Markdown("### Log:"), self.panes.plotstatus_pane), self.panes.plotlog_pane, sizing_mode="stretch_width")
        return panel

        
    def p_layout_simple(self, widget_dict, save_dict, run, abort, save, info_button, run_loop = None, freq_mm = None, height = 500, add_npz = None):
        """
        Creates a simple column based layout suitable for panels such as 1DSE with the two plots and the inputs underneath.
        use with: panel.servable() to initialise the panel in the notebook.

        Inputs:
            widget_dict (dict): dictionary of all the widgets that will be made for the panel inputs 
            save_dict (dict): dictionary containing the save k data input widget
            run (button widget): run button
            abort (button widget): abort button
            save (button widget): save button
            info (button widget): info button
            height (int): height of the plots pane
            
        Output:
            panel (obj): panel object
        """
        if run_loop != None:
            inputs_pane = pn.GridBox(*widget_dict.values(), pn.Row(run, abort), run_loop, pn.Row(pn.pane.Markdown("#### Run Progress:"), self.panes.progress_bar), ncols = 4, sizing_mode="stretch_width")
        else:
            inputs_pane = pn.GridBox(*widget_dict.values(), pn.Row(run, abort), pn.Row(pn.pane.Markdown("#### Run Progress:"), self.panes.progress_bar), ncols = 4, sizing_mode="stretch_width")
        file_save_pane = pn.Row(*save_dict.values(), save, sizing_mode="stretch_width")
        if add_npz != None:
            inputs_pane = pn.GridBox(*widget_dict.values(), pn.Row(run, abort, add_npz), pn.Row(pn.pane.Markdown("#### Run Progress:"), self.panes.progress_bar), ncols = 4, sizing_mode="stretch_width")
        if freq_mm == None:
            panel = pn.Column(pn.Row(pn.pane.Markdown("### Information:"), info_button), self.panes.info_pane, pn.Row(self.panes.plot_pane, self.panes.img_pane, sizing_mode="stretch_width", height = height), inputs_pane, file_save_pane, pn.Row(pn.pane.Markdown("### Log:"), self.panes.status_pane), self.panes.log_pane)
        if freq_mm != None:
            panel = pn.Column(pn.Row(pn.pane.Markdown("### Information:"), info_button), freq_mm,  self.panes.info_pane, pn.Row(self.panes.plot_pane, self.panes.img_pane, sizing_mode="stretch_width", height = height), inputs_pane, file_save_pane, pn.Row(pn.pane.Markdown("### Log:"), self.panes.status_pane), self.panes.log_pane)
        return panel


    
    def p_layout_simpleFOV(self, widget_dict, save_dict, run, abort, save, load_protocol, save_protocol, freq_mm = None, plot = None, height = 500, ):
        """
        Creates a simple column based layout suitable for panels such as 1DSE with the two plots and the inputs underneath.
        use with: panel.servable() to initialise the panel in the notebook.

        Inputs:
            widget_dict (dict): dictionary of all the widgets that will be made for the panel inputs 
            save_dict (dict): dictionary containing the save k data input widget
            run (button widget): run button
            abort (button widget): abort button
            save (button widget): save button
            info (button widget): info button
            height (int): height of the plots pane
            
        Output:
            panel (obj): panel object
        """
        values = list(widget_dict.values())
        p = 2
        m = p + 7
        n = m+1
        
        protocols = pn.GridBox(values[0], load_protocol, values[1], save_protocol, ncols = 4, sizing_mode="stretch_width")
        scan_pane = select_pane = pn.GridBox(*values[p:m], ncols = 4, sizing_mode="stretch_width")
        img_pars_pane = pn.GridBox(*values[m:n], pn.Row(pn.pane.Markdown("**FOV:**"), *values[n:]), pn.Row(run, abort, plot), pn.Row(pn.pane.Markdown("#### Run Progress:"), self.panes.progress_bar), ncols = 3, sizing_mode="stretch_width")

        file_save_pane = pn.Row(*save_dict.values(), save, sizing_mode="stretch_width")
       
        panel = pn.Column(pn.pane.Markdown("# 1DSE Hard Pulse"), freq_mm,  self.panes.info_pane, pn.Row(self.panes.plot_pane, self.panes.img_pane, sizing_mode="stretch_width", height = 400), protocols, pn.pane.Markdown("**Scan Parameters:**"), scan_pane, pn.pane.Markdown("**Image Parameters:**"), img_pars_pane, file_save_pane, pn.Row(pn.pane.Markdown("### Log:"), self.panes.status_pane), self.panes.log_pane, sizing_mode="stretch_width")
        return panel

    def p_layout_1DSEsoft(self, widget_dict, save_dict, run, abort, save, info_button,  soft_pulse, freq_mm = None,height = 500):
        """
        Creates a simple column based layout suitable for panels such as 1DSE with the two plots and the inputs underneath.
        use with: panel.servable() to initialise the panel in the notebook.

        Inputs:
            widget_dict (dict): dictionary of all the widgets that will be made for the panel inputs 
            save_dict (dict): dictionary containing the save k data input widget
            run (button widget): run button
            abort (button widget): abort button
            save (button widget): save button
            info (button widget): info button
            height (int): height of the plots pane
            
        Output:
            panel (obj): panel object
        """
        m = 1
        gp = m + 2
        values = list(widget_dict.values())
        select_pane = pn.Row(pn.pane.Markdown("**Select Pulse Type:**"), values[0])
        grad_pane = pn.Row(pn.pane.Markdown("**Slice Gradient:**"), *values[m:gp])
        inputs_pane = pn.GridBox(*values[gp:], pn.Row(soft_pulse, run, abort), pn.Row(pn.pane.Markdown("#### Run Progress:"), self.panes.progress_bar), ncols = 4, sizing_mode="stretch_width")

        file_save_pane = pn.Row(*save_dict.values(), save, sizing_mode="stretch_width")

        if freq_mm == None:
            panel = pn.Column(pn.Row(pn.pane.Markdown("### Information:"), info_button), self.panes.info_pane, pn.Row(self.panes.plot_pane, self.panes.img_pane, sizing_mode="stretch_width", height = height), select_pane, grad_pane, inputs_pane, file_save_pane, pn.Row(pn.pane.Markdown("### Log:"), self.panes.status_pane), self.panes.log_pane)
        if freq_mm != None:
            panel = pn.Column(pn.Row(pn.pane.Markdown("### Information:"), info_button), freq_mm,  self.panes.info_pane, pn.Row(self.panes.plot_pane, self.panes.img_pane, sizing_mode="stretch_width", height = height), select_pane, grad_pane, inputs_pane, file_save_pane, pn.Row(pn.pane.Markdown("### Log:"), self.panes.status_pane), self.panes.log_pane)
        return panel



    def all_pars_layout(self, run, abort, save, info_button):
        inputs_pane = pn.GridBox(*widget_dict.values(), pn.Row(run, abort), pn.Row(pn.pane.Markdown("#### Run Progress:"), self.panes.progress_bar), ncols = 4, sizing_mode="stretch_width")
            
        file_save_pane = pn.Row(*save_dict.values(), save, sizing_mode="stretch_width")
        if freq_mm == None:
            panel = pn.Column(pn.Row(pn.pane.Markdown("### Information:"), info_button), self.panes.info_pane, pn.Row(self.panes.soft_kplot, self.panes.soft_imgplot, self.soft_diffplot, sizing_mode="stretch_width", height = height), inputs_pane, file_save_pane, pn.Row(pn.pane.Markdown("### Log:"), self.panes.status_pane), self.panes.log_pane)

        return panel

    # soft_pulse_plot
    def p_layout_pulse_calSLR(self, widget_dict, save_dict, run, abort, save, load_protocol, save_protocol, run_loop, add_rf, plot, height = 500):
        """

        """
        values = list(widget_dict.values())
        p = 2
        rf = 4
        grad = rf + 2 
        protocols = pn.GridBox(values[0], load_protocol, values[1], save_protocol, ncols = 4, sizing_mode="stretch_width")

        select_pane = pn.Row(pn.pane.Markdown("Select Pulse Type:"), *values[p:rf])
        grad_pane = pn.Row(pn.pane.Markdown("Slice Select Gradient:"), *values[rf:grad])
        inputs_pane = pn.Row(*values[grad:])
        btn_pane = pn.GridBox(pn.Row(run, abort, run_loop, plot), pn.Row(pn.pane.Markdown("#### Run Progress:"), self.panes.progress_bar), add_rf, ncols = 6, sizing_mode="stretch_width")

            
        file_save_pane = pn.Row(*save_dict.values(), save, sizing_mode="stretch_width")
        
        panel = pn.Column(pn.Row(pn.pane.Markdown("# Soft Pulse Calibration")),  self.panes.soft_pulse_plot, protocols, select_pane, grad_pane, inputs_pane, btn_pane, file_save_pane, pn.Row(pn.pane.Markdown("### Log:"), self.panes.status_pane), self.panes.log_pane, sizing_mode="stretch_width")

        return panel

    def p_layout_pulse_calSLRtab(self, widget_dict, save_dict, run, abort, save, load_protocol, save_protocol, run_loop, add_rf, plot, height = 500):
        """

        """
        values = list(widget_dict.values())
        p = 2
        rf = 4
        grad = rf + 2 
        protocols = pn.GridBox(values[0], load_protocol, values[1], save_protocol, ncols = 4, sizing_mode="stretch_width")
        select_pane = pn.Row(pn.pane.Markdown("Select Pulse Type:"), *values[p:rf])
        grad_pane = pn.Row(pn.pane.Markdown("Slice Select Gradient:"), *values[rf:grad])
        inputs_pane = pn.Row(*values[grad:])
        btn_pane = pn.GridBox(pn.Row(run, abort, run_loop, plot), pn.Row(pn.pane.Markdown("#### Run Progress:"), self.panes.progress_bar), add_rf, ncols = 6, sizing_mode="stretch_width")

            
        file_save_pane = pn.Row(*save_dict.values(), save, sizing_mode="stretch_width")
        
        panel = pn.Column(pn.Row(pn.pane.Markdown("# Soft Pulse Calibration")),  self.panes.soft_pulse_plot, protocols, select_pane, grad_pane,  inputs_pane, btn_pane, file_save_pane, pn.Row(pn.pane.Markdown("### Log:"), self.panes.softstatus_pane), self.panes.softlog_pane, sizing_mode="stretch_width")

        return panel


        
    def p_layout_B0CAL(self, widget_dict, save_dict, run, abort, save, load_protocol, save_protocol, save_b0, plot, height = 550):
        """

        """
        values = list(widget_dict.values())
        p = 2
        protocols = pn.GridBox(values[0], load_protocol, values[1], save_protocol, ncols = 4, sizing_mode="stretch_width")
        
        inputs_pane = pn.GridBox(*values[p:], ncols = 6, sizing_mode="stretch_width")
        btn_pane = pn.GridBox(pn.Row(run, abort, save_b0, plot), pn.Row(pn.pane.Markdown("#### Run Progress:"), self.panes.progress_bar), ncols = 6, sizing_mode="stretch_width")

        file_save_pane = pn.Row(*save_dict.values(), save, sizing_mode="stretch_width")

        panel = pn.Column(pn.Row(pn.pane.Markdown("# B0 Calibration")), pn.Tabs(("Calibration Plot", self.panes.img_pane), ("Spectra Plots", self.panes.tabs_b0), sizing_mode="stretch_width"), pn.Column(protocols, inputs_pane, btn_pane, file_save_pane, pn.Row(pn.pane.Markdown("### Log:"), self.panes.status_pane), self.panes.log_pane,sizing_mode="stretch_width"), sizing_mode="stretch_width")

        return panel

    def p_layout_fit_ec(self, widget_dict, fit_Ebtn, fit_B0btn, fit_E_loop, fit_B0_loop, abort, load, load_protocol, save_protocol, fit_Ebtn_save, fit_B0btn_save, height = 550):
        """

        """
        values = list(widget_dict.values())
        n = 1
        p = n + 2
        e = p + 4
        b = e + 4
        
        measure_pane = pn.GridBox(widget_dict["measure_data_name"], load, ncols = 4)
        protocols = pn.GridBox(widget_dict["protocols"], load_protocol, widget_dict["protocol_name"], save_protocol, ncols=4, sizing_mode="stretch_width")

        
        E_inputs = pn.GridBox(widget_dict["N_tau_E"], widget_dict["lam_E"], widget_dict["tau_min_E"], widget_dict["tau_max_E"], widget_dict["N_exp_max_E"], ncols = 5, sizing_mode="stretch_width")
        E_pane = pn.Column(measure_pane, protocols, pn.pane.Markdown("# Gradient Fitting"),  pn.Row(self.panes.tau_E, self.panes.fit_E, self.panes.res_E, sizing_mode="stretch_width"), E_inputs, pn.Row(fit_Ebtn, fit_E_loop, abort, fit_Ebtn_save), pn.Row(pn.pane.Markdown("### Log:"), self.panes.status_pane), self.panes.log_pane, sizing_mode="stretch_width")
        
        B0_inputs = pn.GridBox(widget_dict["N_tau_B0"], widget_dict["lam_B0"], widget_dict["tau_min_B0"], widget_dict["tau_max_B0"], widget_dict["N_exp_max_B0"], ncols = 5, sizing_mode="stretch_width")
        B0_pane = pn.Column(measure_pane, protocols, pn.pane.Markdown("# Offset Fitting"),  pn.Row(self.panes.tau_B0, self.panes.fit_B0, self.panes.res_B0, sizing_mode="stretch_width"), B0_inputs, pn.Row(fit_B0btn, fit_B0_loop, abort, fit_B0btn_save), pn.Row(pn.pane.Markdown("### Log:"), self.panes.status_pane), self.panes.log_pane, sizing_mode="stretch_width")
        
        panel = pn.Tabs(
            ("Gradient", E_pane),
            ("Offset", B0_pane),
            sizing_mode="stretch_width",
        )

        return panel
                
    def p_layout_measure_ec(self, widget_dict, save_dict, run, abort, save, load_protocol, save_protocol, plot, height = 550):
        """

        """

        protocols = pn.GridBox(widget_dict["protocols"], load_protocol, widget_dict["protocol_name"], save_protocol, ncols=4, sizing_mode="stretch_width")

        axis_pane = pn.GridBox(pn.pane.Markdown("### Probe Configuration Axis:"), widget_dict["probe_axis"], ncols=6)
        
        scanpars_pane = pn.GridBox(pn.pane.Markdown("### Scan Parameters:"), widget_dict["n_scans"], widget_dict["flip_angle"], widget_dict["n_pulses"], widget_dict["spectral_width"], widget_dict["apodisation"], ncols=6, sizing_mode="stretch_width")
        
        times_pane = pn.GridBox(pn.pane.Markdown("### Time Parameters:"), widget_dict["t_acq_delay"], widget_dict["t_grad"], widget_dict["t_ramp"], widget_dict["t_wait"], widget_dict["t_end"], widget_dict["t_next"], ncols=6, sizing_mode="stretch_width")
        
        gradient_pane = pn.GridBox(pn.pane.Markdown("### Gradient:"), widget_dict["select_g"], widget_dict["g_val"], widget_dict["resolution"], ncols=6)
        channel_pane = pn.GridBox(pn.pane.Markdown("### Channels:"), widget_dict["tx_chan"], widget_dict["rx_l"], widget_dict["rx_c"], widget_dict["rx_r"], ncols=6)
        btn_pane = pn.GridBox(pn.Row(run, abort, plot), pn.Row(pn.pane.Markdown("#### Run Progress:"), self.panes.progress_bar), ncols = 6, sizing_mode="stretch_width")

        file_save_pane = pn.Row(*save_dict.values(), save, sizing_mode="stretch_width")

        panel = pn.Column(pn.pane.Markdown("# Measure Eddy Currents"), pn.Row(self.panes.tabs_measure, sizing_mode="stretch_width"), pn.Column(protocols, axis_pane, scanpars_pane, times_pane, gradient_pane, channel_pane, btn_pane, file_save_pane, pn.Row(pn.pane.Markdown("### Log:"), self.panes.status_pane), self.panes.log_pane,sizing_mode="stretch_width"), sizing_mode="stretch_width")

        return panel



    def p_layout_compile_fit(self, widget_dict, save_dict, run_nope, run_pe, abort, save, load_protocol, save_protocol, plot, height = 550):
        """

        """

        protocols = pn.GridBox(widget_dict["protocols"], load_protocol, widget_dict["protocol_name"], save_protocol, ncols=4, sizing_mode="stretch_width")
        scale_pane = pn.GridBox(pn.pane.Markdown("### Scaling of Pre-Emphasis Amplitudes from Fitting:"), 
                               widget_dict["x_e_factor"], 
                               widget_dict["y_e_factor"],
                               widget_dict["z_e_factor"],
                               widget_dict["x_b0_factor"],
                               widget_dict["y_b0_factor"],
                               widget_dict["z_b0_factor"],                               
                               ncols=7)

        axis_pane = pn.GridBox(pn.pane.Markdown("### Probe Configuration Axis:"), widget_dict["probe_axis"], ncols=6)
        
        scanpars_pane = pn.GridBox(pn.pane.Markdown("### Scan Parameters:"), widget_dict["n_scans"], widget_dict["flip_angle"], widget_dict["n_pulses"], widget_dict["spectral_width"], widget_dict["apodisation"], ncols=6, sizing_mode="stretch_width")
        
        times_pane = pn.GridBox(pn.pane.Markdown("### Time Parameters:"), widget_dict["t_acq_delay"], widget_dict["t_grad"], widget_dict["t_ramp"], widget_dict["t_wait"], widget_dict["t_end"], widget_dict["t_next"], ncols=6, sizing_mode="stretch_width")
        
        gradient_pane = pn.GridBox(pn.pane.Markdown("### Gradient:"), widget_dict["select_g"], widget_dict["g_val"], widget_dict["resolution"], ncols=6)
        channel_pane = pn.GridBox(pn.pane.Markdown("### Channels:"), widget_dict["tx_chan"], widget_dict["rx_l"], widget_dict["rx_c"], widget_dict["rx_r"], pn.pane.Markdown("Switch the left and right Rx Channel to change field gradient from positive/negative decay."), ncols=6)
        btn_pane = pn.GridBox(pn.Row(run_nope, run_pe, abort, plot, pn.pane.Markdown("Plot may take some time to load.")), pn.Row(pn.pane.Markdown("#### Run Progress:"), self.panes.progress_bar), ncols = 6, sizing_mode="stretch_width")

        file_save_pane = pn.Row(*save_dict.values(), save, sizing_mode="stretch_width")

        panel = pn.Column(pn.pane.Markdown("# Measure Eddy Currents"), pn.pane.Markdown("Analysis and Specta plots will be shown for the last run sequence in the tabs. Up to 8 pre-emphasis runs can be displayed in the decay plots at a time, and if run more than 8 times will rewrite over the least recent plot.\n Saves gradient and offset parameters into global with each run of a pre-emphasised sequence."), pn.Row(self.panes.tabs_measure_compiler, sizing_mode="stretch_width"), pn.Column(protocols, scale_pane, axis_pane, scanpars_pane, times_pane, gradient_pane, channel_pane, btn_pane, file_save_pane, pn.Row(pn.pane.Markdown("### Log:"), self.panes.status_pane), self.panes.log_pane,sizing_mode="stretch_width"), sizing_mode="stretch_width")

        return panel
    
    def p_layout_FID(self, widget_dict, save_dict, run, abort, save, load_protocol, save_protocol, global_shims, zero_shims, run_loop, plot, height = 500):
        """
        Creates a simple column based layout suitable for panels such as 1DSE with the two plots and the inputs underneath.
        use with: panel.servable() to initialise the panel in the notebook.

        Inputs:
            widget_dict (dict): dictionary of all the widgets that will be made for the panel inputs 
            save_dict (dict): dictionary containing the save k data input widget
            run (button widget): run button
            abort (button widget): abort button
            save (button widget): save button
            info (button widget): info button
            height (int): height of the plots pane
            
        Output:
            panel (obj): panel object
        """
        values = list(widget_dict.values())
        
        p = 2
        
        protocols = pn.GridBox(values[0], load_protocol, values[1], save_protocol, ncols = 4, sizing_mode="stretch_width")
        
        inputs_pane = pn.GridBox(*values[p:], pn.Row(run, abort), pn.Row(run_loop, plot), pn.Row(pn.pane.Markdown("#### Run Progress:"), self.panes.progress_bar), ncols = 5, sizing_mode="stretch_width")

            
        file_save_pane = pn.Row(*save_dict.values(), save, sizing_mode="stretch_width")
        
        panel = pn.Column(pn.Row("# Gradient Shimming"), pn.Row(self.panes.plot_pane, self.panes.img_pane, sizing_mode="stretch_width", height = height), protocols, inputs_pane, file_save_pane, pn.Row(pn.pane.Markdown("Save these shim settings to global:"), global_shims, pn.pane.Markdown("Reset the shim settings in global to zero:"), zero_shims), pn.Row(pn.pane.Markdown("### Log:"), self.panes.status_pane, sizing_mode="stretch_width"), pn.Row(self.panes.log_pane, sizing_mode="stretch_width"), sizing_mode="stretch_width")
        
        return panel


    def p_layout_rare_pe(self, widget_dict, save_dict, slice_dict, run, abort, save, load_protocol, save_protocol, button = None, plot = None, height = 500):
        """
        Creates a simple column based layout suitable for panels such as 1DSE with the two plots and the inputs underneath.
        use with: panel.servable() to initialise the panel in the notebook.

        Inputs:
            widget_dict (dict): dictionary of all the widgets that will be made for the panel inputs 
            save_dict (dict): dictionary containing the save k data input widget
            run (button widget): run button
            abort (button widget): abort button
            save (button widget): save button
            info (button widget): info button
            height (int): height of the plots pane
            
        Output:
            panel (obj): panel object
        """

        values = list(widget_dict.values())
        p = 2
        m = p+11
        rf = m+1
        pe = rf+1
        b0 = pe+1
        gr = b0 + 3
        g1 = gr + 3
        g2 = g1 + 6
        file = g2+2

        protocols = pn.GridBox(values[0], load_protocol, values[1], save_protocol, ncols = 4, sizing_mode="stretch_width") 
        inputs_pane = pn.GridBox(*values[p:m], ncols = 4, sizing_mode="stretch_width")
        rf_pane = pn.GridBox(pn.pane.Markdown("**Select Pulse Type:**", width = 120), *values[m:rf], pn.pane.Markdown("**Use Pre-Emphasis:**", width = 120), *values[rf:pe], pn.pane.Markdown("**Use B0 Compensation:**", width = 120), *values[pe:b0], ncols = 7, )

        g_read_pane = pn.GridBox(pn.pane.Markdown("**G Read:**", width = 120), *values[b0:gr], ncols = 5, )
        g_phase1_pane = pn.GridBox(pn.pane.Markdown("**G Phase:**", width = 120), *values[gr:g1], ncols = 4, )
        g_phase2_pane = pn.GridBox(pn.pane.Markdown("**G Slice:**", width = 120), *values[g1:g2], ncols = 7, )
        
        slice_pane = pn.GridBox(pn.pane.Markdown("**Gaussian Blur:**"), slice_dict["gaussian_blur"], ncols=8)
        file_find_pane = pn.GridBox(*values[g2:file], ncols = 1, sizing_mode="stretch_width")    
        file_save_pane = pn.Row(*save_dict.values(), save, sizing_mode="stretch_width")


        panel = pn.Column(self.panes.twoimagetabs, protocols, pn.pane.Markdown("### Pulse Sequence Parameters:"), inputs_pane,rf_pane, g_read_pane, g_phase1_pane, g_phase2_pane, pn.Row(run, abort, button, pn.Row(pn.pane.Markdown("#### Run Progress:"), self.panes.progress_bar), pn.Row(pn.pane.Markdown("Plot takes a while to load (lots of data points in RARE):"), plot), pn.pane.Markdown("Close napari pop-up to rerun sequence.")),  slice_pane, file_find_pane, file_save_pane, pn.Row(pn.pane.Markdown("### Log:"), self.panes.status_pane), self.panes.log_pane, sizing_mode="stretch_width")
        return panel



    def p_layout_2DSEmultiSLR(self, widget_dict, save_dict, slice_dict, run, abort, save, load_protocol, save_protocol, button = None, plot = None, height = 500):
        """
        Creates a simple column based layout suitable for panels such as 1DSE with the two plots and the inputs underneath.
        use with: panel.servable() to initialise the panel in the notebook.

        Inputs:
            widget_dict (dict): dictionary of all the widgets that will be made for the panel inputs 
            save_dict (dict): dictionary containing the save k data input widget
            run (button widget): run button
            abort (button widget): abort button
            save (button widget): save button
            info (button widget): info button
            height (int): height of the plots pane
            
        Output:
            panel (obj): panel object
        """

        values = list(widget_dict.values())
        p = 2
        m = p + 7
        rx = m+2
        rf = rx+1
        gr = rf + 3
        g1 = gr + 3
        g2 = g1 + 6

        protocols = pn.GridBox(values[0], load_protocol, values[1], save_protocol, ncols = 4, sizing_mode="stretch_width")
        inputs_pane = pn.GridBox(*values[p:m], ncols = 4, sizing_mode="stretch_width")
        rx_pane = pn.GridBox(*values[m:rx], ncols = 8, sizing_mode="stretch_width")
        rf_pane = pn.GridBox(pn.pane.Markdown("**Select Pulse Type:**", width = 120), *values[rx:rf], ncols = 5, )
        g_read_pane = pn.GridBox(pn.pane.Markdown("**G Read:**", width = 120), *values[rf:gr], ncols = 5, )
        g_phase1_pane = pn.GridBox(pn.pane.Markdown("**G Phase:**", width = 120), *values[gr:g1], ncols = 4, )
        g_phase2_pane = pn.GridBox(pn.pane.Markdown("**G Slice:**", width = 120), *values[g1:g2], ncols = 7, )
        
        slice_pane = pn.GridBox(pn.pane.Markdown("**Gaussian Blur:**"), slice_dict["gaussian_blur"], ncols=8)
    
        file_save_pane = pn.Row(*save_dict.values(), save, sizing_mode="stretch_width")

        panel = pn.Column(self.panes.twoimagetabs, protocols, pn.pane.Markdown("### Pulse Sequence Parameters:"), inputs_pane,rx_pane, rf_pane, g_read_pane, g_phase1_pane, g_phase2_pane, pn.Row(run, abort, pn.Row(pn.pane.Markdown("#### Run Progress:"), self.panes.progress_bar), button, plot, pn.pane.Markdown("Close napari pop-up to rerun sequence.")), pn.pane.Markdown("### Image Viewing Parameters:"),  slice_pane, file_save_pane, pn.Row(pn.pane.Markdown("### Log:"), self.panes.status_pane), self.panes.log_pane, sizing_mode="stretch_width")

        return panel


    def p_layout_3DSEFOV(self, widget_dict, save_dict, slice_dict, run, abort, save, load_protocol, save_protocol, plot = None, height = 500):
        """
        Creates a simple column based layout suitable for panels such as 1DSE with the two plots and the inputs underneath.
        use with: panel.servable() to initialise the panel in the notebook.

        Inputs:
            widget_dict (dict): dictionary of all the widgets that will be made for the panel inputs 
            save_dict (dict): dictionary containing the save k data input widget
            run (button widget): run button
            abort (button widget): abort button
            save (button widget): save button
            info (button widget): info button
            height (int): height of the plots pane
            
        Output:
            panel (obj): panel object
        """

        values = list(widget_dict.values())
        p = 2
        m = p + 10
        gr = m + 3
        g1 = gr + 3
        g2 = g1 + 3

        
        protocols = pn.GridBox(values[0], load_protocol, values[1], save_protocol, ncols = 4, sizing_mode="stretch_width")
        
        inputs_pane = pn.GridBox(*values[p:m], ncols = 4, sizing_mode="stretch_width")
        g_read_pane = pn.GridBox(pn.pane.Markdown("**G Read:**", width = 120), *values[m:gr], ncols = 5, )
        g_phase1_pane = pn.GridBox(pn.pane.Markdown("**G Phase 1:**", width = 120), *values[gr:g1], ncols = 4, )
        g_phase2_pane = pn.GridBox(pn.pane.Markdown("**G Phase 2:**", width = 120), *values[g1:g2], ncols = 4, )

        slice_pane = pn.GridBox(pn.pane.Markdown("**Gaussian Blur:**"), slice_dict["gaussian_blur"], ncols=8)
    
        file_save_pane = pn.Row(*save_dict.values(), save, sizing_mode="stretch_width")


        panel = pn.Column(protocols, pn.pane.Markdown("### Pulse Sequence Parameters:"), inputs_pane, g_read_pane, g_phase1_pane, g_phase2_pane, pn.Row(run, abort, pn.Row(pn.pane.Markdown("#### Run Progress:"), self.panes.progress_bar), plot, pn.pane.Markdown("Close napari pop-up to rerun sequence.")), pn.pane.Markdown("### Image Viewing Parameters:"),  slice_pane, file_save_pane, pn.Row(pn.pane.Markdown("### Log:"), self.panes.status_pane), self.panes.log_pane, sizing_mode="stretch_width")
            
        return panel

    
    def p_layout_img_viewer1(self, widget_dict, run_img, load, height = 500):
        values = list(widget_dict.values())
        # filepath_pane = pn.Column(*save_dict.values(), sizing_mode="stretch_width")
        inputs_pane = pn.GridBox(values[0], load, values[1], ncols = 5)

        panel = pn.Column(pn.pane.Markdown("# View 1D Data"), pn.Row(self.panes.plot_pane, self.panes.img_pane, sizing_mode="stretch_width", height = 400), inputs_pane, pn.Row(run_img), pn.pane.Markdown("### Log:"), self.panes.log_pane, sizing_mode="stretch_width")
        return panel

    
    def p_layout_img_viewer2(self, widget_dict, run_img, load, height = 500):
        values = list(widget_dict.values())
        # filepath_pane = pn.Column(*save_dict.values(), sizing_mode="stretch_width")
        inputs_pane = pn.GridBox(values[0], load, values[1], values[2], ncols = 5)

        panel = pn.Column(pn.pane.Markdown("# View 2D Images"), self.panes.twoimagetabs, inputs_pane, pn.Row(run_img), pn.pane.Markdown("### Log:"), self.panes.log_pane, sizing_mode="stretch_width")
        return panel

    
            
    def p_layout_img_viewer3(self, widget_dict, run_img, load, height = 500):
        values = list(widget_dict.values())
        # filepath_pane = pn.Column(*save_dict.values(), sizing_mode="stretch_width")
        inputs_pane = pn.GridBox(values[0], load, values[1], values[2], ncols = 5)

        panel = pn.Column(pn.pane.Markdown("# View 3D Images"), inputs_pane, pn.Row(run_img), pn.pane.Markdown("### Log:"), self.panes.log_pane, sizing_mode="stretch_width")
        return panel
