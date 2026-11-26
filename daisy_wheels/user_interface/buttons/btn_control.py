import asyncio
import traceback
from datetime import date, datetime
from pathlib import Path
import panel as pn
import json
import numpy as np
import os
import yaml
from matipo import SEQUENCE_DIR, GLOBALS_DIR
import pandas as pd
from user_interface.general import general_functions
from user_interface.panes import panes
from user_interface.updating import updating
from user_interface.oneDfunctions import oneDfunctions

class ButtonController:
    """
    Functions that work with the buttons in the pulse seq panels.
    Button functions only want one parameter "event" so we need to pass the rest through self which 
    is only seen as a parameter inside the class.
    """
    def __init__(self, general_functions, panes, updating):
        """
        Initialise parameters to be updated with button clicks.
        """
        self.g = general_functions
        self.panes = panes
        self.u = updating
        self.oneDfunctions = oneDfunctions(self.g, self.panes)


        self.total_runs = 0
        self.run_task = None

        self.run_seq = None
        self.widget_dict = None
        self.scan_type = None
        self.filename_widget = None
        self.kdata_newest = None
        self.array_kdata = None

        self.loop_task = None
        self.soft_pulse_open = False

    def wire(self, run_seq, widget_dict, button_dict, scan_type, save_dict, dimensions, loop_run_seq = None, non_input_params = None, calc_width = 'n', update_secs = 5, kdata = None, img = None, file = None, save_folder = None, fit_Ebtn = None, fit_B0btn = None, fit_Ebtn_loop = None, fit_B0btn_loop = None, abort_fitting = None):
        """
        Send the information into the class from the notebook to be applied when buttons are pressed.
        
        async def run_seq_wrapped():
            await rps.run_seq(pulse_file, widget_dict, scan_type, dimensions, add_amplitudes = 'y')
            
        Inputs:
            run_seq (function): pulse run sequence - use a wrapped function as dont want what is passed into 
                the state to take arguments
            widget_dict (dict):
            button_dict (dict):
            scan_type (str):
            save_dict (dict):
        """
        self.run_seq = run_seq
        self.widget_dict = widget_dict
        self.scan_type = scan_type
        if scan_type != "fitting_ec":
            self.filename_widget = save_dict["filepath"]
        self.button_dict = button_dict

        self.dimensions = dimensions
        if self.dimensions == 1:
            self.freq_mm = 1
        else:
            self.freq_mm = None
        self.non_input_params = non_input_params
        self.calc_width = calc_width
        self.run_loop = False
        self.update_secs = update_secs
        self.loop_run_seq = loop_run_seq
        self.stop_requested = False
        self.save_folder = save_folder
        self.file = file
        self.fitting_dataset = None
        self.fit_Ebtn = fit_Ebtn
        self.fit_B0btn = fit_B0btn
        self.fit_Ebtn_loop = fit_Ebtn_loop
        self.fit_B0btn_loop = fit_B0btn_loop
        self.abort_fitting = abort_fitting
        self.df_freq = None
        self.g_hzpm = 0
        self.folder_path_fitting = None
        self.df_fit_E = None
        self.df_fit_B0 = None


    def load_fitting_data_on_click(self, event):
        # grab files and save to self
        folder_path = Path("data") / self.save_folder / self.widget_dict["measure_data_name"].value
        self.folder_path_fitting = folder_path
        
        try: 
            self.df_freq = pd.read_csv(folder_path/"df_freq.csv")
            self.df_ = self.df_freq[1:] # cut first acq when grad was on
            
            with open(folder_path/"seq_par.txt", "r") as f:
                seq_par = f.read()
                seq_par = eval(seq_par, {"array": np.array})
            self.g.log_msg(f"Scan parameters for loaded sequence: \n{seq_par}")
            self.g_hzpm = self.g.get_yaml_info(info_name = "g_hzpm.yaml", file = folder_path)['g_hzpm_loaded']
            
            
            # invert gradient if negative so can fit positive values
            baseline = self.df_["g_eddy"][:3].mean()
            if baseline < 0:
                self.g.log_msg(f"baseline: {baseline}")
                self.g.log_msg("Inverting negative dataset for GRADIENT")
                self.df_["g_eddy"] = self.df_["g_eddy"] * -1
                self.df_freq["g_eddy"] = self.df_freq["g_eddy"] * -1
                self.df_["E(t)"] = self.df_["E(t)"] * -1
                self.df_freq["E(t)"] = self.df_freq["E(t)"] * -1
            else:
                self.g.log_msg(f"baseline: {baseline}")
                


            
        except Exception:
            tb = traceback.format_exc()
            self.g.log_msg("Error loading data:")
            self.g.status_msg("Error Loading Data")
            self.g.log_msg(tb)
        try:
            # self.oneDfunctions.eddy_decay_plot(self.df_freq, height = 500, fitting = "y")
            pn.state.curdoc.add_next_tick_callback(lambda: self.oneDfunctions.eddy_decay_plot(self.df_freq, height = 500))

            # re-enable the fitting buttons
            self.fit_Ebtn.disabled = False
            self.fit_B0btn.disabled = False
            self.fit_Ebtn_loop.disabled = False
            self.fit_B0btn_loop.disabled = False
            self.abort_fitting.disabled = False
        except Exception:
            tb = traceback.format_exc()
            self.g.log_msg("Error plotting loaded:")
            self.g.status_msg("Error Plotting Data")
            self.g.log_msg(tb)
        

    def fitting_E_on_click(self, event):
        # trigger updating of the E plots in the fitting panel layout 
        
        try:
            if self.widget_dict["tau_min_E"].value > self.widget_dict["tau_max_E"].value:
                self.g.log_msg("Value Error:")
                self.g.status_msg("Error! CHANGE: tau so min < max")
                raise ValueError("tau min needs to be less than tau max") 

            self.MAX_EXP = self.g.get_yaml_info(info_name = "fitting_pars.yaml", file = self.file)['MAX_EXP']
            #  plot for the tau and the E with that data
            self.df_fit_E, e_fit, taus_E, x_E, props_E, pk_E, self.fit_params_E = self.oneDfunctions.fit(self.widget_dict["N_tau_E"].value, self.widget_dict["lam_E"].value, self.widget_dict["tau_min_E"].value, self.widget_dict["tau_max_E"].value, self.df_["E(t)"], self.df_["times (s)"], N_exp_max = self.widget_dict["N_exp_max_E"].value, type_ = "E", MAX_EXP = self.MAX_EXP, g_hzpm = self.g_hzpm)
    
            # e_fit = fitted_values_E * self.g_hzpm
    
            self.oneDfunctions.fit_onto_E(e_fit, self.df_["times (s)"], taus_E, x_E, props_E, pk_E, self.widget_dict["N_exp_max_E"].value)
            if hasattr(self, "_fit_E_callback"):
                self.g.status_msg("Gradient Fitting Loop")
            else:
                self.g.status_msg("Fit Gradient")
        except Exception:
            tb = traceback.format_exc()
            self.g.log_msg("Error fitting gradient data:")
            self.g.status_msg("Error Fitting Gradient")
            self.g.log_msg(tb)     

    def fitting_B0_on_click(self, event):
        # trigger updating of the E plots in the fitting panel layout 
        try:
            if self.widget_dict["tau_min_B0"].value > self.widget_dict["tau_max_B0"].value:
                self.g.log_msg("Value Error:")
                self.g.status_msg("Error! CHANGE: tau so min < max")
                raise ValueError("tau min needs to be less than tau max")

            
            self.MAX_EXP = self.g.get_yaml_info(info_name = "fitting_pars.yaml", file = self.file)['MAX_EXP']
            self.df_fit_B0, b0_fit, taus_B0, x_B0, props_B0, pk_B0, self.fit_params_B0 = self.oneDfunctions.fit(self.widget_dict["N_tau_B0"].value, self.widget_dict["lam_B0"].value, self.widget_dict["tau_min_B0"].value, self.widget_dict["tau_max_B0"].value, self.df_["B0(t)"], self.df_["times (s)"], N_exp_max = self.widget_dict["N_exp_max_B0"].value, type_ = "B0", MAX_EXP = self.MAX_EXP, g_hzpm = self.g_hzpm)
    
            # b0_fit = fitted_values_B0 * self.g_hzpm
    
            self.oneDfunctions.fit_onto_B0(b0_fit, self.df_["times (s)"], taus_B0, x_B0, props_B0, pk_B0, self.widget_dict["N_exp_max_B0"].value)
            if hasattr(self, "_fit_B0_callback"):
                self.g.status_msg("Offset Fitting Loop")
            else:
                self.g.status_msg("Fit Offset")
        except Exception:
            tb = traceback.format_exc()
            self.g.log_msg("Error fitting offset data:")
            self.g.status_msg("Error Fitting Offset")
            self.g.log_msg(tb)    

    def fitting_E_loop_on_click(self, event):
        self.g.status_msg("Gradient Fitting Loop")
        
    
        self._fit_E_callback = pn.state.add_periodic_callback(
            lambda: self.fitting_E_on_click(None),
            period=1500,  # milliseconds
        )
        
    def fitting_B0_loop_on_click(self, event):
        self.g.status_msg("Offset Fitting Loop")

        self._fit_B0_callback = pn.state.add_periodic_callback(
            lambda: self.fitting_B0_on_click(None),
            period=1500,
        )
        
    def abort_fitting_loop(self, event):

        if hasattr(self, "_fit_E_callback"):
            self._fit_E_callback.stop()
            self.g.status_msg("Aborted Gradient Loop")
        if hasattr(self, "_fit_B0_callback"):
            self._fit_B0_callback.stop()
            self.g.status_msg("Aborted Offset Loop")


    def save_fitting_E(self, event):

        if self.df_fit_E is None:
            self.g.status_msg("Error: Fit Saving")
            self.g.log_msg("Fit to enable saving.")
            raise ValueError("Nothing to save.")
        else:
            self.df_fit_E.to_csv(f"{self.folder_path_fitting}/df_fitE.csv", index=False)
            with open(f"{self.folder_path_fitting}/fit_paramsE.yaml", "w") as f:
                yaml.safe_dump(self.fit_params_E, f, sort_keys=False)
            
            
            self.g.log_msg(f"Saved Fitting Parameters to {self.folder_path_fitting}/df_fitE.csv")
            self.g.status_msg("Saved Gradient Fitting Parameters")



    def save_fitting_B(self, event):

        if self.df_fit_B0 is None:
            self.g.status_msg("Error: Fit Saving")
            self.g.log_msg("Fit to enable saving.")
            raise ValueError("Nothing to save.")
        else:
            self.df_fit_B0.to_csv(f"{self.folder_path_fitting}/df_fitB0.csv", index=False)
            with open(f"{self.folder_path_fitting}/fit_paramsB0.yaml", "w") as f:
                yaml.safe_dump(self.fit_params_B0, f, sort_keys=False)
            self.g.log_msg(f"Saved Fitting Parameters to {self.folder_path_fitting}/df_fitB0.csv")
            self.g.status_msg("Saved Offset Fitting Parameters")


    def run_on_click(self, event):
        self.total_runs += 1
        self.g.log_msg(f"### Run Number {self.total_runs}")
        self.stop_requested = False

        if self.scan_type == "grad_cal":
            self.g.threshold = self.widget_dict["width_threshold"].value/100
            self.g.log_msg(f"saved: {self.g.threshold}")
    
        loop = asyncio.get_running_loop()
        self.run_task = loop.create_task(self.run_seq())
        self.run_task = None


    def run_no_pe_on_click(self, event):
        self.total_runs += 1
        self.g.pe = False
        self.g.log_msg(f"### Run Number {self.total_runs}")
        self.stop_requested = False

        loop = asyncio.get_running_loop()
        self.run_task = loop.create_task(self.run_seq())
        self.run_task = None
        if self.g.no_pe_counter == 8:
            self.g.no_pe_counter = 0
        else:
            self.g.no_pe_counter += 1
        

    
    def run_pe_on_click(self, event):
        self.total_runs += 1
        self.g.pe = True
        self.g.log_msg(f"### Run Number {self.total_runs}")
        self.stop_requested = False

        loop = asyncio.get_running_loop()
        self.run_task = loop.create_task(self.run_seq())
        self.run_task = None
        if self.g.pe_counter == 8:
            self.g.pe_counter = 0
        else:
            self.g.pe_counter += 1
        self.g.log_msg(f"PE Plot Counter: {self.g.pe_counter}")
        


    def run_loop_on_click(self, event):
        if self.loop_task and not self.loop_task.done():
            self.g.log_msg("Run loop already running.")
            return
    
        self.total_runs += 1
        self.g.log_msg(f"### Loop: Run Number {self.total_runs}")
        self.run_loop = True
        self.stop_requested = False

        if self.scan_type == "grad_cal":
            self.g.threshold = self.widget_dict["width_threshold"].value/100
    
        loop = asyncio.get_running_loop()
        self.loop_task = loop.create_task(self.loop_run_seq())


    def abort_on_click(self, event):
        try:
            self.g.log_msg("Aborting...")
            self.stop_requested = True
    
            # stop active acquisition
            if self.run_task and not self.run_task.done():
                self.run_task.cancel()
                self.run_task = None
                self.g.log_msg("Active run cancelled.")
    
            # stop loop controller
            if self.loop_task and not self.loop_task.done():
                self.run_loop = False
                self.loop_task.cancel()
                self.g.log_msg("Run loop cancelled.")
    
            self.g.status_msg("#### Run Aborted")
            
    
        except Exception:
            tb = traceback.format_exc()
            self.g.log_msg("Error aborting run:")
            self.g.log_msg(tb)

    def add_grad_npz(self, event):
        
        if self.widget_dict['select_xyz'].value == "X":
            file_path = "xgrad_cal_data.npz"
            
            # New values
            new_strength = self.u.gradient_strength # can't remember what this is - not used in analysis
            new_val =self.widget_dict['grad_select'].value
            new_peak = self.u.peak_width
            if os.path.exists(file_path):
                data = np.load(file_path)
            
                x_strength = list(data["x_strength"])
                x_vals = list(data["x_vals"])
                g_crusher = list(data["g_crusher"])
                peak_width = list(data["peak_width"])
            else:
                x_strength = []
                x_vals = []
                g_crusher = []
                peak_width = []
            
            # Append new data
            x_strength.append(new_strength)
            x_vals.append(new_val)
            g_crusher.append(self.widget_dict['g_crusher'].value)
            peak_width.append(new_peak)
            
            # Save 
            np.savez(file_path,
                     x_strength=np.array(x_strength),
                     x_vals=np.array(x_vals),
                     g_crusher=np.array(g_crusher),
                     peak_width=np.array(peak_width))
        
        if self.widget_dict['select_xyz'].value == "Y": 
            file_path = "ygrad_cal_data.npz"
            
            # New values
            new_strength = self.u.gradient_strength
            new_val = self.widget_dict['grad_select'].value
            new_peak = self.u.peak_width
            
            if os.path.exists(file_path):
                data = np.load(file_path)
            
                y_strength = list(data["y_strength"])
                y_vals = list(data["y_vals"])
                g_crusher = list(data["g_crusher"])
                peak_width = list(data["peak_width"])
            else:
                y_strength = []
                y_vals = []
                g_crusher = []
                peak_width = []
            
            # Append new data
            y_strength.append(new_strength)
            y_vals.append(new_val)
            g_crusher.append(self.widget_dict['g_crusher'].value)
            peak_width.append(new_peak)
            
            # Save 
            np.savez(file_path,
                     y_strength=np.array(y_strength),
                     y_vals=np.array(y_vals),
                     g_crusher=np.array(g_crusher),
                     peak_width=np.array(peak_width)
                    )


        if self.widget_dict['select_xyz'].value == "Z": 
            file_path = "zgrad_cal_data.npz"
            
            # New values
            new_strength = self.u.gradient_strength
            new_val = self.widget_dict['grad_select'].value
            new_peak = self.u.peak_width
            
            if os.path.exists(file_path):
                data = np.load(file_path)
            
                z_strength = list(data["z_strength"])
                z_vals = list(data["z_vals"])
                g_crusher = list(data["g_crusher"])
                peak_width = list(data["peak_width"])
            else:
                z_strength = []
                z_vals = []
                g_crusher = []
                peak_width = []
            
            # Append new data
            z_strength.append(new_strength)
            z_vals.append(new_val)
            g_crusher.append(self.widget_dict['g_crusher'].value)
            peak_width.append(new_peak)
            
            # Save 
            np.savez(file_path,
                     z_strength=np.array(z_strength),
                     z_vals=np.array(z_vals),
                     g_crusher=np.array(g_crusher),
                     peak_width=np.array(peak_width))
        self.g.log_msg(f"Added to npz for {self.widget_dict['select_xyz'].value}")

    
    def save_on_click(self, event):
        """
        Saves the k data stored when clicked to the filename if given or to a pre-made 
        filename based on the pulse seq run and time/date. Will save in a folder locally to the notebook it 
        is run in in a folder that is made if not alreay existing called kdata
        """
        
        if self.kdata_newest is None:
            self.g.log_msg("No data to save.")
            return

        filename = self.filename_widget.value
        today = date.today()

        if not filename:
            t = datetime.now().strftime("%H-%M")
            filename = f"{self.scan_type}_run{self.total_runs}_{today}_{t}"

        save_dir = Path(f"C:/Users/halesda/OneDrive - Victoria University of Wellington - STAFF/Desktop/kahu-workspace/data/{self.save_folder}")
        save_dir.mkdir(exist_ok=True)

        path = save_dir / f"{filename}.npz"

        if self.dimensions == 3:
            self.g.log_msg(f"!!!!! imaging: {self.g.threeD_info}")
            np.savez(path, kdata=self.kdata_newest, scan_info={**self.g.scan_info_dict, **self.g.threeD_info})

        elif self.dimensions == 2:
            if self.scan_type not in ["2DSE_multi", "rare", "rare_pe", "2DSE_multiSLR"]:
                np.savez(path, kdata=self.kdata_newest, scan_info={**self.g.scan_info_dict, **self.g.twoD_info})
            else:
                np.savez(path, kdata=self.array_kdata, scan_info={**self.g.scan_info_dict, **self.g.twoD_info})
        else:
            np.savez(path, kdata=self.kdata_newest, scan_info=self.g.scan_info_dict)
        self.g.log_msg(f"Saved: {path}")

    def toggle_info(self, event):
        """
        Displays the info_pane when clicked if not visible and hides when clicked if visible.
        """
        self.panes.info_pane.visible = not self.panes.info_pane.visible
        if self.panes.info_pane.visible == True:
            self.button_dict['toggle_info'].name = ">"
        else:
            self.button_dict['toggle_info'].name = "^" 

    def apply_shims_on_click(self, event):
        """
        On the click of the button - retrieves the current gradient values and uploads them onto the 
        the shims.yaml file to be applied gloabally.
        """
        try:
            new_offsets = [float(self.g.scan_parameters['g_shims'][0]), float(self.g.scan_parameters['g_shims'][1]),float(self.g.scan_parameters['g_shims'][2]), 0.0] # yaml can store floats not np.float
            self.g.change_offsets(new_offsets, file = self.file)
            self.g.log_msg(f"Saved the offsets to global: {new_offsets}")
        except Exception:
            tb = traceback.format_exc()
            self.g.log_msg("Error in Applying Shims to Global:")
            self.g.status_msg("Error Applying Shims to YAML")
            self.g.log_msg(tb)
            return

    def save_B0cal(self, event):
        try:
            self.g.set_B0cal(file = self.file)
            self.g.log_msg(f"Saved the B0_cal to global: {self.g.B0_cal} Hz/DACunits")
            
        except Exception:
            tb = traceback.format_exc()
            self.g.log_msg("Error in Applying B0 Calibration to Global:")
            self.g.status_msg("Error Applying B0 Calibration to YAML")
            self.g.log_msg(tb)
            return         


    def save_B0cal(self, event):
        try:
            self.g.set_B0cal(file = self.file)
            self.g.log_msg(f"Saved the B0_cal to global: {self.g.B0_cal} Hz/DACunits")
            
        except Exception:
            tb = traceback.format_exc()
            self.g.log_msg("Error in Applying B0 Calibration to Global:")
            self.g.status_msg("Error Applying B0 Calibration to YAML")
            self.g.log_msg(tb)
            return       
    

    def apply_soft_rf_on_click(self, event):
        """
        On the click of the button - retrieves the current gradient values and uploads them onto the 
        the shims.yaml file to be applied gloabally.
        """
        try:
            rf = float(self.g.scan_parameters['a_90']) # yaml can store floats not np.float
            t_90 = float(self.g.scan_parameters['t_90']) # yaml can store floats not np.float
            bw_90 = float(self.g.scan_parameters['bw_90']) # yaml can store floats not np.float
            if self.g.scan_parameters["soft_pulse_type"] == "3-lobe sinc":
                self.g.set_3rf(rf, t_90, bw_90, file = self.file)
                if scan_type != "pulse_cal_SLR_tab":
                    self.g.log_msg(f"Saved the rf to 3-global: {rf}")
                    self.g.status_msg(f"Saved the rf to 3-global: {rf}")
                else:
                    self.g.softlog_msg(f"Saved the rf to 3-global: {rf}")
                    self.g.softstatus_msg(f"Saved the rf to 3-global: {rf}")
            if self.g.scan_parameters["soft_pulse_type"] == "5-lobe sinc":
                self.g.set_5rf(rf, t_90, bw_90, file = self.file)
                if scan_type != "pulse_cal_SLR_tab":
                    self.g.log_msg(f"Saved the rf to 5-global: {rf}")
                    self.g.status_msg(f"Saved to 5-global: {rf}")
                else:
                    self.g.softlog_msg(f"Saved the rf to 5-global: {rf}")
                    self.g.softstatus_msg(f"Saved to 5-global: {rf}")

            if self.g.scan_parameters["soft_pulse_type"] == "SLR":
                self.g.set_SLRrf(rf, t_90, bw_90, file = self.file)
                if scan_type != "pulse_cal_SLR_tab":
                    self.g.log_msg(f"Saved the rf to SLR-global: {rf}")
                    self.g.status_msg(f"Saved to SLR-global: {rf}")
                else:
                    self.g.softlog_msg(f"Saved the rf to SLR-global: {rf}")
                    self.g.softstatus_msg(f"Saved to SLR-global: {rf}")

        except Exception:
            tb = traceback.format_exc()
            if scan_type != "pulse_cal_SLR_tab":
                self.g.log_msg("Error in Applying RF to Global:")
                self.g.status_msg("Error Applying RF to YAML")
                self.g.log_msg(tb)
            else:
                self.g.softlog_msg("Error in Applying RF to Global:")
                self.g.softstatus_msg("Error Applying RF to YAML")
                self.g.softlog_msg(tb)
      
            return
            
            
    def reset_shims_on_click(self, event):
        """
        On the click of the button - sets the global offsets in the shims.yaml file to all be 0.
        """
        try:
            new_offsets = [0,0,0,0]
            self.g.change_offsets(new_offsets, file = self.file)
            self.g.log_msg(f"Saved the offsets to global: {new_offsets}")
        except Exception:
            tb = traceback.format_exc()
            self.g.log_msg("Error in Resetting Shims in Global:")
            self.g.log_msg(tb)
            return

    def change_freq_mm(self, event):
        if self.freq_mm == 0:
            self.button_dict['freq_mm_button'].name = "kHz"
            self.freq_mm = 1 # freq
        elif self.freq_mm == 1:
            self.freq_mm = 0 #mm
            self.button_dict['freq_mm_button'].name = "m"
        if self.freq_mm != None:
            self.u.update_plot(self.kdata_newest, self.widget_dict, self.dimensions, self.scan_type, self.non_input_params, freq_mm = self.freq_mm, calc_width = self.calc_width, file = self.file)


    def apply_protocol_on_click(self, event):
        """
        On the click of the button - applies protocol loaded into widgets
        """
        try:
            # load the protocol file
            filename = self.widget_dict['protocols'].value
            save_folder = self.g.get_yaml_info(info_name='folder_names.yaml',file=GLOBALS_DIR)[self.scan_type]
            if self.scan_type == "fitting_ec":
                file_path = Path("protocols") / "Fit_EC" / filename
            else:
                file_path = Path("protocols") / save_folder / filename
            
            with open(file_path, "r") as f:
                protocol = json.load(f)

                
            # set the protocol value to 
            for param, val in protocol.items():
                if param in self.widget_dict:
                    self.widget_dict[param].value = val

            self.g.log_msg(f"Loaded protocol: {filename}")
            self.g.status_msg(f"Loaded Protocol")
        except Exception:
            tb = traceback.format_exc()
            self.g.log_msg("Error in Applying Protocol to Widgets:")
            self.g.status_msg("Error Applying Protocol")
            self.g.log_msg(tb)
            return


    def save_protocol_on_click(self, event):
        """
        On the click of the button - applies protocol loaded into widgets
        """
        try:
            protocol = {}
            for par in self.widget_dict:
                protocol[par] = self.widget_dict[par].value
    
            filename = self.widget_dict['protocol_name'].value
            
            if self.scan_type == "fitting_ec":
                save_folder = "Fit_EC"
            else:
                save_folder = self.g.get_yaml_info(info_name='folder_names.yaml',file=GLOBALS_DIR)[self.scan_type]
            folder_path = Path("protocols") / save_folder
            folder_path.mkdir(parents=True, exist_ok=True)
            txt_file = folder_path / f"{filename}.txt"
    
            with open(txt_file, "w") as f:
                json.dump(protocol, f, indent=4)
            self.g.log_msg(f"Saved protocol as: {filename}.txt")
            self.g.status_msg(f"Saved Protocol")
            
            self.widget_dict["protocols"].options = self.g.get_protocol_files(self.scan_type)
    
        except Exception:
            tb = traceback.format_exc()
            self.g.log_msg("Error in Applying Protocol to Widgets:")
            self.g.status_msg("Error Applying Protocol")
            self.g.log_msg(tb)
            return
            