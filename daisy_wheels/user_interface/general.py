from matipo import SEQUENCE_DIR, GLOBALS_DIR
from matipo.sequence import Sequence
import numpy as np
import panel as pn
import yaml
import logging
from pathlib import Path
import pandas as pd

from .panes import panes

class general_functions:
    """
    General functions that can be applied in most notebooks for retrieving and filtering variables alongside 
    creating messages in the panels.
    """

    def __init__(self, panes):
        """
        Imports the class panes 
        """
        self.panes = panes
        self.scan_parameters = None
        self.threeD_info = None
        self.twoD_info = None
        self.autosave = None
        self.seq_run = None
        self.seq_on = None
        self.B0_cal = None
        self.fit_btn = None
        self.pe = None
        self.pe_counter = 0
        self.no_pe_counter = 0

    @staticmethod
    def update_file_list(event, folder_widget, file_widget):
    
        selected_folders = folder_widget.value
    
        files = []
    
        for folder in selected_folders:
            folder_path = Path("data") / folder
    
            if folder_path.exists():
                files.extend(
                    [f.name for f in folder_path.iterdir() if f.is_file()]
                )
    
        file_widget.options = sorted(files)
    
        if files:
            file_widget.value = [files[0]]
        else:
            file_widget.value = []
            

        
    @staticmethod
    def grab_measurement_files(save_folder, files_widget):
    
        folders = []
    
        folder_path = Path("data") / save_folder
    
        if folder_path.exists():
            folders.extend(
                [f.name for f in folder_path.iterdir() if f.is_dir()]
            )
    
        files_widget.options = sorted(folders)

    
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
            file = GLOBALS_DIR
 
        if info_name == None:
            info_name = 'shims'

            # general_functions.log_msg(f"Path: {file}/{info_name}")
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
            print(f"The file: {info_name} does not exist in {file}. {file}/{info_name}")



    @staticmethod
    def empty_fit():
        return {
            "A0": 0,
            "A1": 0,
            "tau1": 0,
            "A2": 0,
            "tau2": 0,
            "A3": 0,
            "tau3": 0,
            "A4": 0,
            "tau4": 0,
        }
        
    @staticmethod
    def compile_fitting_pars(measure_folder, globals_path):
        """
        """
        p = Path.cwd()
    
        while p != p.parent:
            if (p / "data").exists():
                print("Found:", p / "data")
                break
            p = p.parent
        
        data_path = p
        
        gradient_summary = {}
        offset_summary = {}
        empty_axes = []
        for axis in ["X", "Y", "Z"]:
        
            folder = data_path / 'data'/ measure_folder / f"{axis}grad_{axis}eddy"
            print(folder)
        
            if not folder.exists(): # if it hasn't been measured then will just be zeros instead
                gradient_summary[axis] = general_functions.empty_fit()
                offset_summary[axis] = general_functions.empty_fit()
                empty_axes.append(axis)
                continue
        
            # -------- E fit --------
            fit_E_file = folder / "df_fitE.csv"
        
            if fit_E_file.exists():
                df_E = pd.read_csv(fit_E_file)
                print(fit_E_file)
                print(df_E)
                gradient_summary[axis] = {
                    row["Parameter"]: float(row["df_fit"])
                    for _, row in df_E.iterrows()
        
                }
            else:
                gradient_summary[axis] = general_functions.empty_fit()
        
            # -------- B0 fit --------
            fit_B0_file = folder / "df_fitB0.csv"
        
            if fit_B0_file.exists():
                df_B0 = pd.read_csv(fit_B0_file)
                print(df_B0)
        
                offset_summary[axis] = {
                    row["Parameter"]: float(row["df_fit"])
                    for _, row in df_B0.iterrows()
        
                }
            else:
                empty_axes.append(axis)
                offset_summary[axis] = general_functions.empty_fit()
        
        globals_path = Path(globals_path)
        
        with open(globals_path / "eddy_E_fit.yaml", "w") as f:
            yaml.safe_dump(gradient_summary, f, sort_keys=False, default_flow_style=False)
        
        with open(globals_path / "eddy_B0_fit.yaml", "w") as f:
            yaml.safe_dump(offset_summary, f, sort_keys=False, default_flow_style=False)

        return empty_axes


    
    @staticmethod
    def get_protocol_files(scan_type):
        if scan_type == "fitting_ec":
            save_folder = "Fit_EC"
        else:
            save_folder = general_functions.get_yaml_info(info_name = 'folder_names.yaml', file = GLOBALS_DIR)[scan_type]
        folder_path = Path("protocols") / save_folder
        files = []
        if folder_path.exists():
            files.extend(
                [f.name for f in folder_path.iterdir() if f.is_file()]
            )

        return sorted(files)

            
    @staticmethod        
    def set_3rf(rf, t_90, bw_90, file = None):
        """
        saves the imported amplitude of the 90 pulse (rf), 90 pulse time (t_90) and bandwidth (bw_90) from the pulse calibration  
        into the 3_lobe yaml filein global functions 
        """
        if file == None:
            file = GLOBALS_DIR
 
        with open(f"{file}/3_lobe.yaml", "r") as f:
            rf_file = yaml.safe_load(f)
        
        rf_file["a_90"] = rf
        rf_file["t_90"] = t_90
        rf_file["bw_90"] = bw_90
        
        
        with open(f"{file}/3_lobe.yaml", "w") as f:
            yaml.safe_dump(rf_file, f, sort_keys=False)


    @staticmethod
    def set_5rf(rf, t_90, bw_90, file = None):
        """
        saves the imported amplitude of the 90 pulse (rf), 90 pulse time (t_90) and bandwidth (bw_90) from the pulse calibration  
        into the 5_lobe yaml filein global functions         
        """
        if file == None:
            file = GLOBALS_DIR
 
        with open(f"{file}/5_lobe.yaml", "r") as f:
           rf_file = yaml.safe_load(f)
        
        rf_file["a_90"] = rf
        rf_file["t_90"] = t_90
        rf_file["bw_90"] = bw_90
        
        
        with open(f"{file}/5_lobe.yaml", "w") as f:
            yaml.safe_dump(rf_file, f, sort_keys=False)

    @staticmethod
    def set_B0cal(file = None):
        """
        saves the imported amplitude of the 90 pulse (rf), 90 pulse time (t_90) and bandwidth (bw_90) from the pulse calibration  
        into the 5_lobe yaml filein global functions         
        """
        if file == None:
            file = GLOBALS_DIR
 
        with open(f"{file}/B0_calibration.yaml", "r") as f:
           B0_c = yaml.safe_load(f)
        
        B0_c["m"] = self.B0_cal

        with open(f"{file}/B0_calibration.yaml", "w") as f:
            yaml.safe_dump(B0_c, f, sort_keys=False)

            
    @staticmethod
    def set_SLRrf(rf, t_90, bw_90, file = None):
        """
        saves the imported amplitude of the 90 pulse (rf), 90 pulse time (t_90) and bandwidth (bw_90) from the pulse calibration  
        into the SLR yaml filein global functions         
        """
        if file == None:
            file = GLOBALS_DIR
 
        with open(f"{file}/SLR.yaml", "r") as f:
           rf_file = yaml.safe_load(f)
        
        rf_file["a_90"] = rf
        rf_file["t_90"] = t_90
        rf_file["bw_90"] = bw_90
        
        with open(f"{file}/SLR.yaml", "w") as f:
            yaml.safe_dump(rf_file, f, sort_keys=False)

    @staticmethod
    def change_spars(new_pars, file = None):
        """
        Retrieves the infomation in the shims.yaml to get the current offset values, replaces it with the new 
        offsets and returns this updated dictionary to the yaml file to replace them.

        Inputs:
            new_offsets (list): list of the new offsets that want to be set in the global yaml file. This
                should be a four parameter list for the four output gradient channels on the kahu.
            file (str): name of the filepath to get the the info_name.yaml with the information in 
                (should be stored in a globals file on your kahu workspace)
        """
        if file == None:
            file = GLOBALS_DIR

        
        with open(f"{file}/par_spulse.yaml", "w") as f:
            yaml.safe_dump(new_pars, f, sort_keys=False)
            
    @staticmethod
    def change_offsets(new_offsets, file = None):
        """
        Retrieves the infomation in the shims.yaml to get the current offset values, replaces it with the new 
        offsets and returns this updated dictionary to the yaml file to replace them.

        Inputs:
            new_offsets (list): list of the new offsets that want to be set in the global yaml file. This
                should be a four parameter list for the four output gradient channels on the kahu.
            file (str): name of the filepath to get the the info_name.yaml with the information in 
                (should be stored in a globals file on your kahu workspace)
        """
        if file == None:
            file = GLOBALS_DIR
 
        with open(f"{file}/shims.yaml", "r") as f:
            offsets = yaml.safe_load(f)
        
        offsets["offsets"] = new_offsets
        
        with open(f"{file}/shims.yaml", "w") as f:
            yaml.safe_dump(offsets, f, sort_keys=False)


    @staticmethod
    def get_g_strength(grad_value, axis, file = None):
        """
        gets the gradient strength based on the grad_value input and axis input
        Loads yaml as a dict that can be indexed through the axis - axis needs to be in capitals (X,Y,Z)
        """
        
        if file == None:
            file = GLOBALS_DIR

        with open(f"{file}/g_hzpm_slope.yaml", "r") as f:
            g_dict = yaml.safe_load(f)

        return g_dict[axis]
    

    
    @staticmethod
    def gradient_max(file):
        """
        Uses the maximum gradient calculations from the 1.5T MRI calibration and the max current setting for each 
        to state the maximum values the gradients can be.
    
        Inputs:
            GX (float): gradient value input to pulse sequence in the X direction
            GY (float): gradient value input to pulse sequence in the Y direction
            GZ (float): gradient value input to pulse sequence in the Z direction

        Output:
            (list): list of maximum values that the gradient can operate at in 0-1 terms that kahu recieves
        """
        params = general_functions.get_yaml_info(info_name = "current_cals", file = file)

        return [params["xgrad"], params["ygrad"], params["zgrad"]]
    
    @staticmethod
    def shims_limitation(file):
        """
        Calculates the limitation to the shims from amps to the kahu terms in 0-1 values.
        """
        
        params = general_functions.get_yaml_info(info_name = "current_cals", file = file)

        return [params["xshimgrad"], params["yshimgrad"], params["zshimgrad"]], [params["xshim"], params["yshim"], params["zshim"]]
        
    @staticmethod
    def gradient_boundaries(GX, GY, GZ, shim = "n", file = None):
        """
        Uses the maximum gradient calculations with the shim values that are currently set in the
        globals yaml file from the 1.5T MRI calibration and the max current setting for each
        to raise an error from the notebook/panel that the gradient inputted is too high.
    
        Inputs:
            GX (float): gradient value input to pulse sequence in the X direction
            GY (float): gradient value input to pulse sequence in the Y direction
            GZ (float): gradient value input to pulse sequence in the Z direction
            shim_only (str): if you are just checking the shims as the G inputs then set to yes so it doesnt add ontop
        """
        g_list = general_functions.gradient_max(file)
        GX_lim, GY_lim, GZ_lim = g_list[0], g_list[1], g_list[2]

        if shim == 'y': 
            if np.abs(GX) > GX_lim:
                raise ValueError(f"!!! Shim Gradient X is too high (GX: {GX}) - needs to be less than {GX_lim} !!!")
            
            if np.abs(GY) > GY_lim:
                raise ValueError(f"!!! Shim Gradient Y is too high (GY: {GY}) - needs to be less than {GY_lim} !!!")
        
            if np.abs(GZ) > GZ_lim:
                raise ValueError(f"!!! Shim Gradient Z is too high (GZ: {GZ}) - needs to be less than {GZ_lim} !!!")
    
        
        else:
            shims_dict = general_functions.get_yaml_info(info_name = 'shims', file = file)
            # print(offsets)
            offsets = shims_dict['offsets']
            if np.abs(GX+offsets[0]) > GX_lim:
                raise ValueError(f"!!! Gradient X is too high (GX: {GX} + offset: {offsets[0]} = {GX + offsets[0]}) - needs to be less than {GX_lim} !!!")
            
            if np.abs(GY+offsets[1]) > GY_lim:
                raise ValueError(f"!!! Gradient Y is too high (GY: {GY} + offset: {offsets[1]} = {GY + offsets[1]}) - needs to be less than {GY_lim} !!!")
        
            if np.abs(GZ+offsets[2]) > GZ_lim:
                raise ValueError(f"!!! Gradient Z is too high (GZ: {GZ} + offset: {offsets[2]} = {GZ + offsets[2]}) - needs to be less than {GZ_lim} !!!")
    
    
    def log_msg(self, msg):
        """
        Adds the msg into the log pane in the panel. Does not replace any messages already logged.
    
        Inputs:
            msg (str): message to be contained in the log_pane defined in the panes class
            
        """
        self.panes.log_pane.object += msg + "\n"
    
    def plotlog_msg(self, msg):
        """
        Adds the msg into the log pane in the panel. Does not replace any messages already logged.
    
        Inputs:
            msg (str): message to be contained in the log_pane defined in the panes class
            
        """
        self.panes.plotlog_pane.object += msg + "\n"
    


    def plotstatus_msg(self, msg):
        """
        Replaces any current message in the status_pane with this message.
    
        Inputs:
            msg (str): message to be contained in the status_pane defined in the panes class
        """
        self.panes.plotstatus_pane.object = msg

        

    def status_msg(self, msg):
        """
        Replaces any current message in the status_pane with this message.
    
        Inputs:
            msg (str): message to be contained in the status_pane defined in the panes class
        """
        self.panes.status_pane.object = msg


    def softlog_msg(self, msg):
        """
        Adds the msg into the log pane in the panel. Does not replace any messages already logged.
    
        Inputs:
            msg (str): message to be contained in the log_pane defined in the panes class
            
        """
        self.panes.softlog_pane.object += msg + "\n"
    


    def softstatus_msg(self, msg):
        """
        Replaces any current message in the status_pane with this message.
    
        Inputs:
            msg (str): message to be contained in the status_pane defined in the panes class
        """
        self.panes.softstatus_pane.object = msg
    
    @staticmethod
    def print_dict(dictionary, widget_dict, add_amplitudes = "n"):
        """
        Creates a string to be printed that prints the 

        Made to be applied to the log_pane to record the scan parameters when a pulse seq is run. 

        Inputs:
            dictionary (dict): scan_info_dict that is made from panel_functions.scan_info_maker - dict containing
                the scan_info.
            widget_dict (dict): dictionary of all the widgets that will be made for the panel inputs 
            add_amplitudes (str): tell whether to give the RF pulse amplitudes in the parameters to be printed
                "y" will give it and any other str will not

        Output:
            string (str): string of scan parameters to print in the log_pane
        """
        string = ""
        for k in dictionary.keys():
            if k in widget_dict.keys():
                string = string + f"{widget_dict[k].name}: {dictionary[k]} \n"

        if add_amplitudes == "y":
            string = string + f"{'a_90'}: {dictionary['a_90']} \n" + f"{'a_180'}: {dictionary['a_180']} \n"
            
        return string
