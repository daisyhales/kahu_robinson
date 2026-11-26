from matipo import SEQUENCE_DIR, GLOBALS_DIR
from matipo.sequence import Sequence

from matipo.util.fft import fft_reconstruction
from matipo.util.fft import get_freq_spectrum

import os
import numpy as np
from bokeh.plotting import figure
import panel as pn
import asyncio
import traceback
from panel.io.state import state
# from daisy_wheels.user_interface.general import general_functions

from user_interface.general import general_functions
from user_interface.panes import panes
from user_interface.updating import updating
from user_interface.oneDfunctions import oneDfunctions
from user_interface.panel import panel_functions
from user_interface.run_seq import run_pulse_sequence
from user_interface.layout import panel_layout
from user_interface.threeDfunctions import threeDfunctions
from user_interface.buttons.btn_control import ButtonController
from dashboards_.soft_pulse_tab.pulse_cal import Soft_Pulse_Calibration

from pathlib import Path
from datetime import date, datetime
from panel.io.state import state

file = general_functions.get_yaml_info(info_name = 'file.yaml', file = GLOBALS_DIR)["file"] # where you want the global parameters to be taken from 


p = panel_functions(file = file)
all_pns = panes() # using the () will allow the self parameters to be used
g = general_functions(all_pns)

u = updating(g, all_pns, oneDfunctions, threeDfunctions)
btn_ctrl = ButtonController(g, all_pns, u)
rps = run_pulse_sequence(u, g, all_pns, p, btn_ctrl)
layout = panel_layout(all_pns)

scan_type = "fitting_ec"
save_folder = general_functions.get_yaml_info(info_name = 'folder_names.yaml', file = GLOBALS_DIR)[scan_type] # folder to save data and protocols
dimensions = 1
input_width = 150 

input_list = [
            "measure_data_name",  
            "protocols", "protocol_name",
            "N_tau_E",
            "lam_E",
            "tau_min_E",
            "tau_max_E",
            "N_exp_max_E",
              
            "N_tau_B0",
            "lam_B0",
            "tau_min_B0",
            "tau_max_B0",  
            "N_exp_max_B0",
            ]


button_list = ['fit_Ebtn', 'fit_B0btn', "fit_Ebtn_loop", "fit_B0btn_loop", "abort", "load", 'load_protocol', 'save_protocol', "fit_Ebtn_save", "fit_B0btn_save"]

widget_dict = p.create_parameter_widgets(input_list, file = file, scan_type = scan_type)
all_buttons = p.create_buttons_widgets(button_list)
g.grab_measurement_files(save_folder, widget_dict["measure_data_name"])


fit_E = all_buttons["fit_Ebtn"]
fit_B0 = all_buttons["fit_B0btn"]
fit_E_loop = all_buttons["fit_Ebtn_loop"]
fit_B0_loop = all_buttons["fit_B0btn_loop"]
abort = all_buttons["abort"]
load = all_buttons["load"]
load_protocol = all_buttons["load_protocol"]
save_protocol = all_buttons["save_protocol"]
fit_Ebtn_save = all_buttons["fit_Ebtn_save"]
fit_B0btn_save = all_buttons["fit_B0btn_save"]

load.name = "Load Data to Fit"
load.button_type = "primary"

fit_E.on_click(btn_ctrl.fitting_E_on_click)
fit_B0.on_click(btn_ctrl.fitting_B0_on_click)

fit_E_loop.on_click(btn_ctrl.fitting_E_loop_on_click)
fit_B0_loop.on_click(btn_ctrl.fitting_B0_loop_on_click)
abort.on_click(btn_ctrl.abort_fitting_loop)

load.on_click(btn_ctrl.load_fitting_data_on_click)
load_protocol.on_click(btn_ctrl.apply_protocol_on_click)
save_protocol.on_click(btn_ctrl.save_protocol_on_click)

fit_Ebtn_save.on_click(btn_ctrl.save_fitting_E)
fit_B0btn_save.on_click(btn_ctrl.save_fitting_B)

btn_ctrl.wire(
    run_seq = None,
    widget_dict = widget_dict,
    scan_type = scan_type,
    save_dict = None,
    button_dict = all_buttons,
    dimensions = dimensions,
    file = file,
    save_folder = save_folder,
    fit_Ebtn = fit_E,
    fit_B0btn = fit_B0,
    fit_Ebtn_loop = fit_E_loop,
    fit_B0btn_loop = fit_B0_loop,
    abort_fitting = abort
)



class FitEC:

    def __init__(self):
        fit_E.disabled = True
        fit_B0.disabled = True
        fit_E_loop.disabled = True
        fit_B0_loop.disabled = True
        abort.disabled = True

        self.main_panel = layout.p_layout_fit_ec(widget_dict, fit_E, fit_B0, fit_E_loop, fit_B0_loop, abort, load, load_protocol, save_protocol, fit_Ebtn_save, fit_B0btn_save)

    def __call__(self):
        return self.main_panel
