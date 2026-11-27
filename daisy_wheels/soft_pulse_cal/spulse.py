from matipo import SEQUENCE_DIR, GLOBALS_DIR
from matipo.sequence import Sequence

from matipo.util.fft import fft_reconstruction
from matipo.util.fft import get_freq_spectrum

import numpy as np
from bokeh.plotting import figure
import panel as pn
pn.extension() # initialisation to set the display interaction

import asyncio
import traceback
from panel.io.state import state
from pathlib import Path
from datetime import date, datetime
from panel.io.state import state

from daisy_wheels.user_interface.general import general_functions
from daisy_wheels.user_interface.oneDfunctions import oneDfunctions
from daisy_wheels.user_interface.panel import panel_functions
from daisy_wheels.user_interface.panes import panes
from daisy_wheels.user_interface.updating import updating
from daisy_wheels.user_interface.run_seq import run_pulse_sequence
from daisy_wheels.user_interface.panel import panel_layout
from daisy_wheels.user_interface.threeDfunctions import threeDfunctions
from daisy_wheels.user_interface.buttons.btn_control import ButtonController

file = GLOBALS_DIR # where the global parameters are
pre_file = GLOBALS_DIR+"soft_pulse_cal/" # where the pulse file lives

p = panel_functions(file = file)
all_pns = panes() # using the () will allow the self parameters to be used
g = general_functions(all_pns)

u = updating(g, all_pns, oneDfunctions, threeDfunctions)
btn_ctrl = ButtonController(g, all_pns, u)
rps = run_pulse_sequence(u, g, all_pns, p, btn_ctrl)
layout = panel_layout(all_pns)


scan_type = "pulse_cal_SLR"
dimensions = 1
input_width = 150

input_list = ["soft_pulse_type", "a_rf_slider","t_pulse","bw_90",
              "select_g_slice",  "slice_thickness",
              "n_scans", "spectral_width", "n_samples", "t_ramp", 
               ]
file_list = ['filepath']
button_list = ['save', 'run', 'abort', 'toggle_info', 'run_loop', "add_rf"]
pulse_file = 'SLR_cal.py'

change_vals = {"n_samples": 96, "t_ramp": 250, "slice_thickness": 5}
for k, v in change_vals.items():
    p.update_value_widget(k, v)

widget_dict = p.create_parameter_widgets(input_list, scan_type, file = file)
save_dict = p.create_parameter_widgets(file_list, file = file)

# ---------------- Set up Buttons ---------------
all_buttons = p.create_buttons_widgets(button_list)


pn.extension() # initialisation to set the display interaction

async def run_seq_wrapped():
    print("HELLO")
    await rps.run_seq(pulse_file, widget_dict, scan_type, dimensions, add_amplitudes = 'y', file = file, pre_file = pre_file)

async def loop_run_seq_wrapped():
    print("entered")
    try:
        await rps.run_loop(pulse_file, widget_dict, scan_type, dimensions, add_amplitudes = 'y', file = file, pre_file = pre_file)
    except Exception:
        print("NOPE")

run = all_buttons["run"]
save = all_buttons["save"]
abort = all_buttons["abort"]
info = all_buttons["toggle_info"]
run_loop = all_buttons["run_loop"]
add_rf = all_buttons["add_rf"]

btn_ctrl.wire(
    run_seq = run_seq_wrapped,
    loop_run_seq = loop_run_seq_wrapped,
    widget_dict = widget_dict,
    scan_type = scan_type,
    save_dict = save_dict,
    button_dict = all_buttons,
    dimensions= dimensions,
    update_secs = 1,
    file = file
)

run.on_click(btn_ctrl.run_on_click)
abort.on_click(btn_ctrl.abort_on_click)
save.on_click(btn_ctrl.save_on_click)
info.on_click(btn_ctrl.toggle_info)
run_loop.on_click(btn_ctrl.run_loop_on_click)
add_rf.on_click(btn_ctrl.apply_soft_rf_on_click)

def get_spulse_panel():
    
    dictionary = g.get_yaml_info(info_name = "par_spulse", file = file)
    for k,v in dictionary.items():
        widget_dict[k].value = v
         
    panel = layout.p_layout_pulse_calSLR(widget_dict, save_dict, run, abort, save, info, run_loop, add_rf, height = 400)
    return panel
















