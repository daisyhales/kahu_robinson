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

pulse_file = general_functions.get_yaml_info(info_name = 'seqnames.yaml', file = GLOBALS_DIR)["2dse"]

file = GLOBALS_DIR # where you want the global parameters to be taken from 
pre_file = general_functions.get_yaml_info(info_name = 'prefiles.yaml', file = GLOBALS_DIR)["2dse"] # where the pulse file lives


p = panel_functions(file = file)
all_pns = panes() # using the () will allow the self parameters to be used
g = general_functions(all_pns)

u = updating(g, all_pns, oneDfunctions, threeDfunctions)
btn_ctrl = ButtonController(g, all_pns, u)
rps = run_pulse_sequence(u, g, all_pns, p, btn_ctrl)
layout = panel_layout(all_pns)

scan_type = "2DSE_multiSLR"
save_folder = general_functions.get_yaml_info(info_name = 'folder_names.yaml', file = GLOBALS_DIR)[scan_type] # folder to save data and protocols
dimensions = 2
input_width = 150 

input_list = ["protocols", "protocol_name",
            "n_scans",   "t_echo", "TR", "t_ramp", "t_crush", "t_spoil","spectral_width", "rx_chan", "tx_chan", 
            
            "soft_pulse_type", 

              
            "select_g_read",  "FOV_read", "resolution_read",
            "select_g_phase1",  "FOV_phase1", "resolution_phase1",
            "select_g_slice",  "slice_thickness", "slice_offset",

            "slice_separation", "num_slices", "autosave"]
file_list = ['filepath']
slice_list = ["gaussian_blur"]
button_list = ['save', 'run', 'abort', 'load_protocol', 'save_protocol']

n = 128
f = 160
change_vals = {"n_scans": 1, "FOV_read": f, "FOV_phase1": f,"spectral_width":50, "slice_thickness": 5,  "resolution_read": n,
               "t_crush": 1, "t_spoil": 1, "t_ramp": 250, "t_echo": 20000,"resolution_phase1": n,"TR": 0.5, "slice_offset": 0}
for k, v in change_vals.items():
    p.update_value_widget(k, v)

widget_dict = p.create_parameter_widgets(input_list, file = file, scan_type = scan_type)
slice_dict = p.create_parameter_slices(slice_list)
save_dict = p.create_parameter_widgets(file_list, file = file)

# ---------------- Set up Buttons ---------------
all_buttons = p.create_buttons_widgets(button_list)

# pn.extension() # initialisation to set the display interaction

# ----------- Call pulse sequence --------------
# uses an async function which will run the contents step by step not all at the same time
async def run_seq_wrapped():
    await rps.run_seq(pulse_file, widget_dict, scan_type, dimensions, slice_dict, add_amplitudes = 'n', file = file, pre_file = pre_file)

run = all_buttons["run"]
save = all_buttons["save"]
abort = all_buttons["abort"]
load_protocol = all_buttons["load_protocol"]
save_protocol = all_buttons["save_protocol"]


btn_ctrl.wire(
    run_seq = run_seq_wrapped,
    widget_dict = widget_dict,
    scan_type = scan_type,
    save_dict = save_dict,
    button_dict = all_buttons,
    dimensions = dimensions,
    file = file,
    save_folder = save_folder
)


run.on_click(btn_ctrl.run_on_click)
abort.on_click(btn_ctrl.abort_on_click)
save.on_click(btn_ctrl.save_on_click)
load_protocol.on_click(btn_ctrl.apply_protocol_on_click)
save_protocol.on_click(btn_ctrl.save_protocol_on_click)

plot = pn.widgets.Button(name="Plot Seq", button_type="warning")
async def plot_tab(event):
    try: 
        rps.plot_seq(pulse_file, widget_dict, scan_type, dimensions, slice_dict, add_amplitudes = 'n', file = file, pre_file = pre_file)
        plot_panel = layout.p_layout_plotseq(all_pns.seq_plot)
        tabs[1] = ("Plot Seq", plot_panel)
    except Exception:
        tb = traceback.format_exc()
        g.plotlog_pane("Error in Plotting:")
        g.plotstatus_msg("Error in Plotting")
        g.plotlog_pane(tb)
        return
    await asyncio.sleep(1)    
    g.log_msg("changing tab now")
    tabs.active = 1

plot.on_click(plot_tab)



button = pn.widgets.Button(name="Calibrate RF Pulse")
async def soft_pulse_tab(event):
    a = btn_ctrl.widget_dict["soft_pulse_type"].value
    new_pars = {
        "soft_pulse_type": btn_ctrl.widget_dict["soft_pulse_type"].value,
        "select_g_slice": btn_ctrl.widget_dict["select_g_slice"].value,
        "slice_thickness": btn_ctrl.widget_dict["slice_thickness"].value,        
        "spectral_width": btn_ctrl.widget_dict["spectral_width"].value,
        "t_ramp": btn_ctrl.widget_dict["t_ramp"].value,
    }
    g.change_spars(new_pars, file = file)
    g.softlog_msg(f"Saved to yaml {new_pars}")

    await asyncio.sleep(1)
    new_rf_panel = Soft_Pulse_Calibration()
    tabs[2] = ("RF Calibration", new_rf_panel)
    
    tabs.active = 2 
    
button.on_click(soft_pulse_tab)

    
class TwoDSE:

    def __init__(self):
     
        self.tabs = None
        plot = pn.widgets.Button(name="Plot Seq", button_type="warning")
        async def plot_tab(event):
            try: 
                g.status_msg("Plotting...")
                
                rps.plot_seq(pulse_file, widget_dict, scan_type, dimensions, slice_dict, add_amplitudes = 'n', file = file, pre_file = pre_file)
                plot_panel = layout.p_layout_plotseq(all_pns.seq_plot)
                self.tabs[1] = ("Plot Seq", plot_panel)
            except Exception:
                tb = traceback.format_exc()
                g.plotlog_pane("Error in Plotting:")
                g.plotstatus_msg("Error in Plotting")
                g.plotlog_pane(tb)
                return
            await asyncio.sleep(1)    
            g.log_msg("changing tab now")
            self.tabs.active = 1

        
        plot.on_click(plot_tab)

        button = pn.widgets.Button(name="Calibrate RF Pulse")
        async def soft_pulse_tab(event):
            a = btn_ctrl.widget_dict["soft_pulse_type"].value
            new_pars = {
                "soft_pulse_type": btn_ctrl.widget_dict["soft_pulse_type"].value,
                "select_g_slice": btn_ctrl.widget_dict["select_g_slice"].value,
                "slice_thickness": btn_ctrl.widget_dict["slice_thickness"].value,        
                "spectral_width": btn_ctrl.widget_dict["spectral_width"].value,
                "t_ramp": btn_ctrl.widget_dict["t_ramp"].value,
            }
            g.change_spars(new_pars, file = file)
            g.softlog_msg(f"Saved to yaml {new_pars}")
        
            await asyncio.sleep(1)
            new_rf_panel = Soft_Pulse_Calibration()()
            if len(self.tabs) <3:
                self.tabs.append(("RF Calibration", new_rf_panel))
            else:
                self.tabs[2] = ("RF Calibration", new_rf_panel)
            
            self.tabs.active = 2 
            
        button.on_click(soft_pulse_tab)
            
        plot_panel = layout.p_layout_plotseq(all_pns.seq_plot)
        main_panel = layout.p_layout_2DSEmultiSLR(widget_dict, save_dict, slice_dict, run, abort, save, load_protocol, save_protocol, button = button, plot = plot, height = 400)

        self.tabs = pn.Tabs(
            ("2DSE", main_panel),
            ("Plot Seq", plot_panel),
            sizing_mode = "stretch_width"
        )


    def __call__(self):
        return self.tabs
