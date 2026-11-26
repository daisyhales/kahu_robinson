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


from pathlib import Path
from datetime import date, datetime
from panel.io.state import state

pulse_file = general_functions.get_yaml_info(info_name = 'seqnames.yaml', file = GLOBALS_DIR)["measure_ec"] 

file = general_functions.get_yaml_info(info_name = 'file.yaml', file = GLOBALS_DIR)["file"] # where you want the global parameters to be taken from 
pre_file = general_functions.get_yaml_info(info_name = 'prefiles.yaml', file = GLOBALS_DIR)["measure_ec"] # where the pulse file lives

p = panel_functions(file = file)
all_pns = panes() # using the () will allow the self parameters to be used
g = general_functions(all_pns)

u = updating(g, all_pns, oneDfunctions, threeDfunctions)
btn_ctrl = ButtonController(g, all_pns, u)
rps = run_pulse_sequence(u, g, all_pns, p, btn_ctrl)
layout = panel_layout(all_pns)

scan_type = "compile_ec"
save_folder = general_functions.get_yaml_info(info_name = 'folder_names.yaml', file = GLOBALS_DIR)[scan_type] # folder to save data and protocols
save_folder_measure = general_functions.get_yaml_info(info_name = 'folder_names.yaml', file = GLOBALS_DIR)["measure_ec"] # folder to save data and protocols
dimensions = 1
input_width = 150 

input_list = ["protocols", "protocol_name",
            "probe_axis",
              
            "x_e_factor",
            "y_e_factor",
            "z_e_factor",
              
            "x_b0_factor",
            "y_b0_factor",
            "z_b0_factor", 
              
            "n_scans",
            "flip_angle",
            "n_pulses",
            
            "spectral_width",
            "apodisation",
              
            "t_acq_delay",
            "t_grad",
            "t_ramp",
            "t_wait",
            
            "t_end",
            "t_next",
                          
            "select_g", "g_val", "resolution", 
            
            "tx_chan",
            "rx_l", "rx_c", "rx_r",
            
]

file_list = ['filepath']

button_list = ['save', 'abort', 'load_protocol', 'save_protocol', "run_nope", "run_pe"]

# n = 256
# f = 220

# file_start = r"C:\Users\halesda\OneDrive - Victoria University of Wellington - STAFF\Desktop\kahu-workspace\notebooks\daisy\kahu_eddy"
change_vals = {"spectral_width": 100, 
                "resolution": 100, 
                "t_acq_delay": 100, 
                "t_grad": 10.0, 
                "apodisation": 100,
                "t_end": 10.0, 
              }
for k, v in change_vals.items():
    p.update_value_widget(k, v)
    


widget_dict = p.create_parameter_widgets(input_list, file = file, scan_type = scan_type)

save_dict = p.create_parameter_widgets(file_list, file = file)

# ---------------- Set up Buttons ---------------
all_buttons = p.create_buttons_widgets(button_list)

# pn.extension() # initialisation to set the display interaction

# ----------- Call pulse sequence --------------
# uses an async function which will run the contents step by step not all at the same time
async def run_seq_wrapped():
    await rps.run_seq(pulse_file, widget_dict, scan_type, dimensions, None, add_amplitudes = 'n', file = file, pre_file = pre_file)

run_nope = all_buttons["run_nope"]
run_pe = all_buttons["run_pe"]
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

run_nope.on_click(btn_ctrl.run_no_pe_on_click)
run_pe.on_click(btn_ctrl.run_pe_on_click)
abort.on_click(btn_ctrl.abort_on_click)
save.on_click(btn_ctrl.save_on_click)
load_protocol.on_click(btn_ctrl.apply_protocol_on_click)
save_protocol.on_click(btn_ctrl.save_protocol_on_click)

plot = pn.widgets.Button(name="Plot Seq", button_type="warning")
async def plot_tab(event):
    try: 
        rps.plot_seq(pulse_file, widget_dict, scan_type, dimensions, None, add_amplitudes = 'n', file = file, pre_file = pre_file)
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



    
class CompileFit:

    def __init__(self):
     
        self.tabs = None
        plot = pn.widgets.Button(name="Plot Seq", button_type="warning")
        async def plot_tab(event):
            try: 
                g.status_msg("Plotting...")
                rps.plot_seq(pulse_file, widget_dict, scan_type, dimensions, None, add_amplitudes = 'n', file = file, pre_file = pre_file)
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
        try: 
            empty_axes = g.compile_fitting_pars(save_folder_measure, GLOBALS_DIR)
            g.log_msg(f"Saved into globals - eddy_E_fit and eddy_B0_fit")
            for ax in empty_axes:
                g.log_msg(f"{ax} Gradient and {ax} Eddy has not been fitted/measured.")

        except Exception:
            tb = traceback.format_exc()
            g.log_msg("Error compiling parameters:")
            g.status_msg("Error Compiling Parameters From Fitting")
            g.log_msg(tb)
            
        plot_panel = layout.p_layout_plotseq(all_pns.seq_plot)
        main_panel = layout.p_layout_compile_fit(widget_dict, save_dict, run_nope, run_pe, abort, save, load_protocol, save_protocol, plot)


        self.tabs = pn.Tabs(
            ("Compile and Compare", main_panel),
            ("Plot Seq", plot_panel),
            sizing_mode = "stretch_width"
        )


    def __call__(self):
        return self.tabs
