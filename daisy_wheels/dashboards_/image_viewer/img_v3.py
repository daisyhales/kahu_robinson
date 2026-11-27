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
from user_interface.buttons.viewer import ViewerButtons

from pathlib import Path
from datetime import date, datetime
from panel.io.state import state


file = GLOBALS_DIR # where you want the global parameters to be taken from 


p = panel_functions(file = file)
all_pns = panes() # using the () will allow the self parameters to be used
g = general_functions(all_pns)
v_btn_ctrl = ViewerButtons(g, all_pns)
u = updating(g, all_pns, oneDfunctions, threeDfunctions)
btn_ctrl = ButtonController(g, all_pns, u)
rps = run_pulse_sequence(u, g, all_pns, p, btn_ctrl)
layout = panel_layout(all_pns)

scan_type = "img_viewer"
input_width = 150
dimension = 3

input_list3 = ["select_folder3", "select_file3",
               "gaussian_blur"
              ]

file_list = ["fileheader", 'import_kdata']
button_list = ['run_img', 'load']

widget_dict = p.create_parameter_widgets(input_list3, file = file)

# ---------------- Set up Buttons ---------------
all_buttons = p.create_buttons_widgets(button_list)

run_img = all_buttons["run_img"]
load = all_buttons["load"]

run_img.on_click(v_btn_ctrl.img_run_on_click)
load.on_click(v_btn_ctrl.load_files_on_click)

v_btn_ctrl.wire(widget_dict, file, dimension)

class Viewer3:

    def __init__(self):

        v_btn_ctrl.wire(widget_dict, file, dimension)

        self.panel3 = layout.p_layout_img_viewer3(widget_dict, run_img, load, height = 500)

    def __call__(self):
        return self.panel3
