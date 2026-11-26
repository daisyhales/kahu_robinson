import panel as pn
import yaml
from matipo import SEQUENCE_DIR, GLOBALS_DIR
from pathlib import Path

from matipo.sequence import Sequence
from matipo.util.autophase import autophase
from matipo.util.decimation import decimate
from matipo.util.fft import fft_reconstruction
from matipo.util.fft import get_freq_spectrum
from matipo.util.etl import deinterlace
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation
import asyncio

# set matplotlib figure size, default is quite small
plt.rcParams['figure.figsize'] = [10, 5]
plt.rcParams['figure.dpi'] = 100
        
pn.extension()




class B0Cal:

    def __init__(self):

        info_name = "shims"
        self.file = Path(GLOBALS_DIR) / f"{info_name}.yaml"
        
        with open(self.file, "r") as f:
            vals = yaml.safe_load(f)
    # define widgets and buttons



    def __call__(self):

        return pn.Column(
            "### Gradient Current Calibration Values",
            self.info,
            
            pn.GridBox(
                pn.pane.Markdown("**Gradient Axis**"),
                pn.pane.Markdown("**Max Current Limit**"),
                pn.pane.Markdown("**Current at 1 DAC Output**"),
                
                pn.pane.Markdown("X (A)"),
                self.gx_lim,
                self.gx,
                
                pn.pane.Markdown("Y (A)"),
                self.gy_lim,
                self.gy,
                
                pn.pane.Markdown("Z (A)"),
                self.gz_lim,
                self.gz, 

                pn.pane.Markdown("B0 (A)"),
                self.b0_lim,
                self.b0, ncols=3, sizing_mode="stretch_width",),
            
            pn.Row(self.edit_button, self.save_button),
            self.status,
            )
