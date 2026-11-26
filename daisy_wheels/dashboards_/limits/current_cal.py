import panel as pn
import yaml
from matipo import SEQUENCE_DIR, GLOBALS_DIR
from pathlib import Path
pn.extension()




class CurrentCal:

    def __init__(self):

        info_name = "current_cals"
        self.file = Path(GLOBALS_DIR) / f"{info_name}.yaml"
        
        with open(self.file, "r") as f:
            vals = yaml.safe_load(f)
        w = 100
        step = 1
        
        self.gx_lim = pn.widgets.FloatInput(value=vals["xlim"], disabled=True, width=w, step = step)
        self.gy_lim = pn.widgets.FloatInput(value=vals["ylim"], disabled=True, width=w, step = step)
        self.gz_lim = pn.widgets.FloatInput(value=vals["zlim"], disabled=True, width=w, step = step)
        self.b0_lim = pn.widgets.FloatInput(value=vals["b0lim"], disabled=True, width=w, step = step)
        
        self.gx = pn.widgets.FloatInput(value=vals["xmax"], disabled=True, width=w, step = step)
        self.gy = pn.widgets.FloatInput(value=vals["ymax"], disabled=True, width=w, step = step)
        self.gz = pn.widgets.FloatInput(value=vals["zmax"], disabled=True, width=w, step = step)
        self.b0 = pn.widgets.FloatInput(value=vals["b0max"], disabled=True, width=w, step = step)


        self.xshim = pn.widgets.FloatInput(value=vals["xshim"], disabled=True, width=w, step = step)
        self.yshim = pn.widgets.FloatInput(value=vals["yshim"], disabled=True, width=w, step = step)
        self.zshim = pn.widgets.FloatInput(value=vals["zshim"], disabled=True, width=w, step = step)

        self.status = pn.pane.Markdown("")
        self.info = pn.pane.Markdown("These are the current measurements for each gradient axis corresponding to 1 DAC unit input - used for calibration of the DAC.")

        self.edit_button = pn.widgets.Button(name="Unlock Calibrations", button_type="warning",)
        self.shim_button = pn.widgets.Button(name="Unlock Shims", button_type="danger",)
        self.save_button = pn.widgets.Button(name="Save", button_type="primary",)

        self.edit_button.on_click(self.toggle_edit)
        self.shim_button.on_click(self.toggle_shims)
        self.save_button.on_click(self.save)

    def toggle_edit(self, event):
    
        if self.gx.disabled:
            # currently locked -> unlock
            self.gx.disabled = False
            self.gy.disabled = False
            self.gz.disabled = False
            self.b0.disabled = False
            self.gx_lim.disabled = False
            self.gy_lim.disabled = False
            self.gz_lim.disabled = False
            self.b0_lim.disabled = False
            self.edit_button.name = "Lock Calibrations"
    
        else:
            # currently unlocked -> lock
            self.gx.disabled = True
            self.gy.disabled = True
            self.gz.disabled = True
            self.b0.disabled = True
            self.gx_lim.disabled = True
            self.gy_lim.disabled = True
            self.gz_lim.disabled = True
            self.b0_lim.disabled = True
            self.edit_button.name = "Unlock Calibrations"


    def toggle_shims(self, event):
    
        if self.xshim.disabled:
            # currently locked -> unlock
            self.xshim.disabled = False
            self.yshim.disabled = False
            self.zshim.disabled = False
            self.shim_button.name = "Lock Shims"
    
        else:
            # currently unlocked -> lock
            self.xshim.disabled = True
            self.yshim.disabled = True
            self.zshim.disabled = True
            self.shim_button.name = "Unlock Shims"

    def save(self, event):

        vals = {
            "xlim": self.gx_lim.value,
            "ylim": self.gy_lim.value,
            "zlim": self.gz_lim.value,
            "b0lim": self.b0_lim.value,
            "xmax": self.gx.value,
            "ymax": self.gy.value,
            "zmax": self.gz.value,
            "b0max": self.b0.value,
            "xgrad": self.gx_lim.value/ self.gx.value,
            "ygrad": self.gy_lim.value/ self.gy.value,
            "zgrad": self.gz_lim.value/ self.gz.value,
            "b0grad": self.b0_lim.value/ self.b0.value,
            "xshim": self.xshim.value,
            "xshimgrad": self.xshim.value/ self.gx.value,
            "yshim": self.yshim.value,
            "yshimgrad": self.yshim.value/ self.gy.value,
            "zshim": self.zshim.value,
            "zshimgrad": self.zshim.value/ self.gz.value,
        }
                
        with open(self.file, "w") as f:
            yaml.safe_dump(vals, f)
        
        self.status.object = (f"Saved to Globals: X={vals['xgrad']:.2f} DAC, " f"Y={vals['ygrad']:.2f} DAC, " f"Z={vals['zgrad']:.2f} DAC, " f"B0={vals['b0grad']:.2f} DAC\n X Shim={vals['xshimgrad']:.2f} DAC, " f"Y Shim={vals['yshimgrad']:.2f} DAC, " f"Z Shim={vals['zshimgrad']:.2f} DAC,")

    def __call__(self):

        return pn.Column(
            "### Gradient Current Calibration Values",
            self.info,
            
            pn.GridBox(
                pn.pane.Markdown("**Gradient Axis**"),
                pn.pane.Markdown("**Shim Current Limit**"),
                pn.pane.Markdown("**Max Current Limit**"),
                pn.pane.Markdown("**Current at 1 DAC Output**"),
                
                pn.pane.Markdown("X (A)"),
                self.xshim,
                self.gx_lim,
                self.gx,
                
                pn.pane.Markdown("Y (A)"),
                self.yshim,
                self.gy_lim,
                self.gy,
                
                pn.pane.Markdown("Z (A)"),
                self.zshim,
                self.gz_lim,
                self.gz, 

                pn.pane.Markdown("B0 (A)"),
                pn.pane.Markdown(""),
                self.b0_lim,
                self.b0, ncols=4, sizing_mode="stretch_width",),
            
            pn.Row(self.shim_button, self.edit_button, self.save_button),
            self.status,
            )
