import panel as pn

pn.extension()
import sys

print(sys.executable)
print(sys.path)

from dashboards_.limits.current_cal import CurrentCal
from dashboards_.soft_pulse_cal.pulse_cal import Soft_Pulse_Calibration
from dashboards_.B0_cal.B0_cal import B0_Calibration


def main(doc=None):
    currentcal = CurrentCal()
    pulse_c = Soft_Pulse_Calibration()
    b0_cal = B0_Calibration()
    
    tabs = pn.Tabs(
        ("Gradient Current Calibration", currentcal()),
        ("Soft Pulse Calibration", pulse_c()),
        ("B0 Calibration", b0_cal()),
    )
    
    return tabs