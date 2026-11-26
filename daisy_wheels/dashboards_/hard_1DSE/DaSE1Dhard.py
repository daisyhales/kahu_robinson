from matipo.sequence import ParDef, floatarray
from collections import namedtuple
import numpy as np
from matipo import SEQUENCE_DIR, GLOBALS_DIR

from matipo.util.kahu_rf_module import pulse_start, pulse_end
from matipo.util.pulseshape import calc_soft_pulse

from matipo.util.kahu_drive_module import set_shims
from matipo.util.kahu_rf_module import pulse

from user_interface.general import general_functions
from user_interface.panes import panes

all_pns = panes() # using the () will allow the self parameters to be used
g = general_functions(all_pns)

import logging
logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)

G_ZERO = np.array([0, 0, 0])

file = g.get_yaml_info(info_name = 'file.yaml', file = GLOBALS_DIR)["file"] # where you want the global parameters to be taken from 
params = g.get_yaml_info(info_name = "current_cals", file = file)
G_MAX = [params["xgrad"], params["ygrad"], params["zgrad"]]

def grad_ramp(seq, grad, g_start, g_end, t_ramp, t_grad_spacing):
    """
    Initially checks that the input gradients are the right size - will not put too much current in the coil.

    
    """
    if np.any(np.abs(g_start) > G_MAX): 
        raise ValueError(f"gradient in use is greater than G_MAX: {g_start} > {G_MAX}")
    if np.any(np.abs(g_end) > G_MAX): 
        raise ValueError(f"gradient in use is greater than G_MAX: {g_start} > {G_MAX}")
        
    n_grad_ramp = int(t_ramp / t_grad_spacing) # number of steps in the ramp up
    mag_vector = (g_end - g_start) / n_grad_ramp # magnitude vector of the gradient that is ramped
    
    for n in range(1, n_grad_ramp + 1):
        vector = g_start + (mag_vector * n)
        # print(n, vector)
        yield grad.vec(*vector) # seq command that controls the gradient on the kahu
        yield seq.wait(t_grad_spacing)



# Define input parameters (must be named 'PARDEF')
PARDEF = [
    # args: name, dtype, default value
    ParDef('n_scans', int, 1, min=1, unit=''),
    ParDef('f', float, 1e6, unit='Hz'),
    ParDef('a_90', float, 0, min=0, max=1, unit=''),
    ParDef('t_90', float, 32e-6, unit='s'),
    ParDef('a_180', float, 0, min=0, max=1, unit=''),
    ParDef('t_180', float, 32e-6, unit='s'),
    ParDef('t_dw', float, 1e-6, min=0.1e-6, max=160e-6, unit='s'),
    ParDef('n_samples', int, 1000, min=2, unit=''),
    ParDef('t_echo', float, 2000e-6, unit='s'),
    ParDef('TR', float, 1, unit='s'),
    ParDef('g_read', floatarray, (0,0,0), max=(1,1,1), unit=''),
    ParDef('t_grad_spacing', float, 10e-6, unit = 's'),
    ParDef('t_ramp', float, 1000e-6, unit='s'),
    ParDef('offsets', floatarray, np.zeros(4), min=-1, max=1, unit=''),
    ParDef('shims', floatarray, np.zeros(16), min=-1, max=1, unit=''),
    ParDef('tx_chan', int, 0, min=0, max=1, unit=''),
    ParDef('rx_chan', int, 0, min=0, max=3, unit=''),
]


# Create parameter set type (must be named 'ParameterSet')
ParameterSet = namedtuple('ParameterSet', [pd.name for pd in PARDEF])




def main(seq, par: ParameterSet):
    """1D spin echo pulse sequence using the read gradient.
    
    1. Calculates the parameters for the sequence initially: soft pulse arrays for the 180 and 90 and then the times (acquire time,
    preread time, preread180 time) - these are found from the input times and the layout of the 1D spin echo

    2. Define the RF phase cycles and initialise the 
    """

    G_SHIM = np.array(par.offsets[:3])

    # define unknown time parameters
    t_acq = par.n_samples * par.t_dw # time to acquire
    t_preread = (t_acq - par.t_ramp)/2# time when read gradient is on before acquisition read gradient is on - ramp time on either side
    t_preread_180 = (par.t_echo - par.t_90 - par.t_180)/2 - t_preread - 2*par.t_ramp # time between preread gradient off and 180 pulse
    if t_preread_180 < 0:
        raise Exception("Echo time too short.")
    t_180_acq = par.t_echo/2 - par.t_180/2 - t_acq/2 - par.t_ramp # time after the 180 pulse and before the acquisition (- par.t_ramp) used 
    # for ramping and then acquiring
    if t_180_acq < 0:
        raise Exception("Echo time too short.")

    t_end = par.TR - 4*par.t_ramp - par.t_90 - par.t_180 - t_preread - t_preread_180 - t_180_acq - t_acq

    
    n_phase_cycle = 8
    phase_cycle_90 = [0, 180, 0, 180, 90, 270, 90, 270]
    phase_cycle_180 = [90, 90, 270, 270, 0, 0, 180, 180]
    
    # choose tx/rx channel
    tx = seq.tx[par.tx_chan]
    rx = seq.rx[par.rx_chan]
    grad = seq.grad[0] # first parameter is the gradient and the second is the aux value - sets access to the gradient controll
    yield grad.vec(*G_SHIM) # set to start at shimming gradients
    
    # set gradient offsets and shim channel values
    # yield from set_shims(seq, par.offsets, par.shims)
    
    # Tx/Rx constant settings
    yield tx.freq(par.f)
    yield rx.freq(par.f)
    yield rx.dwelltime(par.t_dw)
    yield rx.mode(flatfilter=True)
    yield seq.wait(0.01)

    
    for i_scan in range(par.n_scans):
        p_90 = p_acq = phase_cycle_90[i_scan % n_phase_cycle]
        p_180 = phase_cycle_180[i_scan % n_phase_cycle]

        # 90 deg hard pulse
        yield pulse(seq, tx, p_90, par.a_90, par.t_90)
        
        # ramp the preread gradient and hold for t_preread then ramp down
        yield from grad_ramp(seq, grad, g_start = G_SHIM, g_end = par.g_read + G_SHIM, t_ramp = par.t_ramp, t_grad_spacing = par.t_grad_spacing)

        yield seq.wait(t_preread)

        yield from grad_ramp(seq, grad, g_start = par.g_read + G_SHIM, g_end = G_SHIM, t_ramp = par.t_ramp, t_grad_spacing = par.t_grad_spacing)
        

        yield seq.wait(t_preread_180)

        # 180 degree hard pulse (refocussing)
        yield pulse(seq, tx, p_180, par.a_180, par.t_180)
        
        yield seq.wait(t_180_acq) # wait for the spins to realign more after refocussing

        # ramp read gradient then acquire
        yield from grad_ramp(seq, grad, g_start = G_SHIM, g_end = par.g_read + G_SHIM, t_ramp = par.t_ramp, t_grad_spacing = par.t_grad_spacing)
        yield rx.phase(p_acq)
        yield rx.acquire(0, par.n_samples)
        yield seq.wait(t_acq)
        yield from grad_ramp(seq, grad, g_start = par.g_read + G_SHIM, g_end = G_SHIM, t_ramp = par.t_ramp, t_grad_spacing = par.t_grad_spacing)

        # repetition delay
        yield seq.wait(t_end)












