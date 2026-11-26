# from matipo.sequence import ParDef, floatarray
# from collections import namedtuple
# import numpy as np

from matipo.sequence import ParDef, floatarray
from collections import namedtuple
import numpy as np

from matipo.util.kahu_rf_module import pulse_start, pulse_end
from matipo.util.pulseshape import calc_soft_pulse

from matipo.util.kahu_drive_module import set_shims
from matipo.util.kahu_rf_module import pulse

# from matipo.util.kahu_drive_module import shims_set, pulse_start, pulse_end
# from matipo.util.kahu_rf_module import pulse_start, pulse_end
# from matipo.util.kahu_drive_module import set_shims
from matipo.util.pulseshape import calc_soft_pulse

import logging
logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)

G_ZERO = np.array([0, 0, 0])


from matipo import SEQUENCE_DIR, GLOBALS_DIR
from user_interface.general import general_functions
from user_interface.panes import panes
all_pns = panes() # using the () will allow the self parameters to be used
g = general_functions(all_pns)


file = g.get_yaml_info(info_name = 'file.yaml', file = GLOBALS_DIR)["file"] # where you want the global parameters to be taken from 
params = g.get_yaml_info(info_name = "current_cals", file = file)
G_MAX = [params["xgrad"], params["ygrad"], params["zgrad"]]

def grad_ramp(seq, g_start, g_end, t_ramp, t_grad_spacing = 10e-6):
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
        yield seq.grad.vec(*vector) # seq command that controls the gradient on the kahu
        yield seq.wait(t_grad_spacing)


# Define input parameters (must be named 'PARDEF')
PARDEF = [
    # args: name, dtype, default value
    ParDef('n_scans', int, 1, min=1, unit=''),
    ParDef('f', float, 1e6, unit='Hz'),
    ParDef('a_90', float, 0, min=0, max=1, unit=''),
    ParDef('t_90', float, 1e-3, unit='s'),
    # ParDef('bw_pulse_ex', float, 0, unit='Hz'),
    ParDef('a_180', float, 0, min=0, max=1, unit=''),
    ParDef('t_180', float, 1e-3, unit='s'),
    # ParDef('bw_pulse_se', float, 0, unit='Hz'),
    ParDef('t_dw', float, 1e-6, min=0.1e-6, max=160e-6, unit='s'),
    # ParDef('g_slice_ex', floatarray, (0, 0, 0), min=(-1, -1, -1), max=(1, 1, 1), unit=''),

    ParDef('n_samples', int, 1000, min=2, unit=''),
    ParDef('n_phase1', int, 1, min=1, unit=''),
    ParDef('n_phase2', int, 1, min=1, unit=''),
    ParDef('t_echo', float, 20e-3, unit='s'),
    ParDef('TR', float, 1, unit='s'),
    ParDef('g_read', floatarray, (0, 0, 0), min=(-1, -1, -1), max=(1, 1, 1), unit=''),
    ParDef('g_phaseread', floatarray, (0, 0, 0), min=(-1, -1, -1), max=(1, 1, 1), unit=''),
    ParDef('g_phase1', floatarray, (0, 0, 0), min=(-1, -1, -1), max=(1, 1, 1), unit=''),
    ParDef('g_phase2', floatarray, (0, 0, 0), min=(-1, -1, -1), max=(1, 1, 1), unit=''),
    ParDef('g_crush', floatarray, (0, 0, 0), min=(-1, -1, -1), max=(1, 1, 1), unit=''),
    ParDef('g_spoil', floatarray, (0, 0, 0), min=(-1, -1, -1), max=(1, 1, 1), unit=''),
    ParDef('t_phase', float, 1000e-6, unit='s'),
    ParDef('t_crush', float, 1000e-6, unit='s'),
    ParDef('t_ramp', float, 1000e-6, unit='s'),
    ParDef('t_spoil', float, 1e-3, unit='s'),
    ParDef('rx_chan', int, 0, min=0, max=3, unit=''),
    ParDef('tx_chan', int, 0, min=0, max=1, unit=''),
    ParDef('offsets', floatarray, np.zeros(4), min=-1, max=1, unit=''),


]

# Create parameter set type (must be named 'ParameterSet')
ParameterSet = namedtuple('ParameterSet', [pd.name for pd in PARDEF])

# Main pulse sequence function, taking the sequencer top level module and a parameter set
def main(seq, par: ParameterSet):
    
    g_shim = np.array(par.offsets[:3])
    t_acq = par.n_samples * par.t_dw

    t_ex_se = par.t_echo/2 - (par.t_90/2 + 5 * par.t_ramp + par.t_crush + par.t_180/2)
    log.debug(f"t_ex_se: {t_ex_se}")
    if t_ex_se <= 0:
        raise Exception('Echo time too short.')
    t_se_read = par.t_echo/2 - (par.t_180/2 + 5 * par.t_ramp + par.t_crush + par.t_phase + t_acq/2)
    # t_se_read = par.t_echo/2 - (par.t_180/2 + par.t_ramp + par.t_crush + par.t_ramp + par.t_ramp + par.t_phase + par.t_ramp + par.t_ramp + t_acq/2)
    log.debug(f"t_se_read: {t_se_read}")
    if t_se_read <= 0:
        raise Exception('Echo time too short.')

    t_end = par.TR - (12*par.t_ramp + par.t_90 + t_ex_se + 2*par.t_phase + 2*par.t_crush + par.t_180 + t_se_read + t_acq + par.t_spoil)
    if t_end <= 0:
        raise Exception('t_end too short.')
    
    log.debug(f"t_end: {t_end}")

    
    log.debug(f"t_ex_se: {t_ex_se} t_se_read: {t_se_read} t_end: {t_end}, par.t_ramp: {par.t_ramp}, par.t_90: {par.t_90}, par.t_phase: {par.t_phase}, par.t_crush: {par.t_crush}, par.t_180: {par.t_180}, t_se_read: {t_se_read}, t_acq: {t_acq}, par.t_spoil: {par.t_spoil}")


    
    log.debug(f"g_crush: {par.g_crush}")
    log.debug(f"g_spoil: {par.g_spoil}")
    
    g_phase1_array = np.linspace(-par.g_phase1, par.g_phase1, par.n_phase1, endpoint=False)
    g_phase2_array = np.linspace(-par.g_phase2, par.g_phase2, par.n_phase2, endpoint=False)

    
    n_phasecycle = 8
    phase_cycle_90 = [0, 180, 0, 180, 90, 270, 90, 270]
    phase_cycle_180 = [90, 90, 270, 270, 0, 0, 180, 180]

    
    # choose tx/rx channel
    rx = seq.rx[par.rx_chan]
    # yield rx.reset()
    tx = seq.tx[par.tx_chan]


    # Tx/Rx constant settings
    yield from [tx.freq(par.f) for tx in seq.tx]
    yield rx.freq(par.f)
    yield rx.dwelltime(par.t_dw)
    yield rx.mode(flatfilter=True)
    yield seq.wait(0.01)

    # apply shim
    yield grad_ramp(seq, G_ZERO, g_shim, par.t_ramp)
    yield seq.wait(0.1)

    for i_phase_2, g_phase2_i in enumerate(g_phase2_array):
        for i_phase_1, g_phase1_i in enumerate(g_phase1_array):
            acq_i = i_phase_1 + par.n_phase1*i_phase_2
            g_phase_i = g_phase1_i + g_phase2_i
            for i_scan in range(par.n_scans):
                p_ex = p_acq = phase_cycle_90[i_scan % n_phasecycle]
                p_se = phase_cycle_180[i_scan % n_phasecycle]
        
                # Sliced Excitation Pulse

                yield pulse(seq, tx, p_ex, par.a_90, par.t_90)

                yield seq.wait(t_ex_se)

                # crusher gradient before 180
                yield grad_ramp(seq, g_shim, par.g_crush+g_shim, par.t_ramp)
                yield seq.wait(par.t_crush)
                yield grad_ramp(seq, par.g_crush+g_shim, g_shim, par.t_ramp)
                
                # hard 180 pulse (no slice gradient)
                yield pulse(seq, tx, p_se, par.a_180, par.t_180)
                
                # crusher gradient after 180
                yield grad_ramp(seq, g_shim, par.g_crush+g_shim, par.t_ramp)
                yield seq.wait(par.t_crush)
                yield grad_ramp(seq, par.g_crush+g_shim, g_shim, par.t_ramp)

        
                yield seq.wait(t_se_read)
        
                # Phase wind gradient
                yield grad_ramp(seq, g_shim, par.g_phaseread+g_phase_i+g_shim, par.t_ramp)
                yield seq.wait(par.t_phase)
                yield grad_ramp(seq, par.g_phaseread+g_phase_i+g_shim, g_shim, par.t_ramp)
    
                # Readout
                yield grad_ramp(seq, g_shim, par.g_read+g_shim, par.t_ramp)
                yield rx.phase(p_acq)
                yield rx.acquire(acq_i, par.n_samples)
                yield seq.wait(t_acq)
                yield grad_ramp(seq, par.g_read+g_shim, g_shim, par.t_ramp)
    
                # Phase unwind gradient
                yield grad_ramp(seq, g_shim, par.g_phaseread-g_phase_i+g_shim, par.t_ramp)
                yield seq.wait(par.t_phase)
                yield grad_ramp(seq, par.g_phaseread-g_phase_i+g_shim, g_shim, par.t_ramp)
    
                # Spoiler
                yield grad_ramp(seq, g_shim, par.g_spoil+g_shim, par.t_ramp)
                yield seq.wait(par.t_spoil)
                yield grad_ramp(seq, par.g_spoil+g_shim, g_shim, par.t_ramp)
        
                # Repetition Delay
                yield seq.wait(t_end)
        yield grad_ramp(seq, g_shim, G_ZERO, par.t_ramp)