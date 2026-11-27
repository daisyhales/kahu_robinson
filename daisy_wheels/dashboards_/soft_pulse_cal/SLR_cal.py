from matipo.sequence import ParDef, floatarray
from collections import namedtuple
import numpy as np

from matipo.util.kahu_rf_module import pulse_start, pulse_end
from matipo.util.pulseshape import calc_soft_pulse

from matipo.util.kahu_drive_module import set_shims
from matipo.util.kahu_rf_module import pulse
from matipo import SEQUENCE_DIR, GLOBALS_DIR

from user_interface.general import general_functions
from user_interface.panes import panes

all_pns = panes() # using the () will allow the self parameters to be used
g = general_functions(all_pns)


import logging
logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)


G_ZERO = np.array([0, 0, 0])

file = GLOBALS_DIR # where you want the global parameters to be taken from 
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

        yield grad.vec(*vector) # seq command that controls the gradient on the kahu
        yield seq.wait(t_grad_spacing)



def soft_pulse(seq, tx, phase, amp, width, bandwidth):
    N, dt, pts = calc_soft_pulse(width, bandwidth)
    pts *= amp
    cmds = [pulse_start(seq, tx, phase, pts[0])]
    cmds.append(seq.wait(dt))
    cmds += [
        seq.join([
            tx.amp(amp),
            seq.wait(dt)])
        for amp in pts[1:]]
    cmds += [pulse_end(seq, tx)]
    return seq.join(cmds)


def lobe_3_3kHz(N):
    return np.sinc((N - 333) * 2 / 333)
     

def lobe_5_3kHz(N):
    return np.sinc((N - 500) * 3 / 500)

def make_soft_pulse(pulse_type, t_90, amplitude, decim=20):

    if pulse_type == "3-lobe sinc":
        N = 666
        y = lobe_3_3kHz(np.arange(0, N+1))
    elif pulse_type == "5-lobe sinc":
        N = 1000
        y = lobe_5_3kHz(np.arange(0, N+1))

    #  reduce resolution (CRITICAL)
    y = y[::decim]

    # recompute time axis
    pulse_time = np.linspace(0, t_90, len(y))

    # normalize + scale
    y = y / np.sum(y)
    y *= amplitude

    return pulse_time, y

def run_soft_pulse(seq, tx, phase, t, y):

    yield pulse_start(seq, tx, phase, y[0])

    for dt, amp in zip(np.diff(t), y[1:]):
        yield tx.amp(amp)
        yield seq.wait(dt)

    yield pulse_end(seq, tx)

# Define input parameters (must be named 'PARDEF')
PARDEF = [
    # args: name, dtype, default value
    ParDef('n_scans', int, 1, min=1, unit=''),
    ParDef('f', float, 1e6, unit='Hz'),
    ParDef('t_dw', float, 1e-6, min=0.1e-6, max=160e-6, unit='s'),
    ParDef('t_ramp', float, 1000e-6, unit='s'),
    # ParDef('soft_pulse_type', str, "3-lobe sinc", unit=''),
    ParDef('n_samples', int, 1000, min=2, unit=''),
    ParDef('g_slice_select', floatarray, (0,0,0), max=(1,1,1), unit=''),
    ParDef('t_grad_spacing', float, 10e-6, unit = 's'),
    # ParDef('t_end', float, 1, unit='s'),
    ParDef('offsets', floatarray, np.zeros(4), min=-1, max=1, unit=''),
    ParDef('soft_pulse_type', str, "SLR", unit=''),

    ParDef('a_90', float, 0, min=0, max=1, unit=''),
    ParDef('t_90', float, 0, min=0, max=1, unit=''),
    ParDef('bw_90', float, 0, min=0, max=1, unit=''),

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



    
    g_shim = np.array(par.offsets[:3])

    t_acq = par.n_samples * par.t_dw # time to acquire
    tacq_2 = t_acq/2

    t_filter = par.t_dw / 10
    t_safety = 100e-6
    
    templ1 = par.t_90 + t_safety
    templ2 = tacq_2 + par.t_90/2 + t_filter + t_safety
    
    templ2 = max(templ1, templ2)
    

    tau = (np.floor(templ2 * 1e3) + 1) * 1e-3
    tau_2 = tau + 1e-3


    tau_ext_a = tau - par.t_90 - t_safety
    tau_ext_b = tau - tacq_2 - par.t_90/2
    tau_ext_c = tau_ext_b - t_filter - t_safety


    # tradd = (4*tau) + tau_2 + (par.t_90/2000) + (tacq_2/1000)
    log.debug(f" t_filter: {t_filter}, templ1: {templ1} , templ2: {templ2} ,templ2: {templ2} , tau: {tau} , tau_2: {tau_2}, tau_ext_a: {tau_ext_a} ,tau_ext_b: {tau_ext_b} , tau_ext_c: {tau_ext_c}")

    # pulse_time, pulse_shape = make_soft_pulse(par.soft_pulse_type, par.t_90, par.a_rf)
    if par.soft_pulse_type in ["3-lobe sinc", "5-lobe sinc"]:
        log.debug("SINC !!!")
        pulse_time, y = make_soft_pulse(par.soft_pulse_type, par.t_90, par.a_90, decim=20)
    else:
        log.debug("SLR !!!")
    log.debug("DEBUG available")
    n_phase_cycle = 8
    phase_cycle_90 = [0, 180, 0, 180, 90, 270, 90, 270]
    
    # choose tx/rx channel
    tx = seq.tx[par.tx_chan]
    rx = seq.rx[par.rx_chan]
    grad = seq.grad[0] # first parameter is the gradient and the second is the aux value - sets access to the gradient controll
    yield grad.vec(*g_shim) # set to start at shimming gradients
    
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

                # # ramp the preread gradient and hold for t_preread then ramp down
        yield from grad_ramp(seq, grad, g_start = g_shim, g_end = par.g_slice_select, t_ramp = par.t_ramp, t_grad_spacing = par.t_grad_spacing)
        if par.soft_pulse_type in ["3-lobe sinc", "5-lobe sinc"]:
            yield from run_soft_pulse(seq, tx, p_90, pulse_time, y)
        if par.soft_pulse_type == "SLR":
            yield soft_pulse(seq, tx, p_90, par.a_90, par.t_90, par.bw_90)
        


        yield seq.wait(tau_ext_a)

        if par.soft_pulse_type in ["3-lobe sinc", "5-lobe sinc"]:
            yield from run_soft_pulse(seq, tx, p_90, pulse_time, y)
        if par.soft_pulse_type == "SLR":
            yield soft_pulse(seq, tx, p_90, par.a_90, par.t_90, par.bw_90)
        


        yield seq.wait(tau_ext_b)
        
        # start acquisition - SPIN ECHO
        yield rx.phase(p_acq)
        yield rx.acquire(0, par.n_samples)
        yield seq.wait(t_acq)
        
        yield seq.wait(tau_ext_c)
        
        yield seq.wait(tau_2)

        if par.soft_pulse_type in ["3-lobe sinc", "5-lobe sinc"]:
            yield from run_soft_pulse(seq, tx, p_90, pulse_time, y)
        if par.soft_pulse_type == "SLR":
            yield soft_pulse(seq, tx, p_90, par.a_90, par.t_90, par.bw_90)
        
        
        yield seq.wait(tau_ext_b)

        # start acquisition - SIMULATED ECHO
        yield rx.phase(p_acq)
        yield rx.acquire(par.n_samples, par.n_samples)
        yield seq.wait(t_acq)



        yield from grad_ramp(seq, grad, g_start = par.g_slice_select, g_end = g_shim, t_ramp = par.t_ramp, t_grad_spacing = par.t_grad_spacing)
        
        yield seq.wait(par.t_ramp)
 