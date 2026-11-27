from matipo.sequence import ParDef, floatarray
from collections import namedtuple
import numpy as np

from matipo.util.kahu_rf_module import pulse_start, pulse_end
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


file = GLOBALS_DIR # where you want the global parameters to be taken from 
params = g.get_yaml_info(info_name = "current_cals", file = file)
G_MAX = [params["xgrad"], params["ygrad"], params["zgrad"]]

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

def grad_ramp(seq, g_start, g_end, t):
    # check gradient amplitudes

    if np.any(np.abs(g_start) > G_MAX):
        raise Exception(f'Gradient Too Large! |g_start|={g_start}>{G_MAX}')
    if np.any(np.abs(g_end) > G_MAX):
        raise Exception(f'Gradient Too Large! |g_end|={g_end}>{G_MAX}')
    
    T_STEP_MIN = 10e-6
    n = int(round(t/T_STEP_MIN-0.4999)) # equivalent to floor with some tolerance (0.0001) for float rounding errors
    t_step = t/n
    return seq.join([
        seq.join([seq.grad[0].vec(*v), seq.wait(t_step)])
        for v in np.linspace(g_start, g_end, n)
    ])

# Define input parameters (must be named 'PARDEF')
PARDEF = [
    # args: name, dtype, default value
    ParDef('n_scans', int, 1, min=1, unit=''),#
    ParDef('f', float, 1e6, unit='Hz'),#
    ParDef('a_90', float, 0, min=0, max=1, unit=''),#
    ParDef('t_90', float, 1e-3, unit='s'),#
    # ParDef('bw_pulse_ex', float, 0, unit='Hz'),
    ParDef('a_180', float, 0, min=0, max=1, unit=''),#
    ParDef('t_180', float, 1e-3, unit='s'),#
    ParDef('soft_pulse_type', str, "3-lobe sinc", unit=''),

    ParDef('t_dw', float, 1e-6, min=0.1e-6, max=160e-6, unit='s'),#
    ParDef('n_samples', int, 1000, min=2, unit=''),#
    ParDef('n_phase', int, 1, min=1, unit=''),#
    ParDef('t_echo', float, 20e-3, unit='s'),#
    # ParDef('t_phase', float, 20e-3, unit='s'),#
    # ParDef('t_end', float, 1, unit='s'),#
    ParDef("TR", float, 1, unit='s'),
    ParDef('bw_90', float, 3000, unit='Hz'),
    

    ParDef('offsets', floatarray, np.zeros(4), min=-1, max=1, unit=''),
    ParDef('f_slice_offset', float, 0, unit='Hz'),
    # ParDef('g_shim', floatarray, (0, 0, 0), min=(-1, -1, -1), max=(1, 1, 1), unit=''),
    ParDef('rx_chan_list', list, [0], unit=''),
    
    ParDef('g_slice', floatarray, (0, 0, 0), min=(-1, -1, -1), max=(1, 1, 1), unit=''),
    # ParDef('g_slice_se', floatarray, (0, 0, 0), min=(-1, -1, -1), max=(1, 1, 1), unit=''),
    ParDef('g_read', floatarray, (0, 0, 0), min=(-1, -1, -1), max=(1, 1, 1), unit=''),#
    # ParDef('g_phaseread', floatarray, (0, 0, 0), min=(-1, -1, -1), max=(1, 1, 1), unit=''),
    ParDef('g_phase', floatarray, (0, 0, 0), min=(-1, -1, -1), max=(1, 1, 1), unit=''),
    ParDef('t_phase', float, 1000e-6, unit='s'),
    ParDef('t_crush', float, 1000e-6, unit='s'),
    ParDef('t_ramp', float, 1000e-6, unit='s'),#
    ParDef('t_spoil', float, 1e-3, unit='s'),
    ParDef('rx_chan', int, 0, min=0, max=3, unit=''),#
    ParDef('tx_chan', int, 0, min=0, max=1, unit=''),
    ParDef('t_unslice_ex_adjust', float, 0e-3, unit='s'),
]

# Create parameter set type (must be named 'ParameterSet')
ParameterSet = namedtuple('ParameterSet', [pd.name for pd in PARDEF])

# Main pulse sequence function, taking the sequencer top level module and a parameter set
def main(seq, par: ParameterSet):
    t_acq = par.n_samples * par.t_dw

    te_ms = ((par.t_echo/2) - ((par.t_90/2) + (4*par.t_ramp) + (par.t_180/2) + par.t_phase + par.t_crush))/2
    te_ms_2 = ((par.t_echo/2) - ((par.t_180/2) + (2*par.t_ramp) + (t_acq/2) + par.t_crush))
    if te_ms <= 0:
        raise Exception('increase TE: te_ms too short')
    if te_ms_2 <= 0:
        raise Exception('increase TE: te_ms_2 too short')

    t_end = par.TR - (8*par.t_ramp + par.t_90 + 2 * te_ms + par.t_phase + 2 * par.t_crush + par.t_180 + te_ms_2 + t_acq)
    if t_end <= 0:
        raise Exception('increase TR: t_end too short')

    g_phase_array = np.linspace(par.g_phase, -par.g_phase, par.n_phase, endpoint=False)
    
    n_phase_cycle = 8
    phase_cycle_90 = [0, 180, 0, 180, 90, 270, 90, 270]
    phase_cycle_180 = [90, 90, 270, 270, 0, 0, 180, 180]

    p_rf_acc = 0 # accumation of RF phase by running at a different frequency
    p_90_inc = 0 # phase accumulated by a 90 pulse (to be calculated)
    p_180_inc = 0 # phase accumulated by a 180 pulse (to be calculated)

    if par.soft_pulse_type in ["3-lobe sinc", "5-lobe sinc"]:
        pt_90, y_90 = make_soft_pulse(par.soft_pulse_type, par.t_90, par.a_90, decim=20) # pulse_times, sinc amplitudes
        pt_180, y_180 = make_soft_pulse(par.soft_pulse_type, par.t_180, par.a_180, decim=20)

    p_90_inc = 360*par.f_slice_offset*par.t_90
    p_180_inc = 360*par.f_slice_offset*par.t_180
    log.debug(f'p_90_inc: {p_90_inc}, p_180_inc: {p_180_inc}')
    log.debug(f'f_offset: {par.f_slice_offset} Hz')
    log.debug(f'g_read: {par.g_read}')
    log.debug(f'g_phase: {par.g_phase}')
    log.debug(f'g_slice: {par.g_slice}')
    # choose tx/rx channel
    rx = seq.rx[par.rx_chan]
    rx_list = list(seq.rx[i] for i in par.rx_chan_list)
    tx = seq.tx[par.tx_chan]

    g_shim = np.array(par.offsets[:3])

    # Tx/Rx constant settings
    yield from [tx.freq(par.f) for tx in seq.tx]
    yield rx.freq(par.f)
    yield rx.dwelltime(par.t_dw)
    yield rx.mode(flatfilter=True)
    yield seq.wait(0.01)

    # apply shim
    yield grad_ramp(seq, G_ZERO, g_shim, par.t_ramp)
    yield seq.wait(0.1)
    
    for i_phase, g_phase_i in enumerate(g_phase_array):
        for i_scan in range(par.n_scans):
            p_rf_acc = p_rf_acc % 360 # prevent growing too large and causing floating point rounding errors
            p_90 = p_acq = phase_cycle_90[i_scan % n_phase_cycle]
            p_180 = phase_cycle_180[i_scan % n_phase_cycle]
    
            # Sliced Excitation Pulse
            p_rf_acc += p_90_inc/2

            yield grad_ramp(seq, g_shim, g_shim+par.g_slice, par.t_ramp) 
            yield tx.freq(par.f+par.f_slice_offset) # set to offset frequency
           
            if par.soft_pulse_type in ["3-lobe sinc", "5-lobe sinc"]:
                yield from run_soft_pulse(seq, tx, (p_90-p_rf_acc)%360, pt_90, y_90)
            if par.soft_pulse_type == "SLR":
               yield soft_pulse(seq, tx, (p_90-p_rf_acc)%360, par.a_90, par.t_90, par.bw_90)
            
            yield tx.freq(par.f) # reset transmitter back to reference frame frequency
            yield grad_ramp(seq, g_shim+par.g_slice, g_shim, par.t_ramp) 
            p_rf_acc += p_90_inc/2
            
            yield seq.wait(te_ms)

            yield grad_ramp(seq, g_shim, g_shim+g_phase_i+par.g_read, par.t_ramp)
            yield seq.wait(par.t_phase)
            yield grad_ramp(seq, g_shim+g_phase_i+par.g_read, g_shim, par.t_ramp)
            yield seq.wait(te_ms)
            
            p_rf_acc += p_180_inc/2
            yield grad_ramp(seq, g_shim, g_shim+par.g_slice, par.t_ramp) 
            yield seq.wait(par.t_crush)
            yield tx.freq(par.f+par.f_slice_offset) # set to offset frequency

            if par.soft_pulse_type in ["3-lobe sinc", "5-lobe sinc"]:
                yield from run_soft_pulse(seq, tx, (p_180-p_rf_acc)%360, pt_180, y_180)
            if par.soft_pulse_type == "SLR":
               yield soft_pulse(seq, tx, (p_180-p_rf_acc)%360, par.a_180, par.t_180, par.bw_90)

            yield tx.freq(par.f) # reset transmitter back to reference frame frequency
            yield seq.wait(par.t_crush)
            yield grad_ramp(seq, g_shim+par.g_slice, g_shim, par.t_ramp) 
            p_rf_acc += p_180_inc/2
            
            yield seq.wait(te_ms_2)


            yield grad_ramp(seq, g_shim, g_shim+par.g_read, par.t_ramp)

            yield from [rx.phase(p_acq) for rx in rx_list]
            if i_phase is not None:
                yield from [
                    # indexing: [channel, phase_1, phase_2]
                    rx.acquire(i_rx*par.n_phase+i_phase, par.n_samples)
                    for i_rx, rx in enumerate(rx_list)]

            yield seq.wait(t_acq)
            yield grad_ramp(seq, g_shim+par.g_read, g_shim, par.t_ramp)

            yield seq.wait(t_end)
            

        yield grad_ramp(seq, g_shim, G_ZERO, par.t_ramp)











