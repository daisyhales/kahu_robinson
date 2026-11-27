from matipo.sequence import ParDef, floatarray
from collections import namedtuple
import numpy as np

from matipo.util.kahu_rf_module import pulse_start, pulse_end
# from matipo.util.pulseshape import calc_soft_pulse
from scipy import signal
import logging
import yaml
import scipy as sp
logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)


g_ZERO = np.array([0, 0, 0])

from matipo import SEQUENCE_DIR, GLOBALS_DIR
from user_interface.general import general_functions
from user_interface.panes import panes

all_pns = panes() # using the () will allow the self parameters to be used
g = general_functions(all_pns)


g_ZERO = np.array([0, 0, 0])
file = GLOBALS_DIR # where you want the global parameters to be taken from 
params = g.get_yaml_info(info_name = "current_cals", file = file)
G_MAX = [params["xgrad"], params["ygrad"], params["zgrad"]]
B0_MAX = [params["b0grad"]]


def g_ideal1(g_start, g_end, t_ramp, t_grad, t_rf_wait, t_next, t_acq_delay, t_acq, t_wait, waits_pulses, t_end, dt = 10e-6, pulse_time = None, amp = None, phase = None, acq = "n", p_acq = 0, ):
    """
    """

    N_rise = int(t_ramp/dt)
    N_flat = int(t_grad/dt)
    N_fall = int(t_ramp/dt)
    N_pad = int((t_end + t_next)/dt)
    
    N_rfwaitb = int((t_grad - t_rf_wait)/dt)
    N_rfwaitd = int(t_rf_wait/dt)
    N_rf = int(pulse_time/dt)
    N_acq_delay = int(t_acq_delay/dt)
    N_acq = int(t_acq/dt)
    N_wait = int(t_wait/dt)
    
    g_rise_array = np.linspace(g_start, g_end, N_rise+1)[:-1] # excluding the first 0 point, we want to be setting a non-zero value immediately
    g_flat_array = np.array([g_end]*N_flat)
    g_fall_array = np.linspace(g_end, g_start, N_fall+1)[1:] # excluding the first point, we want to start falling immediately and end at 0
    g_pad_array = np.linspace(g_start, g_start, N_pad+1)[:-1]

    g_ideal = np.concatenate((g_rise_array, g_flat_array, g_fall_array, g_pad_array))

    # --------------------------------------
    rf_arr1 = np.zeros(len(g_rise_array))
    freq_arr1 = np.zeros(len(g_rise_array))
    phase_arr1 = np.zeros(len(g_rise_array))
    acq_arr1 = np.zeros(len(g_rise_array))
    
    rf_arr2 = np.zeros(N_rfwaitb)
    freq_arr2 = np.zeros(N_rfwaitb)
    phase_arr2 = np.zeros(N_rfwaitb)
    acq_arr2 = np.zeros(N_rfwaitb)

    # pulse rf (hard pulse)
    i_pulse = [1]
    rf_arr3 = np.linspace(amp, amp, N_rf)
    freq_arr3 = np.ones(N_rf)
    phase_arr3 = np.linspace(phase, phase, N_rf)
    acq_arr3 = np.zeros(N_rf)
    
    # turn rf off, wait t_acq_delay then acquire -- this is the one with the g_read still on
    rf_arr4 = np.zeros(N_acq_delay + N_acq)
    freq_arr4 = np.zeros(N_acq_delay + N_acq)
    phase_arr4 = np.concatenate((np.zeros(N_acq_delay), np.linspace(p_acq, p_acq, N_acq)))
    acq_arr4 = np.concatenate((np.zeros(N_acq_delay), np.ones(N_acq)))

    # ramp off the g_read
    rf_arr5 = np.zeros(N_fall+N_wait)
    freq_arr5 = np.zeros(N_fall+N_wait)
    phase_arr5 = np.zeros(N_fall+N_wait)
    acq_arr5 = np.zeros(N_fall+N_wait)

    rf_arr6 = []
    freq_arr6 = []
    phase_arr6 = []
    acq_arr6 = []
    
    for j, t_pulse_wait in enumerate(waits_pulses):
        i_pulse.append(i_pulse[-1] + 1)
        # hard pulse
        rf_arrw = np.linspace(amp, amp, N_rf)
        freq_arrw = np.ones(N_rf)
        phase_arrw = np.linspace(phase, phase, N_rf)
        acq_arrw = np.zeros(N_rf)

        # wait acq delay then acquire
        rf_arrw1 = np.zeros(N_acq_delay + N_acq)
        freq_arrw1 = np.zeros(N_acq_delay + N_acq)
        phase_arrw1 = np.concatenate((np.zeros(N_acq_delay), np.linspace(p_acq, p_acq, N_acq)))
        acq_arrw1 = np.concatenate((np.zeros(N_acq_delay), np.ones(N_acq)))

        N_wait_pulse = int(t_pulse_wait / dt)
        # wait the exponential spacing required before next pulse
        
        rf_arrw2 = np.zeros(N_wait_pulse)
        freq_arrw2 = np.zeros(N_wait_pulse)
        phase_arrw2 = np.zeros(N_wait_pulse)
        acq_arrw2 = np.zeros(N_wait_pulse)

        rf_arr6.append(rf_arrw)
        rf_arr6.append(rf_arrw1)
        rf_arr6.append(rf_arrw2)
        freq_arr6.append(freq_arrw)
        freq_arr6.append(freq_arrw1)
        freq_arr6.append(freq_arrw2)
        phase_arr6.append(phase_arrw)
        phase_arr6.append(phase_arrw1)
        phase_arr6.append(phase_arrw2)
        acq_arr6.append(acq_arrw)
        acq_arr6.append(acq_arrw1)
        acq_arr6.append(acq_arrw2)

    
    rf_arr6 = np.concatenate(rf_arr6)
    freq_arr6 = np.concatenate(freq_arr6)
    phase_arr6 = np.concatenate(phase_arr6)
    acq_arr6 = np.concatenate(acq_arr6)
    
    # final pulse 
    i_pulse.append(i_pulse[-1] + 1)

    rf_arr7 = np.linspace(amp, amp, N_rf)
    freq_arr7 = np.ones(N_rf)
    phase_arr7 = np.linspace(phase, phase, N_rf)
    acq_arr7 = np.zeros(N_rf)
    
    # final acq delay and acquire
    rf_arr8 = np.zeros(N_acq_delay + N_acq)
    freq_arr8 = np.zeros(N_acq_delay + N_acq)
    phase_arr8 = np.concatenate((np.zeros(N_acq_delay), np.linspace(p_acq, p_acq, N_acq)))
    acq_arr8 = np.concatenate((np.zeros(N_acq_delay), np.ones(N_acq)))

    
    
    rf_arr = np.concatenate((rf_arr1, rf_arr2, rf_arr3, rf_arr4, rf_arr5, rf_arr6, rf_arr7, rf_arr8))
    freq_arr = np.concatenate((freq_arr1, freq_arr2, freq_arr3, freq_arr4, freq_arr5, freq_arr6, freq_arr7, freq_arr8))
    phase_arr = np.concatenate((phase_arr1, phase_arr2, phase_arr3, phase_arr4, phase_arr5, phase_arr6, phase_arr7, phase_arr8))
    acq_arr = np.concatenate((acq_arr1, acq_arr2, acq_arr3, acq_arr4, acq_arr5, acq_arr6, acq_arr7, acq_arr8))


    # generate time axis
    t = dt*np.arange(len(g_ideal))

    # pad to gradient length
    pad_len = len(g_ideal) - len(rf_arr)

    if pad_len > 0:
        rf_arr = np.concatenate((rf_arr, np.zeros(pad_len)))
        freq_arr = np.concatenate((freq_arr, np.zeros(pad_len)))
        phase_arr = np.concatenate((phase_arr, np.zeros(pad_len)))
        acq_arr = np.concatenate((acq_arr, np.zeros(pad_len)))

    return g_ideal, t, rf_arr, freq_arr, acq_arr, phase_arr, i_pulse



    
def g_ideal_ramp(g_start, g_end, t_ramp, t_grad, dt = 10e-6, acq = "n", p_acq = None):
    """
    """
    N_rise = int(t_ramp/dt)
    N_flat = int(t_grad/dt)
    
    g_rise_array = np.linspace(g_start, g_end, N_rise+1)[:-1] # excluding the first 0 point, we want to be setting a non-zero value immediately
    g_flat_array = np.array([g_end]*N_flat)

    g_ideal = np.concatenate((g_rise_array, g_flat_array))
    
    if acq == "n":
        acq_arr = np.zeros(len(g_ideal))
    else:
        raise ValueError("add acq in function")

    rf_arr = np.zeros(len(g_ideal))
    freq_arr = np.zeros(len(g_ideal))       
    phase_arr = np.zeros(len(g_ideal))       
    
    # generate time axis
    t = dt*np.arange(N_rise+N_flat)

    return g_ideal, t, rf_arr, freq_arr, acq_arr, phase_arr




def get_exp(file = GLOBALS_DIR, b0 = "n", opt = "n", compare = "n", file_parsE = None, file_parsB = None):
    """
    """
    if opt == "y":
        if b0 =="y":
           info_name = "eddy_B0opt.yaml" 
        else:
            info_name = "eddy_Eopt.yaml"
            
    elif compare == "y":
        log.debug(f"file_parsE {file_parsE}, file_parsB {file_parsB}")
        if file_parsE != None:
            info_name = file_parsE
            log.debug(f"Getting pars in file: file_parsE, {file_parsE}")
        elif file_parsB != None:
            info_name = file_parsB
            log.debug(f"Getting pars in file: file_parsB, {file_parsB}")
        else: 
            raise ValueError("file_parsB or E need value.")
            
        
    else:
        if b0 =="y":
           info_name = "eddy_B0.yaml" 
        else:
            info_name = "eddy_E.yaml"
            
    g.log_msg(f"{file}{info_name}")
    print(f"{file}{info_name}")
    
    try:
        with open(f"{file}{info_name}", "r") as f:
            val = yaml.safe_load(f)

        if isinstance(val, dict):
            return val
        
        params_dict = {}
        for item in val:
            params_dict.update(item)
        
        return params_dict
        
    except Exception as e:
        print(e)
        raise ValueError(e)

    
def get_calibrations(file = GLOBALS_DIR):
    """
    """
    try:
        with open(f"{file}B0_calibration.yaml", "r") as f:
            B0_cal = yaml.safe_load(f)
    
        with open(f"{file}g_hzpm_slope.yaml", "r") as f:
            G_cal = yaml.safe_load(f)

    except Exception as e:
        ValueError(f"file in folder {file} doesnt exist")

    return B0_cal["m"], G_cal



def add_b0(g_preemph_g, params_dict_b0, dt = 10e-6, file = GLOBALS_DIR): 
    """
    """
    # log.debug(f"B0 params: {params_dict_b0}")
    g_b0_all = []
    g_all = []
    
    for i, axis in enumerate(params_dict_b0.keys()):
        # go through the axes
        amps = [params_dict_b0[axis][k] for k in ["A1", "A2", "A3", "A4"]]#, "A4", "A5", "A6"]]#, "A7", "A8", "A9", "A10", "A11", "A12"]]
        taus = [params_dict_b0[axis][k] for k in ["tau1", "tau2", "tau3", "tau4"]]#, "tau4", "tau5", "tau6"]]#, "tau7", "tau8", "tau9", "tau10", "tau11", "tau12"]]
        # apply eddy current and inverse filters to ideal gradient
        # to calculate the resulting gradient waveform and preemphisized waveform

        g_preemphB0 = np.zeros(len(g_preemph_g[:, i]))
        for coeff, time_const in zip(amps, taus):

            if coeff == 0 or time_const == 0:
                continue

            alpha = dt / time_const

            b = [-coeff, coeff]
            a = [1, -1/(1+alpha)]

            g_preemphB0 += -sp.signal.lfilter(b, a, g_preemph_g[:, i])

        g_b0_all.append(g_preemphB0)
        
    g_b0_all = np.column_stack(g_b0_all) 

    B0_cal, G_cal = get_calibrations(file)

    # apply G_cal/B0_cal for each col of g_b0_all[x, Y, Z]
    for i, axis in enumerate(["X", "Y", "Z"]):
    
        g_cal = np.where(g_preemph_g[:, i] >= 0, G_cal[axis], G_cal[axis]) # creates an array of calibration values pos or neg depending on sign of g_ideal
    
        g_b0_all[:, i] = g_b0_all[:, i] * (g_cal / B0_cal) # scale the convolution (g_b0_all axis) by the g_cal and B0_cal 
    
    b0_sum = g_b0_all.sum(axis=1) # sums up all the values in each vector
    
    return b0_sum


def add_pre_emph(g_ideal, params_dict, dt = 10e-6):
    """
    """
    g_all = []
    g_preemph_all = []
    for i, axis in enumerate(params_dict.keys()):
        # go through the axes
        amps = [params_dict[axis][k] for k in ["A1", "A2", "A3"]]
        taus = [params_dict[axis][k] for k in ["tau1", "tau2", "tau3"]]
        # apply eddy current and inverse filters to ideal gradient
        # to calculate the resulting gradient waveform and preemphisized waveform
        
        g_preemph = g_ideal[:, i]
        for j in range(len(amps)):
        
            coeff = amps[j]
            time_const = taus[j]

            if coeff == 0 or time_const == 0:
                continue
        
            alpha = dt / time_const
            b_inv = [1, -1/(1+alpha)]
            a_inv = [1 - coeff, coeff - 1/(1+alpha)]
                
            g_preemph = sp.signal.lfilter(b_inv, a_inv, g_preemph)

        g_preemph_all.append(g_preemph)

    return np.column_stack(g_preemph_all)


def run_seq(seq, tx, phase, rf_all, g_preemph_all, f_all, a_all, phase_all, f, f_slice_offset, n_acq, i_pulse, n_samples, rx_list, b0_sum = None, dt=10e-6, G_MAX = None, B0_MAX = None):

    cmds = []
    n = len(rf_all)
    rf_on = False
    acq_on = False
    acq_counter = 0

    if b0_sum is None:
        for i in range(n):
            # catch big gradients
            if np.any(np.abs(g_preemph_all[i]) > G_MAX):
                raise Exception(f'Gradient Too Large! |g_preemph {i}|={g_preemph_all[i]}>{G_MAX}')

            # set the tx on to the freq and start the hard pulse
            if f_all[i] == 1 and not rf_on:
                step_cmds = [tx.freq(f+f_slice_offset),
                            pulse_start(seq, tx, phase_all[i], rf_all[i]), 
                            seq.grad[0].vec(*g_preemph_all[i]),
                            seq.wait(dt)]
                rf_on = True

               
            # freq on and pulse continues
            elif f_all[i] == 1 and rf_on:     
                step_cmds = [
                    tx.amp(rf_all[i]),
                    seq.grad[0].vec(*g_preemph_all[i]),
                    seq.wait(dt)
                ]
                
            # rf, phase and freq turns off at end of pulse
            elif f_all[i] == 0 and rf_on:
                step_cmds = [tx.freq(f), 
                             pulse_end(seq, tx), 
                             seq.grad[0].vec(*g_preemph_all[i]),
                             seq.wait(dt)]
                rf_on = False


            # if matrix says to acquire - start the acquision (acq_on) and build acq 
            elif a_all[i] == 1 and not acq_on:
                step_cmds_comb = [[rx.phase(phase_all[i]) for rx in rx_list], 
                                  [rx.acquire(i_pulse[acq_counter] + n_acq * i_chan, n_samples) for i_chan, rx in enumerate(rx_list)], 
                                  seq.grad[0].vec(*g_preemph_all[i]),
                                  seq.wait(dt)]
    
                step_cmds = []
                
                for x in step_cmds_comb:
                    if isinstance(x, list):
                        step_cmds.extend(x)
                    else:
                        step_cmds.append(x)
                acq_on = True
                acq_counter += 1
            # turn off acquision once done
            elif a_all[i] == 0 and acq_on:  
                acq_on = False
                step_cmds = [
                    seq.grad[0].vec(*g_preemph_all[i]),
                    seq.wait(dt)
                ]
            # if not acquiring or pulsing just play gradient
            else:
                
                step_cmds = [
                    seq.grad[0].vec(*g_preemph_all[i]),
                    seq.wait(dt)
                ]
    
            cmds.append(seq.join(step_cmds))
    
        return seq.join(cmds)
        
    else:
        log.debug(f"MAX B0: {np.max(b0_sum)} !!!!!!!!!!!!!!!!")
        for i in range(n):
            
            if np.any(np.abs(g_preemph_all[i]) > G_MAX):
                raise Exception(f'Gradient Too Large! |g_preemph {i}|={g_preemph_all[i]}>{G_MAX}')
                
            if np.abs(b0_sum[i]) > B0_MAX:
                raise Exception(f'Gradient Too Large! |b0_sum {i}|={b0_sum[i]}>{B0_MAX}')
                
            if f_all[i] == 1 and not rf_on:
                step_cmds = [tx.freq(f+f_slice_offset),
                            pulse_start(seq, tx, phase_all[i], rf_all[i]), 
                            seq.grad[0].vec(*g_preemph_all[i]),
                            seq.grad[0].aux(b0_sum[i]),
                            seq.wait(dt)]
                rf_on = True
    
            elif f_all[i] == 1 and rf_on:     
                step_cmds = [
                    tx.amp(rf_all[i]),
                    seq.grad[0].vec(*g_preemph_all[i]),
                    seq.grad[0].aux(b0_sum[i]),
                    seq.wait(dt)
                ]
                
            elif f_all[i] == 0 and rf_on:
                step_cmds = [tx.freq(f), 
                             pulse_end(seq, tx), 
                             seq.grad[0].vec(*g_preemph_all[i]),
                             seq.grad[0].aux(b0_sum[i]),
                             seq.wait(dt)]
                rf_on = False
                
            elif a_all[i] == 1 and not acq_on:
                # if acq_counter >= len(i_pulse):
                    # log.debug(f"ERROR")
                    # log.debug(f"acq_counter = {acq_counter}")
                    # log.debug(f"len(i_pulse) = {len(i_pulse)}")
                    # log.debug(f"i_pulse = {i_pulse}")

                step_cmds_comb = [[rx.phase(phase_all[i]) for rx in rx_list], 
                                  [rx.acquire(i_pulse[acq_counter] + n_acq * i_chan, n_samples) for i_chan, rx in enumerate(rx_list)], 
                                  seq.grad[0].vec(*g_preemph_all[i]),
                                  seq.grad[0].aux(b0_sum[i]),
                                  seq.wait(dt)]
                step_cmds = []
                
                for x in step_cmds_comb:
                    if isinstance(x, list):
                        step_cmds.extend(x)
                    else:
                        step_cmds.append(x)
                acq_on = True
                acq_counter += 1
    
            elif a_all[i] == 0 and acq_on:  
                acq_on = False
                step_cmds = [
                    seq.grad[0].vec(*g_preemph_all[i]),
                    seq.grad[0].aux(b0_sum[i]), 
                    seq.wait(dt)
                ]
            else:
                
                step_cmds = [
                    seq.grad[0].vec(*g_preemph_all[i]),
                    seq.grad[0].aux(b0_sum[i]), 
                    seq.wait(dt)
                ]
    
            cmds.append(seq.join(step_cmds))
    
        return seq.join(cmds)


def combine_g_ideal(g_list, t_list, rf_list, f_list, acq_list, phase_list):
    """
    """
    g_ideal_all = np.concatenate(g_list)
    rf_all = np.concatenate(rf_list)
    f_all = np.concatenate(f_list)
    acq_all = np.concatenate(acq_list)
    phase_all = np.concatenate(phase_list)
    
    t_out = []
    offset = 0
    
    for t in t_list:
        t_out.append(t + offset)
        offset += t[-1] + (t[1] - t[0])
    t_ideal_all = np.concatenate(t_out)
    return g_ideal_all, t_ideal_all, rf_all, f_all, acq_all, phase_all


PARDEF = [
    # args: name, dtype, default value
    ParDef('n_scans', int, 1, min=1, unit=''),
    ParDef('f', float, 1e6, unit='Hz'),
    ParDef('a_pulse', float, 0, min=0, max=1, unit=''),
    ParDef('t_pulse', float, 32e-6, unit='s'),


    ParDef('t_dw', float, 1e-6, min=0.1e-6, max=160e-6, unit='s'),
    ParDef('n_samples', int, 1000, min=2, unit=''),
    ParDef('n_pulses', int, 1000, min=2, unit=''),

    ParDef('t_acq_delay', float, 100e-6, unit='s'),
    ParDef('g_read', floatarray, (0,0,0), max=(1,1,1), unit=''),

    ParDef('t_grad_spacing', float, 10e-6, unit = 's'),
    ParDef('t_grad', float, 10, unit = 's'),
    ParDef('t_ramp', float, 1000e-6, unit='s'),
    ParDef('t_wait', float, 1000e-6, unit='s'),
    ParDef('t_end', float, 1000e-6, unit='s'),
    ParDef('t_next', float, 1000e-6, unit='s'),
    
    ParDef('pre_emphasis', bool, True, unit=''),
    ParDef('b0', bool, True, unit=''),
    
    ParDef('offsets', floatarray, np.zeros(4), min=-1, max=1, unit=''),
    # ParDef('shims', floatarray, np.zeros(16), min=-1, max=1, unit=''),
    ParDef('tx_chan', int, 0, min=0, max=1, unit=''),
    ParDef('rx_chan_list', floatarray, 0, min=0, max=3, unit=''),
    ParDef('file_cals', str, GLOBALS_DIR),
    ParDef('file_pars', str, GLOBALS_DIR),
    ParDef('opt', str, "n"),
    ParDef('compare', str, "n"),
    ParDef('file_parsE', str, GLOBALS_DIR),
    ParDef('file_parsB', str, GLOBALS_DIR),
    
        
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
    t_acq = par.n_samples * par.t_dw

    vals = np.geomspace(par.t_wait, par.t_end, num=par.n_pulses)
    
    seps = np.diff(vals)
    
    minus = par.t_pulse + t_acq + par.t_acq_delay
    waits_pulses = []
    for i in seps:
        w = float(i-minus)
        if w < 0:
            raise ValueError("exp spacing is less than zero, change t_pulse or t_acq time")
        else:
            waits_pulses.append(w)

    # log.debug(f"g_read: {par.g_read}")

    n_phase_cycle = 8
    phase_cycle_90 = [0, 180, 0, 180, 90, 270, 90, 270]
    phase_cycle_180 = [90, 90, 270, 270, 0, 0, 180, 180]
    
    # choose tx/rx channel
    tx = seq.tx[par.tx_chan]
    
    rxs = [seq.rx[(int(i))] for i in par.rx_chan_list] # rx array for the rx channels -- seq.rx[par.rx_chan]
    grad = seq.grad[0] # first parameter is the gradient and the second is the aux value - sets access to the gradient controll

    n_acq = par.n_pulses + 1 # number of acquisitions per channel
    t_rf_wait = par.t_pulse + par.t_acq_delay + t_acq
    # t_next = 10 # time between scans to not overload the amplifiers
    t_shim_set = 0.1
    dt = 10e-6
    
    g0, t0, rf0, f0, a0, p0 = g_ideal_ramp(g_ZERO, g_shim, par.t_ramp, t_shim_set, dt)
    
    runs_rf = [rf0]
    runs_g = [g0]
    runs_t = [t0]
    runs_f = [f0]
    runs_a = [a0]
    runs_p = [p0]
    i_pulse_scans = []
    
    for i_scan in range(par.n_scans):

        p_90 = p_acq = phase_cycle_90[i_scan % n_phase_cycle]
    
        g1, t1, rf1, f1, a1, p1, i_pulse = g_ideal1(g_shim, g_shim+par.g_read, par.t_ramp, par.t_grad, t_rf_wait, par.t_next, par.t_acq_delay, t_acq, par.t_wait, waits_pulses, par.t_end, dt = dt, pulse_time = par.t_pulse, amp = par.a_pulse, phase = p_90, acq = "n", p_acq = p_acq)
    
        runs_rf.append(rf1)
        runs_g.append(g1)
        runs_t.append(t1)
        runs_f.append(f1)
        runs_a.append(a1)
        runs_p.append(p1)
        i_pulse_scans.extend(i_pulse)

    g2, t2, rf2, f2, a2, p2 = g_ideal_ramp(g_shim, g_ZERO, par.t_ramp, t_shim_set, dt)
    
    runs_rf.append(rf2)
    runs_g.append(g2)
    runs_t.append(t2)
    runs_f.append(f2)
    runs_a.append(a2)
    runs_p.append(p2)

    
    g_ideal_all, t_ideal_all, rf_all, f_all, a_all, p_all = combine_g_ideal(runs_g, runs_t, runs_rf, runs_f, runs_a, runs_p)
    # i_pulse_scans.flatten()
 
    if par.pre_emphasis and par.b0: # both true
        params_dict = get_exp(file = par.file_pars, opt = par.opt, compare = par.compare, file_parsE = par.file_parsE)
        params_dict_b0 = get_exp(file = par.file_pars, b0 = "y", opt = par.opt, compare = par.compare, file_parsB = par.file_parsB)
        
        g_preemph_all = add_pre_emph(g_ideal_all, params_dict)

        log.debug(f"B0 uses pre-emphasised gradient")        
        b0_sum = add_b0(g_preemph_all, params_dict_b0, file = par.file_cals)
    elif par.b0 and not par.pre_emphasis: # just B0 is true
        params_dict_b0 = get_exp(file = par.file_pars, b0 = "y", opt = par.opt, compare = par.compare, file_parsB = par.file_parsB)
        log.debug(f"B0 uses ideal gradient")
        b0_sum = add_b0(g_ideal_all, params_dict_b0, file = par.file_cals)
    elif not par.b0 and not par.pre_emphasis:
        pass
    else: # both PE true and B0 not 
        params_dict = get_exp(file = par.file_pars, opt = par.opt, compare = par.compare, file_parsE = par.file_parsE)
        params_dict_b0 = get_exp(file = par.file_pars, b0 = "y", opt = par.opt, compare = par.compare, file_parsB = par.file_parsB)
        
        g_preemph_all = add_pre_emph(g_ideal_all, params_dict)
        
    grad = seq.grad[0]
    rxs = [seq.rx[(int(i))] for i in par.rx_chan_list] # rx array for the rx 
    # Tx/Rx constant settings
    yield tx.freq(par.f)

    yield from [rx.freq(par.f) for rx in rxs]
    yield from [rx.dwelltime(par.t_dw) for rx in rxs]
    yield from [rx.mode(flatfilter=True) for rx in rxs]
    yield seq.wait(0.01)

    acq_starts = np.sum((a_all[1:] == 1) & (a_all[:-1] == 0))

    if par.pre_emphasis == True:
        log.debug("Pre-emphasis ON")
        if par.b0 == True:
            log.debug("B0 ON")
            yield run_seq(seq, tx, 0, rf_all, g_preemph_all, f_all, a_all, p_all, par.f, 0, n_acq, i_pulse_scans, par.n_samples, rxs, b0_sum, dt, G_MAX, B0_MAX)
        if par.b0 == False:
            log.debug("B0 OFF")
            yield run_seq(seq, tx, 0, rf_all, g_preemph_all, f_all, a_all, p_all, par.f, 0, n_acq, i_pulse_scans, par.n_samples, rxs, None, dt, G_MAX, B0_MAX)
    else:
        log.debug("Pre-emphasis OFF")
        if par.b0 == True:
            log.debug("B0 ON")
            yield run_seq(seq, tx, 0, rf_all, g_ideal_all, f_all, a_all, p_all, par.f, 0, n_acq, i_pulse_scans, par.n_samples, rxs,  b0_sum, dt, G_MAX, B0_MAX)
        if par.b0 == False:
            log.debug("B0 OFF")
            yield run_seq(seq, tx, 0, rf_all, g_ideal_all, f_all, a_all, p_all, par.f, 0, n_acq, i_pulse_scans, par.n_samples, rxs, None, dt, G_MAX, B0_MAX)