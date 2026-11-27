from matipo.sequence import ParDef, floatarray
from collections import namedtuple
import numpy as np

from matipo.util.kahu_rf_module import pulse_start, pulse_end
from matipo.util.pulseshape import calc_soft_pulse
from scipy import signal
import logging
import yaml
import scipy as sp
logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)


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
MAX_EXP = g.get_yaml_info(info_name = "fitting_pars", file = file)["MAX_EXP"]

def calc_soft_pulse(pulse_width, pulse_bandwidth, flip_angle=90, N_max=256, spacing_min=1e-6):
    if pulse_bandwidth == 0: # hardpulse
        return 1, pulse_width, np.array([1.0])

    N = int(pulse_width / spacing_min)
    if N > N_max:
        N = N_max
    spacing = pulse_width / N

    # SLR method see http://ee-classes.usc.edu/ee591/library/Pauly-SLR.pdf
    a = [5.309e-3, 7.114e-2, -4.761e-1, -2.66e-3, -5.941e-1, -4.278e-1]
    delta_1_eff = 0.01
    delta_2_eff = 0.01

    # for 90 pulse
    if flip_angle == 90:
        delta_1 = np.sqrt(delta_1_eff / 2)
        delta_2 = delta_2_eff / np.sqrt(2)
    elif flip_angle == 180:
        delta_1 = delta_1_eff / 8
        delta_2 = np.sqrt(delta_2_eff / 2)
    else:
        delta_1 = delta_1_eff
        delta_2 = delta_2_eff
    # for spin echo
    # delta_1 = delta_1_eff/4
    # delta_2 = np.sqrt(delta_2_eff)

    B = pulse_bandwidth
    TB = pulse_width * B
    L_1 = np.log10(delta_1)
    L_2 = np.log10(delta_2)
    D_inf = (a[0] * L_1 * L_1 + a[1] * L_1 + a[2]) * L_2 + (a[3] * L_1 * L_1 + a[4] * L_1 + a[5])
    W = D_inf / TB
    BW = B * W
    F_sample = N / pulse_width
    F_pass = ((B - BW) / 2) / F_sample
    F_stop = ((B + BW) / 2) / F_sample

    if F_pass < 0:
        raise Exception("Softpulse (Bandwidth)*(Pulse Width) product too small (%.2f < %.2f)." % (TB, D_inf))

    if F_stop > 0.5:
        raise Exception(
            "Softpulse Bandwidth too large (%.2E Hz > %.2E Hz)." % (pulse_bandwidth, 1 / spacing_min - BW))

    points = signal.remez(N, [0, F_pass, F_stop, 0.5], [1, 0], weight=[1 / delta_1, 1 / delta_2])
    points *= 1 / np.max(points)

    return N, spacing, points

    
def g_ideal(g_start, g_end, t_ramp, t_grad, t_next, dt = 10e-6, rf_pulse = "n", pulse_width = None, pulse_bandwidth = None, amp = None, phase = None, acq = "n", p_acq = None):
    """
    """
    N_rise = int(t_ramp/dt)
    N_flat = int(t_grad/dt)
    N_fall = int(t_ramp/dt)
    N_pad = int(t_next/dt)
    
    g_rise_array = np.linspace(g_start, g_end, N_rise+1)[:-1] # excluding the first 0 point, we want to be setting a non-zero value immediately
    g_flat_array = np.array([g_end]*N_flat)
    g_fall_array = np.linspace(g_end, g_start, N_fall+1)[1:] # excluding the first point, we want to start falling immediately and end at 0
    g_pad_array = np.linspace(g_start, g_start, N_pad+1)[:-1]

    
    g_ideal = np.concatenate((g_rise_array, g_flat_array, g_fall_array, g_pad_array))

    if acq == "n":
        acq_arr = np.zeros(len(g_ideal))
    else:
        acq_arr1 = np.zeros(len(g_rise_array))
        phase_arr1 = np.zeros(len(g_rise_array))
        acq_arr2 = np.ones(len(g_flat_array))
        phase_arr2 = np.linspace(p_acq, p_acq, len(g_flat_array))
        acq_arr3 = np.zeros(len(g_fall_array) + len(g_pad_array))
        phase_arr3 = np.zeros(len(g_fall_array) + len(g_pad_array))
        
        acq_arr = np.concatenate((acq_arr1, acq_arr2, acq_arr3))
        phase_arr = np.concatenate((phase_arr1, phase_arr2, phase_arr3))
    
    if rf_pulse == "n":
        rf_arr = np.zeros(len(g_ideal))
        freq_arr = np.zeros(len(g_ideal))
        if acq == "n": # only do phase if acq is off - rf and acq shouldnt be on at same time
            phase_arr = np.zeros(len(g_ideal))
    else:
        rf_arr1 = np.zeros(len(g_rise_array))
        freq_arr1 = np.zeros(len(g_rise_array))
        phase_arr1 = np.zeros(len(g_rise_array))
        
        N2, spacing2, rf_arr2 = calc_soft_pulse(pulse_width, pulse_bandwidth, flip_angle=90, N_max=256, spacing_min=dt)
        rf_arr2 *= amp
        freq_arr2 = np.ones(len(rf_arr2))
        phase_arr2 = np.linspace(phase, phase, len(rf_arr2))
        
        rf_arr3 = np.zeros(len(g_fall_array) + len(g_pad_array))
        freq_arr3 = np.zeros(len(g_fall_array) + len(g_pad_array))
        phase_arr3 = np.zeros(len(g_fall_array) + len(g_pad_array))
        
        rf_arr = np.concatenate((rf_arr1, rf_arr2, rf_arr3))
        freq_arr = np.concatenate((freq_arr1, freq_arr2, freq_arr3))
        phase_arr = np.concatenate((phase_arr1, phase_arr2, phase_arr3))
    
    # generate time axis
    t = dt*np.arange(N_rise+N_flat+N_fall+N_pad)

    return g_ideal, t, rf_arr, freq_arr, acq_arr, phase_arr
    
def g_ideal_two(g_start, g_first, g_second, t_ramp, t_grad_first, t_grad_second, t_next, dt = 10e-6, rf_pulse = "n", pulse_width = None, pulse_bandwidth = None, amp = None, phase = None, acq = "n", p_acq = None):
    """
    """
    N_rise = int(t_ramp/dt)
    N_flat_first = int(t_grad_first/dt)
    N_flat_second = int(t_grad_second/dt)
    N_fall = int(t_ramp/dt)
    N_pad = int(t_next/dt)
    
    g_rise_array1 = np.linspace(g_start, g_first, N_rise+1)[:-1] # excluding the first 0 point, we want to be setting a non-zero value immediately
    g_flat_array1 = np.array([g_first]*N_flat_first)
    g_rise_array2 = np.linspace(g_first, g_second, N_rise+1)[:-1] # excluding the first 0 point, we want to be setting a non-zero value immediately
    g_flat_array2 = np.array([g_second]*N_flat_second)
    
    g_fall_array1 = np.linspace(g_second, g_first, N_fall+1)[1:] # excluding the first point, we want to start falling immediately and end at 0
    g_flat_array3 = np.array([g_first]*N_flat_first)
    g_fall_array2 = np.linspace(g_first, g_start, N_fall+1)[1:] # excluding the first point, we want to start falling immediately and end at 0
    
    g_pad_array = np.linspace(g_start, g_start, N_pad+1)[:-1]

    
    g_ideal = np.concatenate((g_rise_array1, g_flat_array1, g_rise_array2, g_flat_array2, g_fall_array1, g_flat_array3, g_fall_array2, g_pad_array))

    if acq == "n":
         acq_arr = np.zeros(len(g_ideal))
    else:
        raise ValueError("add acq in function")
    
    if rf_pulse == "n":
        rf_arr = np.zeros(len(g_ideal))
        freq_arr = np.zeros(len(g_ideal))    
        phase_arr = np.zeros(len(g_ideal))
        
    else:
        rf_arr1 = np.zeros(len(g_rise_array1)+ len(g_flat_array1)+ len(g_rise_array2))
        freq_arr1 = np.zeros(len(g_rise_array1)+ len(g_flat_array1)+ len(g_rise_array2))
        phase_arr1 = np.zeros(len(g_rise_array1)+ len(g_flat_array1)+ len(g_rise_array2))
        
        N2, spacing2, rf_arr2 = calc_soft_pulse(pulse_width, pulse_bandwidth, flip_angle=90, N_max=256, spacing_min=dt)
        rf_arr2 *= amp
        freq_arr2 = np.ones(len(rf_arr2))
        phase_arr2 = np.linspace(phase, phase, len(rf_arr2))
        
        rf_arr3 = np.zeros(len(g_fall_array1)+ len(g_flat_array3)+ len(g_fall_array2) + len(g_pad_array))
        freq_arr3 = np.zeros(len(rf_arr3))
        phase_arr3 = np.zeros(len(rf_arr3))
        
        rf_arr = np.concatenate((rf_arr1, rf_arr2, rf_arr3))
        freq_arr = np.concatenate((freq_arr1, freq_arr2, freq_arr3))
        phase_arr = np.concatenate((phase_arr1, phase_arr2, phase_arr3))
        
    
    # generate time axis
    t = dt*np.arange(len(g_ideal))

    return g_ideal, t, rf_arr, freq_arr, acq_arr, phase_arr
    
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
    

def combine_g_ideal(g_list, t_list, rf_list, f_list, acq_list, phase_list):
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



def get_exp(file = GLOBALS_DIR, b0 = "n"):
    """
    """

    try:
        if b0 =="y":
           with open(f"{file}", "r") as f:
               val = yaml.safe_load(f)
        else:
           with open(f"{file}", "r") as f:
               val = yaml.safe_load(f)

    except Exception as e:
        raise ValueError(f"{file} doesnt exist")
        
    if isinstance(val, dict):
        return val
    
    params_dict = {}
    for item in val:
        params_dict.update(item)
    
    return params_dict


    
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



def add_b0(g_preemph_g, params_dict_b0, dt = 10e-6, file = GLOBALS_DIR, factor = 1): 
    """
    """
    # log.debug(f"B0 params: {params_dict_b0}")
    g_b0_all = []
    g_all = []

        
    a_list = []
    t_list = []
    for m in range(MAX_EXP):
        m = m+1
        a_list.append(f"A{m}")
        t_list.append(f"tau{m}")
    
    for i, axis in enumerate(params_dict_b0.keys()):
        # go through the axes
        amps = [params_dict_b0[axis][k] for k in a_list]#, "A4", "A5", "A6"]]#, "A7", "A8", "A9", "A10", "A11", "A12"]]
        taus = [params_dict_b0[axis][k] for k in t_list]#, "tau4", "tau5", "tau6"]]#, "tau7", "tau8", "tau9", "tau10", "tau11", "tau12"]]
        # apply eddy current and inverse filters to ideal gradient
        # to calculate the resulting gradient waveform and preemphisized waveform

        g_preemphB0 = np.zeros(len(g_preemph_g[:, i]))
        for coeff, time_const in zip(amps, taus):

            if coeff == 0 or time_const == 0:
                continue
            coeff = coeff * factor
            
            alpha = dt / time_const

            b = [-coeff, coeff]
            a = [1, -1/(1+alpha)]

            g_preemphB0 += -sp.signal.lfilter(b, a, g_preemph_g[:, i])

        g_b0_all.append(g_preemphB0)
        
    g_b0_all = np.column_stack(g_b0_all) 

    B0_cal, G_cal = get_calibrations(file)

    print("B0 cal:", B0_cal, "Hz/DAC\n", "G_cal:", G_cal, "Hz/m")
    # apply G_cal/B0_cal for each col of g_b0_all[x, Y, Z]
    for i, axis in enumerate(["X", "Y", "Z"]):
    
        g_cal = np.where(g_preemph_g[:, i] >= 0, G_cal[axis], G_cal[axis]) # creates an array of calibration values pos or neg depending on sign of g_ideal
    
        g_b0_all[:, i] = g_b0_all[:, i] * (g_cal / B0_cal) # scale the convolution (g_b0_all axis) by the g_cal and B0_cal 
    
    b0_sum = g_b0_all.sum(axis=1) # sums up all the values in each vector
    
    return b0_sum




def add_pre_emph(g_ideal, params_dict, dt = 10e-6, factor = 1):
    """
    """
    g_all = []
    g_preemph_all = []

            
    a_list = []
    t_list = []
    for m in range(MAX_EXP):
        m = m+1
        a_list.append(f"A{m}")
        t_list.append(f"tau{m}")

        
    for i, axis in enumerate(params_dict.keys()):
        # go through the axes
        amps = [params_dict[axis][k] for k in a_list]
        taus = [params_dict[axis][k] for k in t_list]
        # apply eddy current and inverse filters to ideal gradient
        # to calculate the resulting gradient waveform and preemphisized waveform
        
        g_preemph = g_ideal[:, i]
        for j in range(len(amps)):
        
            coeff = amps[j]
            time_const = taus[j]

            if coeff == 0 or time_const == 0:
                continue
            coeff = coeff * factor 
            alpha = dt / time_const
            b_inv = [1, -1/(1+alpha)]
            a_inv = [1 - coeff, coeff - 1/(1+alpha)]
                
            g_preemph = sp.signal.lfilter(b_inv, a_inv, g_preemph)

        g_preemph_all.append(g_preemph)

    return np.column_stack(g_preemph_all)



def run_seq(seq, tx, phase, rf_all, g_preemph_all, f_all, a_all, phase_all, f, f_slice_offset, n_phase, i_phase_list, n_samples, rx_list, b0_sum = None, dt=10e-6, G_MAX = None, B0_MAX = None):

    cmds = []
    n = len(rf_all)
    rf_on = False
    acq_on = False
    acq_counter = 0
    if b0_sum is None:
        for i in range(n):
            
            if np.any(np.abs(g_preemph_all[i]) > G_MAX):
                raise Exception(f'Gradient Too Large! |g_preemph {i}|={g_preemph_all[i]}>{G_MAX}')

                
            if f_all[i] == 1 and not rf_on:
                step_cmds = [tx.freq(f+f_slice_offset),
                            pulse_start(seq, tx, phase_all[i], rf_all[i]), 
                            seq.grad[0].vec(*g_preemph_all[i]),

                            seq.wait(dt)]
                rf_on = True
                
    
            elif f_all[i] == 1 and rf_on:     
                step_cmds = [
                    tx.amp(rf_all[i]),
                    seq.grad[0].vec(*g_preemph_all[i]),
                    seq.wait(dt)
                ]
            elif f_all[i] == 0 and rf_on:
                step_cmds = [pulse_end(seq, tx), tx.freq(f)]
                rf_on = False
                
            elif a_all[i] == 1 and not acq_on:
    
                step_cmds_comb = [[rx.phase(phase_all[i]) for rx in rx_list], 
                                  [rx.acquire(i_rx*n_phase+i_phase_list[acq_counter], n_samples) for i_rx, rx in enumerate(rx_list)], 
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
    
            elif a_all[i] == 0 and acq_on:  
                acq_on = False
                step_cmds = [
                    seq.grad[0].vec(*g_preemph_all[i]),
                    seq.wait(dt)
                ]
            else:
                
                step_cmds = [
                    seq.grad[0].vec(*g_preemph_all[i]),
                    seq.wait(dt)
                ]
    
            cmds.append(seq.join(step_cmds))
    
        return seq.join(cmds)
        
    else:
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
                step_cmds = [pulse_end(seq, tx), tx.freq(f)]
                rf_on = False
                
            elif a_all[i] == 1 and not acq_on:
    
                step_cmds_comb = [[rx.phase(phase_all[i]) for rx in rx_list], 
                                  [rx.acquire(i_rx*n_phase+i_phase_list[acq_counter], n_samples) for i_rx, rx in enumerate(rx_list)], 
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


# Define input parameters (must be named 'PARDEF')
PARDEF = [
    # args: name, dtype, default value
    ParDef('n_scans', int, 1, min=1, unit=''),#
    ParDef('f', float, 1e6, unit='Hz'),#
    ParDef('a_90', float, 0, min=0, max=1, unit=''),#
    ParDef('t_90', float, 1e-3, unit='s'),#
    ParDef('a_180', float, 0, min=0, max=1, unit=''),#
    ParDef('t_180', float, 1e-3, unit='s'),#
    ParDef('bw_90', float, 0, unit='Hz'),
    
    ParDef('pre_emphasis', bool, True, unit=''),
    ParDef('b0', bool, True, unit=''),
    # ParDef('soft_pulse_type', str, "3-lobe sinc", unit=''),

    ParDef('t_dw', float, 1e-6, min=0.1e-6, max=160e-6, unit='s'),#
    ParDef('n_samples', int, 1000, min=2, unit=''),#
    ParDef('n_phase', int, 1, min=1, unit=''),#
    ParDef('t_echo', float, 0, min=0, unit='s'),#t
    ParDef("TR", float, 1, unit='s'),
    ParDef('n_ETL', int, 1, min=1, unit=''),#

    ParDef('offsets', floatarray, np.zeros(4), min=-1, max=1, unit=''),
    ParDef('f_slice_offset', float, 0, unit='Hz'),
    ParDef('rx_chan_list', list, [0], unit=''),
    
    
    ParDef('g_slice', floatarray, (0, 0, 0), min=(-1, -1, -1), max=(1, 1, 1), unit=''),
    ParDef('g_crush', floatarray, (0, 0, 0), min=(-1, -1, -1), max=(1, 1, 1), unit=''),
    ParDef('g_spoil', floatarray, (0, 0, 0), min=(-1, -1, -1), max=(1, 1, 1), unit=''),
    ParDef('g_read', floatarray, (0, 0, 0), min=(-1, -1, -1), max=(1, 1, 1), unit=''),#
    ParDef('g_phase', floatarray, (0, 0, 0), min=(-1, -1, -1), max=(1, 1, 1), unit=''),
    ParDef('t_phase', float, 1000e-6, unit='s'),
    ParDef('t_crush', float, 1000e-6, unit='s'),
    ParDef('t_ramp', float, 1000e-6, unit='s'),#
    ParDef('t_spoil', float, 1e-3, unit='s'),
    ParDef('rx_chan', int, 0, min=0, max=3, unit=''),#
    ParDef('tx_chan', int, 0, min=0, max=1, unit=''),
    ParDef('file_cals', str, GLOBALS_DIR),
    ParDef('file_parsE', str, GLOBALS_DIR),
    ParDef('file_parsB', str, GLOBALS_DIR),
]

# Create parameter set type (must be named 'ParameterSet')
ParameterSet = namedtuple('ParameterSet', [pd.name for pd in PARDEF])

# Main pulse sequence function, taking the sequencer top level module and a parameter set
def main(seq, par: ParameterSet):

    t_shim_set = 0.1
    dt = 10e-6
    
    # choose tx/rx channel
    tx = seq.tx[par.tx_chan]
    rx_list = list(seq.rx[i] for i in par.rx_chan_list)
    g_shim = np.array(par.offsets[:3])
    # zero gradient constant
    g_ZERO = np.array([0, 0, 0])
    
    t_acq = par.n_samples * par.t_dw

    t_pre_adjust = 0#  -0.5*par.t_ramp


    log.debug(f'g_slice: {str(par.g_slice)}')
    log.debug(f'g_read: {str(par.g_read)}')
    log.debug(f'g_phase: {str(par.g_phase)}')
    
    
    t_phase_read = t_acq/2 - par.t_ramp/2 + t_pre_adjust
    log.debug(f"t_phase_read: {t_phase_read*1e6} us")
    if t_phase_read <= 0:
        raise Exception('increase t_acq: t_phase_read too short')
    
    t_phase_read_180 = (par.t_echo - par.t_90 - par.t_180)/2 - t_phase_read - 5*par.t_ramp - par.t_crush # 3->4
    log.debug(f"preread gradient to 180 time: {t_phase_read_180*1e6} us")
    if t_phase_read_180 <= 0:
        raise Exception('phase gradient time t_phase_read_180 too short')

    t_adjust = (par.t_echo - par.t_180 - t_acq)/2 - 5*par.t_ramp - par.t_phase - par.t_crush # done
    if t_adjust <= 0:
        raise Exception(f'echo time too short t_adjust: {t_adjust}')
        
    repeat_loop = 10 * par.t_ramp + par.t_180 + 2 * par.t_phase + 2 * t_adjust + t_acq + par.t_crush * 2

    start_seq = 4 * par.t_ramp + par.t_90 + t_phase_read + t_phase_read_180

    log.debug(f"TR: {par.TR} s")
    t_end = par.TR - start_seq - (par.n_ETL * repeat_loop) - 2*par.t_ramp - par.t_spoil
    if t_end <= 0:
        raise Exception(f'TR too short, t_end: {t_end}')

    TE_eff = par.t_90/2 + 3*par.t_ramp + t_phase_read + t_phase_read_180 + (repeat_loop * par.n_ETL/2) - t_acq/2 - 3*par.t_ramp - par.t_phase - t_adjust
    log.debug(f"TE_eff: {TE_eff*1e3} ms")
    log.debug(f"t_end: {t_end*1e3} ms")
    
    n_phase_cycle = 4
    phase_cycle_90 = [0, 180, 0, 180]
    phase_cycle_180 = [90, 90, 270, 270]
    
    p_rf_acc = 0 # accumation of RF phase by running at a different frequency
    p_90_inc = 0 # phase accumulated by a 90 pulse (to be calculated)
    p_180_inc = 0 # phase accumulated by a 180 pulse (to be calculated)

    # calculate the the gradient amplitude required to match half the slice gradient area during the read prephasing gradient time
    g_unslice = - ((par.g_slice * 0.5 * (par.t_ramp + par.t_90))/ (par.t_ramp + t_phase_read)) #* par.slice_factor
    log.debug(f'g_unslice: {str(g_unslice)}')
    log.debug(f'n_ETL: {str(par.n_ETL)}')

    log.debug('using softpulse')
    # to offset the slice, the RF pulses must have a frequency offset,
    # which introduces a phase change in the transmitter relative to the receiver,
    # p_90_inc and p_180_inc are the amount of phase accumulated during a 90 and 180 pulse
    p_90_inc = 360*par.f_slice_offset*par.t_90
    p_180_inc = 360*par.f_slice_offset*par.t_180
    log.debug(f'p_90_inc: {p_90_inc}, p_180_inc: {p_180_inc}')

    n_phase = par.n_phase
    n_runs = int(np.ceil(n_phase/par.n_ETL))

    g_phase_array = np.linspace(par.g_phase, -par.g_phase, par.n_phase, endpoint=False)

    g0, t0, rf0, f0, a0, p0 = g_ideal_ramp(g_ZERO, g_shim, par.t_ramp, t_shim_set, dt)
    
    runs_rf = [rf0]
    runs_g = [g0]
    runs_t = [t0]
    runs_f = [f0]
    runs_a = [a0]
    runs_p = [p0]
    
    i_phase_list = []
    
    for i_scan in range(par.n_scans):
        p_rf_acc = p_rf_acc % 360 # prevent growing too large and causing floating point rounding errors
        p_90 = p_acq = phase_cycle_90[i_scan % n_phase_cycle]
        p_180 = phase_cycle_180[i_scan % n_phase_cycle]
        for i_run in range(n_runs):
            
            p_rf_acc += p_90_inc/2

            g1, t1, rf1, f1, a1, p1 = g_ideal(g_shim, g_shim+par.g_slice, par.t_ramp, par.t_90, 0, dt, rf_pulse = "y", pulse_width = par.t_90, pulse_bandwidth = par.bw_90, amp = par.a_90, phase = (p_90-p_rf_acc)%360)
            
            p_rf_acc += p_90_inc/2
            
            g2, t2, rf2, f2, a2, p2 = g_ideal(g_shim, (g_shim+par.g_read+g_unslice), par.t_ramp, t_phase_read, t_phase_read_180, dt, rf_pulse = "n", pulse_width = None, pulse_bandwidth = None, amp = None, phase = None)
            
            echo_rf = []
            echo_g = []
            echo_t = []
            echo_f = []
            echo_a = []
            echo_p = []
            
            for i_echo in range(par.n_ETL):
                i_phase = i_echo * n_runs + i_run
                i_phase_list.append(i_phase)
                if i_phase < n_phase:
                    g_phase_i = g_phase_array[i_phase]
                    # log.debug(f"g_phase_i: {g_phase_i}")
                else:
                    g_phase_i = g_ZERO
                
                p_rf_acc += p_180_inc/2
                    
                g3, t3, rf3, f3, a3, p3 = g_ideal_two(g_shim, g_shim+par.g_crush, g_shim+par.g_slice, par.t_ramp, par.t_crush, par.t_180, 0, dt, rf_pulse = "y", pulse_width = par.t_180, pulse_bandwidth = par.bw_90, amp = par.a_180, phase = (p_180-p_rf_acc)%360)
                
                p_rf_acc += p_180_inc/2
                
                g4, t4, rf4, f4, a4, p4 = g_ideal(g_shim, (g_shim+g_phase_i), par.t_ramp, par.t_phase, t_adjust, dt, rf_pulse = "n", pulse_width = None, pulse_bandwidth = None, amp = None, phase = None)
                g5, t5, rf5, f5, a5, p5 = g_ideal(g_shim, (g_shim+par.g_read), par.t_ramp, t_acq, t_adjust, dt, rf_pulse = "n", pulse_width = None, pulse_bandwidth = None, amp = None, phase = None, acq = "y", p_acq = p_acq)
                g6, t6, rf6, f6, a6, p6 = g_ideal(g_shim, (g_shim-g_phase_i), par.t_ramp, par.t_phase, 0, dt, rf_pulse = "n", pulse_width = None, pulse_bandwidth = None, amp = None, phase = None)
                # HAVEN'T DONE P_ACQ
                echo_g.extend([g3, g4, g5, g6])
                echo_rf.extend([rf3, rf4, rf5, rf6])
                echo_t.extend([t3, t4, t5, t6])
                echo_f.extend([f3, f4, f5, f6])
                echo_a.extend([a3, a4, a5, a6])
                echo_p.extend([p3, p4, p5, p6])
                
            
            g7, t7, rf7, f7, a7, p7 = g_ideal(g_shim, (g_shim+par.g_spoil), par.t_ramp, par.t_spoil, t_end, dt, rf_pulse = "n", pulse_width = None, pulse_bandwidth = None, amp = None, phase = None)
           
        
            runs_rf.extend([rf1, rf2, *echo_rf, rf7])
            runs_g.extend([g1, g2, *echo_g, g7])
            runs_t.extend([t1, t2, *echo_t, t7])
            runs_f.extend([f1, f2, *echo_f, f7])
            runs_a.extend([a1, a2, *echo_a, a7])
            runs_p.extend([p1, p2, *echo_p, p7])

    g8, t8, rf8, f8, a8, p8 = g_ideal_ramp(g_shim, g_ZERO, par.t_ramp, t_shim_set, dt)
    runs_rf.extend([rf8])
    runs_g.extend([g8])
    runs_t.extend([t8])
    runs_f.extend([f8])
    runs_a.extend([a8])
    runs_p.extend([p8])

    g_ideal_all, t_ideal_all, rf_all, f_all, a_all, p_all = combine_g_ideal(runs_g, runs_t, runs_rf, runs_f, runs_a, runs_p)

    
    params_dict = get_exp(file = par.file_parsE)
    params_dict_b0 = get_exp(file = par.file_parsB, b0 = "y")

    fact_perc = 100 # % 
    fact = fact_perc/100
    
    g_preemph_all = add_pre_emph(g_ideal_all, params_dict, factor = fact)
    if par.pre_emphasis == True:
        log.debug(f"B0 uses pre-emphasised gradient")        
        b0_sum = add_b0(g_preemph_all, params_dict_b0, file = par.file_cals, factor = fact)
    else:
        log.debug(f"B0 uses ideal gradient")        
        b0_sum = add_b0(g_ideal_all, params_dict_b0, file = par.file_cals, factor = fact)

    tx = seq.tx[par.tx_chan]
    rx = seq.rx[par.rx_chan]
    
    grad = seq.grad[0]
    # Tx/Rx constant settings
    yield from [tx.freq(par.f) for tx in seq.tx]
    yield rx.freq(par.f)
    yield rx.dwelltime(par.t_dw)
    yield rx.mode(flatfilter=True)
    yield seq.wait(0.01)
    log.debug(f"i_phase_list: {len(i_phase_list)}")
    log.debug(f"g_preemph_all: {len(g_preemph_all)}")

    if par.pre_emphasis == True:
        log.debug("Pre-emphasis ON")
        if par.b0 == True:
            log.debug("B0 ON")
            yield run_seq(seq, tx, 0, rf_all, g_preemph_all, f_all, a_all, p_all, par.f, par.f_slice_offset, par.n_phase, i_phase_list, par.n_samples, rx_list, b0_sum, dt, G_MAX, B0_MAX)
        if par.b0 == False:
            log.debug("B0 OFF")
            yield run_seq(seq, tx, 0, rf_all, g_preemph_all, f_all, a_all, p_all, par.f, par.f_slice_offset, par.n_phase, i_phase_list, par.n_samples, rx_list, None, dt, G_MAX, B0_MAX)
    else:
        log.debug("Pre-emphasis OFF")
        if par.b0 == True:
            log.debug("B0 ON")
            yield run_seq(seq, tx, 0, rf_all, g_ideal_all, f_all, a_all, p_all, par.f, par.f_slice_offset, par.n_phase, i_phase_list, par.n_samples, rx_list,  b0_sum, dt, G_MAX, B0_MAX)
        if par.b0 == False:
            log.debug("B0 OFF")
            yield run_seq(seq, tx, 0, rf_all, g_ideal_all, f_all, a_all, p_all, par.f, par.f_slice_offset, par.n_phase, i_phase_list, par.n_samples, rx_list, None, dt, G_MAX, B0_MAX)
