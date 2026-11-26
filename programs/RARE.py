from matipo.sequence import ParDef, floatarray
from collections import namedtuple
import numpy as np

from matipo.util.kahu_rf_module import pulse_start, pulse_end
from matipo.util.kahu_drive_module import set_shims
from matipo.util.pulseshape import calc_soft_pulse
from matipo.util import etl

import logging
logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)

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

def grad_ramp(seq, g_start, g_end, t, n):
    t_step = t/n
    return seq.join([
        seq.join([seq.grad[0].vec(*v), seq.wait(t_step)])
        for v in np.linspace(g_start, g_end, n)
    ])

# Define input parameters (must be named 'PARDEF')
PARDEF = [
    # args: name, dtype, default value
    ParDef('n_scans', int, 2, min=1, unit=''),
    ParDef('f', float, 1e6, unit='Hz'),
    ParDef('a_90', float, 0, min=0, max=1, unit=''),
    ParDef('t_90', float, 32e-6, unit='s'),
    ParDef('f_slice_offset', float, 0, unit='Hz'),
    ParDef('bw_90', float, 0, unit='Hz'),
    ParDef('a_180', float, 0, min=0, max=1, unit=''),
    ParDef('t_180', float, 32e-6, unit='s'),
    ParDef('t_inv', float, 1e-3, unit='s'),
    ParDef('t_dw', float, 1e-6, min=0.1e-6, max=80e-6, unit='s'),
    ParDef('n_samples', int, 320, min=2, unit=''),
    ParDef('t_read', float, 320e-6, unit='s'),
    ParDef('t_phase', float, 160e-6, unit='s'),
    ParDef('n_phase_1', int, 32, min=1, unit=''),
    ParDef('n_phase_2', int, 32, min=1, unit=''),
    ParDef('n_ETL', int, 0, min=0, unit=''),
    ParDef('t_end', float, 1, unit='s'),
    ParDef('tx_chan', int, 0, min=0, max=1, unit=''),
    ParDef('rx_chan_list', list, [0], unit=''),
    ParDef('g_slice', floatarray, (0, 0, 0), min=(-1, -1, -1), max=(1, 1, 1), unit=''),
    ParDef('g_read', floatarray, (0, 0, 0), min=(-1, -1, -1), max=(1, 1, 1), unit=''),
    ParDef('g_phase_1', floatarray, (0, 0, 0), min=(-1, -1, -1), max=(1, 1, 1), unit=''),
    ParDef('g_phase_2', floatarray, (0, 0, 0), min=(-1, -1, -1), max=(1, 1, 1), unit=''),
    ParDef('t_spoil', float, 1e-3, unit='s'),
    ParDef('g_spoil', floatarray, (0, 0, 0), min=(-1, -1, -1), max=(1, 1, 1), unit=''),
    ParDef('t_grad_ramp', float, 200e-6, unit='s'),
    ParDef('n_grad_ramp', int, 20, min=2, unit=''),
    ParDef('offsets', floatarray, np.zeros(4), min=-1, max=1, unit=''),
    ParDef('shims', floatarray, np.zeros(16), min=-1, max=1, unit=''),
    ParDef('enable_IR', bool, False),
    ParDef('enable_dummy_run', bool, False)
]

# Create parameter set type (must be named 'ParameterSet')
ParameterSet = namedtuple('ParameterSet', [pd.name for pd in PARDEF])

# Main pulse sequence function, taking the sequencer top level module and a parameter set
def main(seq, par: ParameterSet):
    # choose tx/rx channel
    tx = seq.tx[par.tx_chan]
    rx_list = list(seq.rx[i] for i in par.rx_chan_list)
    
    # zero gradient constant
    g_ZERO = np.array([0, 0, 0])
    
    if par.n_phase_1>1:
        g_phase_1_step = -par.g_phase_1/(par.n_phase_1//2)
    else:
        g_phase_1_step = 0
    log.debug(f'phase 1 step: {str(g_phase_1_step)}')
    if par.n_phase_2>1:
         g_phase_2_step = -par.g_phase_2/(par.n_phase_2//2)
    else:
        g_phase_2_step = 0
    log.debug(f'phase 2 step: {str(g_phase_2_step)}')

    # calculate number of runs required
    n_echos = par.n_ETL
    if par.n_ETL==0: # set automatically
        n_echos = par.n_phase_2
    n_phase = par.n_phase_1 * par.n_phase_2
    n_runs = int(np.ceil(n_phase/n_echos))
    
    log.debug(f"n_runs: {n_runs}, n_echos: {n_echos}")

    t_echo = par.t_180 + 4*par.t_grad_ramp + 2*par.t_phase + par.t_read
    
    log.debug(f"echo time: {t_echo}")
    
    t_phase_read = par.t_read/2 - par.t_grad_ramp/2
    log.debug(f"t_phase_read: {t_phase_read*1e6} us")
    t_phase_read_180 = (t_echo - par.t_90 - par.t_180)/2 - t_phase_read - 3*par.t_grad_ramp
    log.debug(f"preread gradient to 180 time: {t_phase_read_180*1e6} us")
    if t_phase_read_180 <= 0:
        raise Exception('phase gradient time too short')
    
    t_acq_delay = (par.t_read - par.n_samples*par.t_dw)/2
    if t_acq_delay <= 0:
        raise Exception('t_read too short!')

    
    
    # time from end of inversion pulse to start of excitation pulse
    t_inv_90 = par.t_inv - (par.t_90+par.t_180)/2
    if t_inv_90 <= 0:
        raise Exception('t_inv too short!')
    
    n_phase_cycle = 4
    phase_cycle_90 = [0, 180, 0, 180]
    phase_cycle_180 = [90, 90, 270, 270]
    
    p_rf_acc = 0 # accumation of RF phase by running at a different frequency
    p_90_inc = 0 # phase accumulated by a 90 pulse (to be calculated)
    p_180_inc = 0 # phase accumulated by a 180 pulse (to be calculated)
    
    enable_softpulse = par.bw_90 > 0
    if enable_softpulse:
        t_inv_90 -= par.t_grad_ramp # keep t_inv setting the time between centres of pulses
        
        # calculate the the gradient amplitude required to match half the slice gradient area during the read prephasing gradient time
        g_unslice = -par.g_slice*0.5*(par.t_90+par.t_grad_ramp)/(t_phase_read+par.t_grad_ramp)
        log.debug(f'g_unslice: {str(g_unslice)}')

        log.debug('using softpulse')
        # to offset the slice, the RF pulses must have a frequency offset,
        # which introduces a phase change in the transmitter relative to the receiver,
        # p_90_inc and p_180_inc are the amount of phase accumulated during a 90 and 180 pulse
        p_90_inc = 360*par.f_slice_offset*par.t_90
        p_180_inc = 360*par.f_slice_offset*par.t_180
        log.debug(f'p_90_inc: {p_90_inc}, p_180_inc: {p_180_inc}')
    else:
        g_unslice = g_ZERO
    
    # pregenerate fixed ramps (optimisation)
    g_ramp_zero_to_slice = grad_ramp(seq, g_ZERO, par.g_slice, par.t_grad_ramp, par.n_grad_ramp)
    g_ramp_slice_to_zero = grad_ramp(seq, par.g_slice, g_ZERO, par.t_grad_ramp, par.n_grad_ramp)
    g_ramp_zero_to_read_unslice = grad_ramp(seq, g_ZERO, par.g_read+g_unslice, par.t_grad_ramp, par.n_grad_ramp)
    g_ramp_read_unslice_to_zero = grad_ramp(seq, par.g_read+g_unslice, g_ZERO, par.t_grad_ramp, par.n_grad_ramp)
    g_ramp_zero_to_read = grad_ramp(seq, g_ZERO, par.g_read, par.t_grad_ramp, par.n_grad_ramp)
    g_ramp_read_to_zero = grad_ramp(seq, par.g_read, g_ZERO, par.t_grad_ramp, par.n_grad_ramp)
    
    # set gradient offsets and shim channels
    yield from set_shims(seq, par.offsets, par.shims)

    # Tx constant settings
    yield tx.freq(par.f)

    # Rx constant settings
    yield from [rx.freq(par.f) for rx in rx_list]
    yield from [rx.dwelltime(par.t_dw) for rx in rx_list]
    yield from [rx.mode(flatfilter=True) for rx in rx_list]
    yield seq.wait(0.1) # allow shims to settle
    
    for i_scan in range(par.n_scans):
        p_rf_acc = p_rf_acc % 360 # prevent growing too large and causing floating point rounding errors
        p_90 = p_acq = phase_cycle_90[i_scan % n_phase_cycle]
        p_180 = phase_cycle_180[i_scan % n_phase_cycle]
        for i_run in range(n_runs):
            if par.enable_IR:
                # inversion pulse
                yield pulse_start(seq, tx, p_rf_acc, par.a_180)
                yield seq.wait(par.t_180)
                yield pulse_end(seq, tx)
                yield seq.wait(t_inv_90)

            if enable_softpulse:
                p_rf_acc += p_90_inc/2
                yield g_ramp_zero_to_slice
                yield tx.freq(par.f+par.f_slice_offset) # set to offset frequency
                yield soft_pulse(seq, tx, (p_90-p_rf_acc)%360, par.a_90, par.t_90, par.bw_90)
                yield tx.freq(par.f) # reset transmitter back to reference frame frequency
                yield g_ramp_slice_to_zero
                p_rf_acc += p_90_inc/2
            else:
                yield pulse_start(seq, tx, p_90, par.a_90)
                yield seq.wait(par.t_90)
                yield pulse_end(seq, tx)
                yield seq.wait(par.t_grad_ramp)

            yield g_ramp_zero_to_read_unslice
            yield seq.wait(t_phase_read)
            yield g_ramp_read_unslice_to_zero
            yield seq.wait(t_phase_read_180)

            for i_echo in range(n_echos):
                # calculate interlaced phase index
                i_phase = i_echo * n_runs + i_run
                if i_phase < n_phase:
                    # calculate gradient
                    i_phase_1 = i_phase % par.n_phase_1
                    i_phase_2 = i_phase // par.n_phase_1
                    g_phase = par.g_phase_1 + i_phase_1*g_phase_1_step + par.g_phase_2 + i_phase_2*g_phase_2_step
                else:
                    # special case when par.n_phase_1*par.n_phase_2 does not divide evenly into par.n_ETL
                    # and there are extra echos
                    i_phase_1 = None
                    i_phase_2 = None
                    g_phase = g_ZERO

                if enable_softpulse:
                    p_rf_acc += p_180_inc/2
                    # refocusing pulse
                    yield g_ramp_zero_to_slice
                    yield tx.freq(par.f+par.f_slice_offset) # set to offset frequency
                    yield pulse_start(seq, tx, (p_180-p_rf_acc)%360, par.a_180)
                    yield seq.wait(par.t_180)
                    yield tx.freq(par.f) # reset transmitter back to reference frame frequency
                    yield pulse_end(seq, tx)
                    yield g_ramp_slice_to_zero
                    p_rf_acc += p_180_inc/2
                else:
                    # refocusing pulse
                    yield pulse_start(seq, tx, p_180, par.a_180)
                    yield seq.wait(par.t_180)
                    yield pulse_end(seq, tx)
                
                # phase blip then apply read gradient
                yield grad_ramp(seq, g_ZERO, g_phase, par.t_grad_ramp, par.n_grad_ramp)
                yield seq.wait(par.t_phase)
                yield grad_ramp(seq, g_phase, par.g_read, par.t_grad_ramp, par.n_grad_ramp)

                # readout
                yield seq.wait(t_acq_delay)
                yield from [rx.phase(p_acq) for rx in rx_list]
                if i_phase_1 is not None:
                    yield from [
                        # indexing: [channel, phase_1, phase_2]
                        rx.acquire(i_rx*par.n_phase_1*par.n_phase_2+i_phase_1*par.n_phase_2+i_phase_2, par.n_samples)
                        for i_rx, rx in enumerate(rx_list)]
                yield seq.wait(par.t_read-t_acq_delay)
                    
                # turn off read gradient and unphase blip
                yield grad_ramp(seq, par.g_read, -g_phase, par.t_grad_ramp, par.n_grad_ramp)
                yield seq.wait(par.t_phase)
                yield grad_ramp(seq, -g_phase, g_ZERO, par.t_grad_ramp, par.n_grad_ramp)
            
            # spoiler pulse
            yield seq.grad[0].vec(*par.g_spoil)
            yield seq.wait(par.t_spoil)
            yield seq.grad[0].vec(*g_ZERO)
            yield seq.wait(2*par.t_grad_ramp)

            # reference noise acquisition
            ref_offset = len(rx_list)*par.n_phase_1*par.n_phase_2
            yield from [rx.acquire(ref_offset + i_rx, par.n_phase_1*par.n_phase_2*par.n_samples) for i_rx, rx in enumerate(rx_list)]
            yield seq.wait(par.n_phase_1*par.n_phase_2*par.n_samples*par.t_dw)

            yield seq.wait(par.t_end)
