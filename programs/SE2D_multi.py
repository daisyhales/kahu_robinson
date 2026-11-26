from matipo.sequence import ParDef, floatarray
from collections import namedtuple
import numpy as np

from matipo.util.kahu_rf_module import pulse_start, pulse_end
from matipo.util.kahu_drive_module import set_shims

import logging
logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(logging.WARNING)

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
    ParDef('t_end', float, 1, unit='s'),
    ParDef('tx_chan', int, 0, min=0, max=1, unit=''),
    ParDef('rx_chan_list', list, [0], unit=''),
    ParDef('g_read', floatarray, (0, 0, 0), min=(-1, -1, -1), max=(1, 1, 1), unit=''),
    ParDef('g_phase', floatarray, (0, 0, 0), min=(-1, -1, -1), max=(1, 1, 1), unit=''),
    ParDef('n_phase', int, 1, min=1, unit=''),
    ParDef('g_spoil', floatarray, (0, 0, 0), min=(-1, -1, -1), max=(1, 1, 1), unit=''),
    ParDef('t_spoil', float, 1e-3, unit='s'),
    ParDef('t_grad_rise', float, 50e-6, unit='s'),
    ParDef('offsets', floatarray, np.zeros(4), min=-1, max=1, unit=''),
    ParDef('shims', floatarray, np.zeros(16), min=-1, max=1, unit=''),
]

# Create parameter set type (must be named 'ParameterSet')
ParameterSet = namedtuple('ParameterSet', [pd.name for pd in PARDEF])

# Main pulse sequence function, taking the sequencer top level module and a parameter set
def main(seq, par: ParameterSet):
    # choose tx/rx channel
    tx = seq.tx[par.tx_chan]
    rx_list = list(seq.rx[i] for i in par.rx_chan_list)
    
    t_acq = (par.n_samples) * par.t_dw
    t_preread = (par.t_grad_rise+t_acq)/2
    t_preread_180 = (par.t_echo - par.t_90 - par.t_180)/2 - t_preread
    log.debug(f"preread gradient to 180 time: {t_preread_180}")
    if t_preread_180 <= 0:
        raise Exception('Echo time too short, or acquisition too long.')
    t_180_acq = (par.t_echo - par.t_180 - t_acq)/2
    log.debug(f"180 to acq time: {t_180_acq}")
    if t_180_acq <= 0:
        raise Exception('Echo time too short, or acquisition too long.')

    g_phase_array = np.linspace(par.g_phase, -par.g_phase, par.n_phase, endpoint=False)

    # zero gradient constant
    g_ZERO = np.array([0, 0, 0])
    
    n_phase_cycle = 8
    phase_cycle_90 = [0, 180, 0, 180, 90, 270, 90, 270]
    phase_cycle_180 = [90, 90, 270, 270, 0, 0, 180, 180]

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
        for i_phase, g_phase in enumerate(g_phase_array):
            p_90 = p_acq = phase_cycle_90[i_scan % n_phase_cycle]
            p_180 = phase_cycle_180[i_scan % n_phase_cycle]
            yield from [rx.phase(p_acq) for rx in rx_list]

            # hard 90 pulse
            yield pulse_start(seq, tx, p_90, par.a_90)
            yield seq.wait(par.t_90)
            yield pulse_end(seq, tx)

            # read prephase gradient
            yield seq.grad[0].vec(*(par.g_read - g_phase))
            yield seq.wait(t_preread)
            yield seq.grad[0].vec(*g_ZERO)

            # echo time matching delay
            yield seq.wait(t_preread_180)

            # hard 180 pulse
            yield pulse_start(seq, tx, p_180, par.a_180)
            yield seq.wait(par.t_180)
            yield pulse_end(seq, tx)

            # echo time matching delay
            yield seq.wait(t_180_acq)

            # read gradient + acquisition
            yield seq.grad[0].vec(*par.g_read)
            yield seq.wait(par.t_grad_rise)
            yield from [rx.phase(p_acq) for rx in rx_list]
            yield from [rx.acquire(i_rx*par.n_phase+i_phase, par.n_samples) for i_rx, rx in enumerate(rx_list)]
            yield seq.wait(t_acq)

            # spoiler gradient
            yield seq.grad[0].vec(*par.g_spoil)
            yield seq.wait(par.t_spoil)
            yield seq.grad[0].vec(*g_ZERO)
            yield seq.wait(par.t_grad_rise)

            # repetition delay
            yield seq.wait(par.t_end)

        # reference noise acquisition
        yield from [rx.acquire(len(rx_list)*par.n_phase + i_rx*par.n_phase, par.n_phase*par.n_samples) for i_rx, rx in enumerate(rx_list)]
        yield seq.wait(par.n_phase*par.n_samples*par.t_dw)
