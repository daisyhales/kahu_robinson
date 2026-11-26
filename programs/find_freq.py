from matipo.sequence import ParDef, floatarray
from collections import namedtuple
import numpy as np

from matipo.util.kahu_rf_module import pulse
from matipo.util.kahu_drive_module import set_shims

import logging
logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)


# Define input parameters (must be named 'PARDEF')
PARDEF = [
    # args: name, dtype, default value
    ParDef('n_scans', int, 1, min=1, unit=''),
    ParDef('f', float, 1e6, unit='Hz'),
    ParDef('a_90', float, 0, min=0, max=1, unit=''),
    ParDef('t_90', float, 32e-6, unit='s'),
    ParDef('t_dw', float, 1e-6, min=0.1e-6, max=160e-6, unit='s'),
    ParDef('n_samples', int, 1000, min=2, unit=''),
    ParDef('t_acqdelay', float, 1e-6, unit='s'),
    ParDef('t_end', float, 1, unit='s'),
    ParDef('tx_chan', int, 0, min=0, max=1, unit=''),
    ParDef('rx_chan', int, 0, min=0, max=15, unit=''),
    ParDef('offsets', floatarray, np.zeros(4), min=-1, max=1, unit=''),
    ParDef('shims', floatarray, np.zeros(16), min=-1, max=1, unit=''),
]

# Create parameter set type (must be named 'ParameterSet')
ParameterSet = namedtuple('ParameterSet', [pd.name for pd in PARDEF])


# Main pulse sequence function, taking the sequencer top level module and a parameter set
def main(seq, par: ParameterSet):
    t_acq = (par.n_samples) * par.t_dw
    
    n_phase_cycle = 4
    phase_cycle = [0, 180, 90, 270]
    
    # choose tx/rx channel
    tx = seq.tx[par.tx_chan]
    rx = seq.rx[par.rx_chan]

    # set gradient offsets and shim channels
    yield from set_shims(seq, par.offsets, par.shims)
    log.debug(f"offsets: {par.offsets}")
    yield tx.freq(par.f)
    yield rx.freq(par.f)
    yield rx.dwelltime(par.t_dw)
    yield rx.mode(flatfilter=True)
    yield seq.wait(0.1) # allow shims to stabilise
    
    for i_scan in range(par.n_scans):
        p_90 = p_acq = phase_cycle[i_scan % n_phase_cycle]

        yield pulse(seq, tx, p_90, par.a_90, par.t_90)

        yield rx.phase(p_acq)
        yield seq.wait(par.t_acqdelay)
        
        yield rx.acquire(0, par.n_samples)
        yield seq.wait(t_acq)
        
        yield seq.wait(par.t_end)
