# Noise acquisition pulse sequence that allows selecting channels to enable

from matipo.sequence import ParDef, floatarray
from collections import namedtuple
import numpy as np

from matipo.util.kahu_drive_module import set_shims

import logging
logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(logging.WARNING)

# Define input parameters (must be named 'PARDEF')
PARDEF = [
    # args: name, dtype, default value
    ParDef('f', float, 1e6, unit='Hz'),
    ParDef('t_dw', float, 1e-6, min=0.1e-6, max=160e-6, unit='s'),
    ParDef('n_samples', int, 1000, min=2, unit=''),
    ParDef('rx_chan', int, 0, min=0, max=15, unit=''),
    ParDef('offsets', floatarray, np.zeros(4), min=-1, max=1, unit=''),
    ParDef('shims', floatarray, np.zeros(16), min=-1, max=1, unit=''),
]

# Create parameter set type (must be named 'ParameterSet')
ParameterSet = namedtuple('ParameterSet', [pd.name for pd in PARDEF])

# Main pulse sequence function, taking the sequencer top level module and a parameter set
def main(seq, par: ParameterSet):
    t_acq = (par.n_samples) * par.t_dw
    t_pad = 10e-6 + 10*par.t_dw # time to allow between acquisitions

    rx = seq.rx[par.rx_chan]

    # set gradient offsets and shim channels
    yield from set_shims(seq, par.offsets, par.shims)
    
    yield rx.freq(par.f)
    yield rx.dwelltime(par.t_dw)
    yield rx.mode(flatfilter=True)
    yield seq.wait(t_pad)
    
    yield rx.acquire(0, par.n_samples)
    yield seq.wait(t_acq)
    
    yield seq.wait(t_pad)
