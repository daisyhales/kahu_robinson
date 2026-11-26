from matipo.sequence import ParDef, floatarray
from collections import namedtuple
import numpy as np

import logging
logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(logging.WARNING)

# Define input parameters (must be named 'PARDEF')
PARDEF = [
    # args: name, dtype, default value
    ParDef('f', float, 1e6, unit='Hz'),
    ParDef('t_dw', float, 1e-6, min=0.1e-6, max=160e-6, unit='s'),
    ParDef('n_samples', int, 1000, min=2, unit='')
]

# Create parameter set type (must be named 'ParameterSet')
ParameterSet = namedtuple('ParameterSet', [pd.name for pd in PARDEF])

# Main pulse sequence function, taking the sequencer top level module and a parameter set
def main(seq, par: ParameterSet):
    t_acq = (par.n_samples) * par.t_dw
    t_pad = 10e-6 + 10*par.t_dw # time to allow between acquisitions

    yield from [rx.freq(par.f) for rx in seq.rx]
    yield from [rx.dwelltime(par.t_dw) for rx in seq.rx]
    yield from [rx.mode(flatfilter=True) for rx in seq.rx]
    yield seq.wait(t_pad)
    
    yield from [rx.acquire(i_rx, par.n_samples) for i_rx, rx in enumerate(seq.rx)]
    yield seq.wait(t_acq)
    
    yield seq.wait(t_pad)
