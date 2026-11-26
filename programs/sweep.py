from matipo.sequence import ParDef, floatarray
from collections import namedtuple
import numpy as np

import logging
logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)

# Define input parameters (must be named 'PARDEF')
PARDEF = [
    # args: name, dtype, default value
    ParDef('f', float, 1e6, unit='Hz'),
    ParDef('f_bw', float, 1e6, unit='Hz'),
    ParDef('a_sweep', float, 0, min=0, max=1, unit=''),
    ParDef('t_dw', float, 1e-6, min=0.1e-6, max=160e-6, unit='s'),
    ParDef('n_samples', int, 16, min=2, unit=''),
    ParDef('n_sweep', int, 1001, min=1, unit=''),
    ParDef('tx_chan', int, 0, min=0, max=1, unit=''),
    ParDef('loop_forever', bool, False),
]

# Create parameter set type (must be named 'ParameterSet')
ParameterSet = namedtuple('ParameterSet', [pd.name for pd in PARDEF])

# Main pulse sequence function, taking the sequencer top level module and a parameter set
def main(seq, par: ParameterSet):
    log.debug("entering main")
    t_acq = par.n_samples * par.t_dw
    t_pad = 10e-6 + 3*par.t_dw # time to allow between acquisitions and rf updates
    f_array = np.linspace(par.f - par.f_bw/2, par.f + par.f_bw/2, par.n_sweep, endpoint=True)

    tx = seq.tx[par.tx_chan]
    yield tx.amp(par.a_sweep)
    yield tx.enable()
    yield from [rx.dwelltime(par.t_dw) for rx in seq.rx]
    yield from [rx.mode(flatfilter=False) for rx in seq.rx]
    yield seq.wait(t_pad)

    while True:
        for i_sweep,f in enumerate(f_array):
            yield tx.freq(f)
            yield from [rx.freq(f) for rx in seq.rx]
            yield seq.wait(t_pad)
            yield from [rx.acquire(i_rx*par.n_sweep + i_sweep, par.n_samples) for i_rx, rx in enumerate(seq.rx)]
            yield seq.wait(t_acq)
            yield seq.wait(t_pad)

        yield seq.wait(t_pad)
        log.debug("out of for loop")
        if not par.loop_forever:
            break
    
    yield tx.disable()
    yield seq.wait(t_pad)
