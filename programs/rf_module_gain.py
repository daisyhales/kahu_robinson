from matipo.sequence import ParDef, floatarray
from collections import namedtuple
import numpy as np
#from matipo.util.kahu_rf_module import gain_set

RXA_SDO = 0b00100
RXA_CLK = 0b01000
RXA_CSN = 0b10000
SPI_DELAY = 0.5e-6 # slow SPI for long cables

def gain_set(seq, gain):
    gain = max(0, min(15, int(gain/3))) # 3dB steps, round and limit to range [0, 15]
    yield seq.gpo[0].set(RXA_CSN)
    # yield seq.gpo[0].clear(RXA_CLK)
    yield seq.wait(SPI_DELAY)
    yield seq.gpo[0].clear(RXA_CSN)
    yield seq.wait(SPI_DELAY)
    for i in reversed(range(4)): # MSB first
        if gain & (1<<i) != 0:
            yield seq.gpo[0].set(RXA_SDO)
        else:
            yield seq.gpo[0].clear(RXA_SDO)
        yield seq.gpo[0].clear(RXA_CLK) # data change on falling edge
        yield seq.wait(SPI_DELAY)
        yield seq.gpo[0].set(RXA_CLK) # data stable on rising edge
        yield seq.wait(SPI_DELAY)
    yield seq.gpo[0].clear(RXA_CLK | RXA_SDO)
    yield seq.gpo[0].set(RXA_CSN)
    yield seq.wait(SPI_DELAY)

PARDEF = [
    ParDef('gain', int, 21, min=0, max=45, unit='dB')
]

ParameterSet = namedtuple('ParameterSet', [pd.name for pd in PARDEF])

def main(seq, par: ParameterSet):
    yield from gain_set(seq, par.gain)