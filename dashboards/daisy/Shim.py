import panel as pn

pn.extension()

import sys

print(sys.executable)
print(sys.path)

from dashboards_.shim.shimming import Shim

def main(doc=None):
    shim = Shim()
    return shim()
