import panel as pn

pn.extension()
import sys

print(sys.executable)
print(sys.path)

from dashboards_.hard_1DSE.oneDSE import OneDSE


def main(doc=None):
    one = OneDSE()

    tabs = pn.Tabs(
        ("1DSE Hard Pulse", one()),
    )
    
    return tabs