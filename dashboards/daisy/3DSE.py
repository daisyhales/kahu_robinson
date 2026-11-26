import panel as pn

pn.extension()
import sys

print(sys.executable)
print(sys.path)

from dashboards_._3DSE.threeDSE import ThreeDSE


def main(doc=None):
    three = ThreeDSE()
    
    return three()

