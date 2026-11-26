import panel as pn

pn.extension()
import sys

print(sys.executable)
print(sys.path)

from dashboards_._2DSE.twoDSE import TwoDSE


def main(doc=None):
    two = TwoDSE()
    
    return two()

