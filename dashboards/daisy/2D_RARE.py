import panel as pn

pn.extension()
import sys

print(sys.executable)
print(sys.path)

from dashboards_.RARE2D.rare import Rare2D


def main(doc=None):
    rare = Rare2D()
    
    return rare()
