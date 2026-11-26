import panel as pn

pn.extension()
import sys

print(sys.executable)
print(sys.path)

from dashboards_.image_viewer.img_v1 import Viewer1
from dashboards_.image_viewer.img_v2 import Viewer2
from dashboards_.image_viewer.img_v3 import Viewer3


def main(doc=None):
    v1 = Viewer1()
    v2 = Viewer2()
    v3 = Viewer3()
    
    tabs = pn.Tabs(
        ("1D", v1()),
        ("2D", v2()),
        ("3D", v3()),
    )
    
    return tabs

