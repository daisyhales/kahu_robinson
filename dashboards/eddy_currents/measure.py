import panel as pn

pn.extension()
import sys

print(sys.executable)
print(sys.path)

from dashboards_.eddy_measure.measure_ec import MeasureEddyCurrents

def main(doc=None):
    m_ec = MeasureEddyCurrents()

    return m_ec()