import panel as pn

pn.extension()
import sys

print(sys.executable)
print(sys.path)


from dashboards_.fitting_tab.fitting_ec import FitEC

def main(doc=None):
    fit_ec = FitEC()

    return fit_ec()