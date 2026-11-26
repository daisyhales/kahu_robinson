import panel as pn

pn.extension()
import sys

print(sys.executable)
print(sys.path)

from dashboards_.compiler.compile_fit_pars import CompileFit

def main(doc=None):
    comp_fit = CompileFit()

    return comp_fit()