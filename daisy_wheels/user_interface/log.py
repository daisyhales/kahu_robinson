from matipo import SEQUENCE_DIR, GLOBALS_DIR
from matipo.sequence import Sequence

from .general import general_functions
import logging
import panel as pn

class PanelLogHandler(logging.Handler):
    """
    Logging handler that forwards log records to Panel log pane.
    """
    def __init__(self, general_functions):
        super().__init__()
        self.g = general_functions

    def emit(self, record):
        try:
            msg = self.format(record)
            self.g.log_msg(msg)
        except Exception:
            pass