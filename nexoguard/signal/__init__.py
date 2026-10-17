"""
nexoguard.signal
Signal acquisition, WFDB binary parsing, pre-processing, and dynamic windowing.
"""

from nexoguard.signal.reader import CTGRecord, read_wfdb_record, parse_header_file
from nexoguard.signal.preprocessor import clean_fhr_signal, calculate_signal_loss_pct
from nexoguard.signal.windowing import SignalWindow, slice_signal_windows

__all__ = [
    "CTGRecord",
    "read_wfdb_record",
    "parse_header_file",
    "clean_fhr_signal",
    "calculate_signal_loss_pct",
    "SignalWindow",
    "slice_signal_windows",
]
