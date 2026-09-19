# -*- coding: utf-8 -*-
from pathlib import Path
import os
import matplotlib.pyplot as plt
from matplotlib import font_manager


def _register_windows_fonts():
    """Register common Windows system fonts when available."""
    candidates = [
        r"C:\Windows\Fonts\times.ttf",
        r"C:\Windows\Fonts\timesbd.ttf",
        r"C:\Windows\Fonts\timesi.ttf",
        r"C:\Windows\Fonts\timesbi.ttf",
    ]
    for path in candidates:
        if os.path.exists(path):
            try:
                font_manager.fontManager.addfont(path)
            except Exception:
                pass


def set_english_font():
    """Prefer Times New Roman for manuscript figures, with safe fallbacks."""
    _register_windows_fonts()
    names = {f.name for f in font_manager.fontManager.ttflist}
    if "Times New Roman" in names:
        chosen = "Times New Roman"
    elif "Times" in names:
        chosen = "Times"
    elif "Liberation Serif" in names:
        chosen = "Liberation Serif"
    else:
        chosen = "DejaVu Serif"

    plt.rcParams.update({
        "font.family": chosen,
        "font.serif": [chosen],
        "mathtext.fontset": "stix",
        "axes.unicode_minus": False,
    })
    return chosen
