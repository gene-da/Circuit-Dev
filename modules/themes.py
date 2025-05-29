import matplotlib.pyplot as plt
from cycler import cycler

import matplotlib.pyplot as plt
from cycler import cycler

class PlotTheme:
    def __init__(self, theme: str = "tokyo_night"):
        self.themes = {
            "tokyo_night": {
                # --- Matplotlib rcParams ---
                "figure.facecolor": "#1f2335",
                "axes.facecolor": "#24283b",
                "savefig.facecolor": "#1f2335",
                "text.color": "#c0caf5",
                "axes.labelcolor": "#c0caf5",
                "xtick.color": "#a9b1d6",
                "ytick.color": "#a9b1d6",
                "axes.edgecolor": "#565f89",
                "grid.color": "#3b4261",
                "axes.prop_cycle": cycler('color', [
                    "#7aa2f7", "#9ece6a", "#f7768e", "#e0af68",
                    "#bb9af7", "#7dcfff", "#ff9e64", "#41a6b5",
                    "#89ddff", "#ff007c"
                ]),
                "font.family": "sans-serif",
                "font.size": 11,
                "grid.alpha": 0.3,
                "grid.linestyle": "--",
                "axes.spines.top": False,
                "axes.spines.right": False,
                "legend.facecolor": "#1f2335",
                "legend.edgecolor": "#3b4261",
                "legend.fontsize": 10,
                "errorbar.capsize": 3,
                "image.cmap": "viridis",

                # --- Tag-specific roles ---
                "tag_colors": {
                    "ch1": "#e0af68",    # yellow
                    "ch2": "#7aa2f7",    # blue
                    "ch3": "#bb9af7",    # purple
                    "ch4": "#9ece6a",    # green
                    "math": "#f7768e",   # red
                    "fft": "#7dcfff",     # cyan
                    "pos_fft": "#41a6b5",
                    "neg_fft": "#ff9e64",
                    "marker": "#ff007c"
                }
            },
            "ayu_mirage": {
                "figure.facecolor": "#0f131a",
                "axes.facecolor": "#1c212b",
                "savefig.facecolor": "#1c212b",
                "text.color": "#f8f9fa",
                "axes.labelcolor": "#f8f9fa",
                "xtick.color": "#bfbdb6",
                "ytick.color": "#bfbdb6",
                "axes.edgecolor": "#565b66",
                "grid.color": "#6c7380",
                "axes.prop_cycle": cycler('color', [
                    "#5ccfe6", "#aad94c", "#f07178", "#e6b450",
                    "#d95757", "#39bae6", "#ffaa33", "#4cbf99",
                    "#a37acc", "#ffd173"
                ]),
                "font.family": "sans-serif",
                "font.size": 11,
                "grid.alpha": 0.3,
                "grid.linestyle": "--",
                "axes.spines.top": False,
                "axes.spines.right": False,
                "legend.facecolor": "#1c212b",
                "legend.edgecolor": "#6c7380",
                "legend.fontsize": 10,
                "errorbar.capsize": 3,
                "image.cmap": "viridis",

                "tag_colors": {
                    "ch1": "#e6b450",    # yellow-gold
                    "ch2": "#5ccfe6",    # blue-cyan
                    "ch3": "#a37acc",    # purple
                    "ch4": "#aad94c",    # green
                    "math": "#f07178",   # red
                    "fft": "#39bae6",     # bright cyan
                    "marker": "#d95757"
                }
            }
        }

        # Minimal required for all themes
        self.required_keys = [
            "axes.facecolor",
            "grid.color",
            "text.color",
            "tag_colors"
        ]
        self.required_tags = ["ch1", "ch2", "ch3", "ch4", "math", "fft"]

        self.validate()
        self.apply_theme(theme)

    def validate(self):
        for name, theme in self.themes.items():
            for key in self.required_keys:
                if key not in theme:
                    raise ValueError(f"Theme '{name}' is missing required key: '{key}'")
            for tag in self.required_tags:
                if tag not in theme["tag_colors"]:
                    raise ValueError(f"Theme '{name}' is missing tag color for: '{tag}'")

    def apply_theme(self, theme: str):
        if theme not in self.themes:
            raise ValueError(f"Theme '{theme}' is not defined.")
        
        theme_dict = self.themes[theme]
        valid_keys = set(plt.rcParams.keys())
        
        rc_theme = {k: v for k, v in theme_dict.items() if k in valid_keys}
        plt.rcParams.update(rc_theme)

        self.active = theme
        self.colors = theme_dict["tag_colors"]

