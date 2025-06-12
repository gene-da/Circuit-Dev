## Oscope Class
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter, ScalarFormatter
from matplotlib.widgets import CheckButtons
import matplotlib.lines as mlines
import mplcursors

from tools import metric_notation as mn

def style_checkbuttons_custom_compatible(check_ax, check,
                                         box_color="#1e1e2e",
                                         check_color="#7aa2f7",
                                         text_color="#c0caf5",
                                         fontweight="bold",
                                         fontsize=12,
                                         marker_size=12):
    # Style the text labels
    for label in check.labels:
        label.set_color(text_color)
        label.set_fontweight(fontweight)
        label.set_fontsize(fontsize)

    # Grab all Line2D objects in the checkbox axes
    for child in check_ax.get_children():
        if isinstance(child, mlines.Line2D):
            child.set_marker("s")
            child.set_markersize(marker_size)
            child.set_color(check_color)
            child.set_markerfacecolor(check_color)
            child.set_markeredgecolor(check_color)

def generic(data, labels=None, title="Generic Plot"):
    plt.style.use('tools/theme/tokyonight.mplstyle')

    fig, ax = plt.subplots(figsize=(12, 6))
    plt.subplots_adjust(left=0.1, right=0.85)

    lines = []
    all_labels = []
    for idx, (x, y) in enumerate(data):
        label = labels[idx] if labels and idx < len(labels) else f"Trace {idx + 1}"
        line, = ax.plot(x, y, label=label)
        lines.append(line)
        all_labels.append(label)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title(title)
    ax.grid(True, which="both", linestyle="--", linewidth=0.5)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda val, _: mn.tm(val, 0)))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda val, _: mn.tm(val, 0)))

    check_ax = fig.add_axes([0.87, 0.3, 0.1, 0.4])
    check = CheckButtons(check_ax, all_labels, [True] * len(lines))

    style_checkbuttons_custom_compatible(
        check_ax, check,
        box_color="#1e1e2e",
        check_color="#bb9af7",
        text_color="#c0caf5",
        fontweight="bold",
        fontsize=12,
        marker_size=14
    )

    def toggle(label):
        for line, name in zip(lines, all_labels):
            if name == label:
                line.set_visible(not line.get_visible())
        plt.draw()

    check.on_clicked(toggle)

    # 🧠 Add mplcursors tooltip support
    cursor = mplcursors.cursor(lines, hover=True)

    @cursor.connect("add")
    def on_hover(sel):
        x, y = sel.target
        sel.annotation.set_text(f"X: {mn.tm(x, 3)}\nY: {mn.tm(y, 3)}")
        sel.annotation.get_bbox_patch().set_alpha(0.9)  # Optional: less transparent
        sel.annotation.get_bbox_patch().set_facecolor("#1e1e2e")
        sel.annotation.get_bbox_patch().set_edgecolor("#7aa2f7")

    ax.legend()
    plt.show()


# def generic_grouped(grouped_data: dict, title="Grouped Plot"):
#     plt.style.use('tools/theme/tokyonight.mplstyle')
#     fig, ax = plt.subplots(figsize=(10, 6))
#     plt.subplots_adjust(left=0.3)

#     lines = []
#     labels = []

#     for label, traces in grouped_data.items():
#         for i, (x, y) in enumerate(traces):
#             trace_label = f"{label} [{i}]" if len(traces) > 1 else label
#             line, = ax.plot(x, y, label=trace_label)
#             lines.append(line)
#             labels.append(trace_label)

#     ax.set_xlabel("Frequency [Hz]")
#     ax.set_ylabel("Response")
#     ax.set_title(title)
#     ax.grid(True, which="both", linestyle="--", linewidth=0.5)
#     ax.xaxis.set_major_formatter(FuncFormatter(lambda val, _: mn.tm(val, 0)))
#     ax.yaxis.set_major_formatter(FuncFormatter(lambda val, _: mn.tm(val, 0)))

#     check_ax = fig.add_axes([0.02, 0.2, 0.25, 0.6])
#     check = CheckButtons(check_ax, labels, [True] * len(lines))
#     style_checkbuttons_basic(check)

#     def toggle(label_clicked):
#         for line, label in zip(lines, labels):
#             if label == label_clicked:
#                 line.set_visible(not line.get_visible())
#         plt.draw()

#     check.on_clicked(toggle)
#     ax.legend()
#     plt.show()
    
# def plot_ac_sweep(data, labels, title, y_limit_high, x_limit_high, y_limit_low=-60, x_limit_low=10):
#     plt.style.use('tools/theme/tokyonight.mplstyle')
#     fig, ax = plt.subplots(figsize=(10, 6))
#     plt.subplots_adjust(left=0.3)

#     lines = []
#     for (freq, trace), label in zip(data, labels):
#         line, = ax.plot(freq, trace, label=label)
#         lines.append(line)

#     ax.set_xscale('log')
#     ax.set_xlim(mn.fm(x_limit_low), mn.fm(x_limit_high))
#     ax.set_ylim(mn.fm(y_limit_low), mn.fm(y_limit_high))
#     ax.set_xlabel("Frequency")
#     ax.set_ylabel("Amplitude (dB)")
#     ax.set_title(title)
#     ax.grid(True, which="both", linestyle='--', linewidth=0.5)
#     ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: mn.tm(x, 0)))
#     ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: mn.tm(y, 0)))

#     check_ax = fig.add_axes([0.02, 0.2, 0.25, 0.6])
#     check = CheckButtons(check_ax, labels, [True] * len(lines))
#     style_checkbuttons_basic(check)

#     def toggle(label_clicked):
#         for line, label in zip(lines, labels):
#             if label == label_clicked:
#                 line.set_visible(not line.get_visible())
#         plt.draw()

#     check.on_clicked(toggle)
#     ax.legend()
#     plt.show()


# def plot_ac_sweep(data, labels, title, y_limit_high, x_limit_high, y_limit_low=-60, x_limit_low=10):
#     plt.style.use(r'tools/theme/tokyonight.mplstyle')
#     plt.figure(figsize=(10, 6))

#     for (freq, trace), label in zip(data, labels):
#         plt.plot(freq, trace, label=label)

#     plt.xscale('log')
#     plt.xlim(mn.fm(x_limit_low), mn.fm(x_limit_high))
#     plt.ylim(mn.fm(y_limit_low), mn.fm(y_limit_high))

#     # Axis labels
#     plt.xlabel("Frequency")
#     plt.ylabel("Amplitude (dB)")
#     plt.title(title)

#     # --- Format x- and y-axis ticks using mn.tm() ---
#     ax = plt.gca()
#     ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: mn.tm(x, 0)))
#     ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: mn.tm(y, 0)))

#     plt.grid(True, which="both", linestyle='--', linewidth=0.5)
#     plt.legend()
#     plt.tight_layout()
#     plt.show()
    



# def transient(data, labels, title="Transient Response",
#               x_center=None, x_divs=10, y_center=None, y_divs=8):
#     if not data or not labels:
#         raise ValueError("Data and labels must be non-empty")

#     if len(data) != len(labels):
#         raise ValueError("Length of data and labels must match")

#     plt.style.use(r"tools/theme/tokyonight.mplstyle")
#     plt.figure(figsize=(10, 6))

#     # Plot all waveforms
#     for (time, signal), label in zip(data, labels):
#         plt.plot(time, signal, label=label)

#     # X-axis: Time
#     time_all = np.concatenate([t for t, _ in data])
#     x_min_raw, x_max_raw = time_all.min(), time_all.max()

#     if x_center is not None:
#         x_center = mn.fm(x_center) if isinstance(x_center, str) else x_center
#         x_step = (x_max_raw - x_min_raw) / x_divs
#         x_min = x_center - (x_divs / 2) * x_step
#         x_max = x_center + (x_divs / 2) * x_step
#     else:
#         x_min, x_max = x_min_raw, x_max_raw
#         x_step = (x_max - x_min) / x_divs

#     plt.xlim(x_min, x_max)
#     plt.xticks(np.linspace(x_min, x_max, x_divs + 1))

#     # Y-axis: Signal
#     signal_all = np.concatenate([s for _, s in data])
#     y_min_raw, y_max_raw = signal_all.min(), signal_all.max()

#     if y_center is not None:
#         y_center = mn.fm(y_center) if isinstance(y_center, str) else y_center
#         y_step = (y_max_raw - y_min_raw) / y_divs
#         y_min = y_center - (y_divs / 2) * y_step
#         y_max = y_center + (y_divs / 2) * y_step
#     else:
#         y_min, y_max = y_min_raw, y_max_raw
#         y_step = (y_max - y_min) / y_divs

#     plt.ylim(y_min, y_max)
#     plt.yticks(np.linspace(y_min, y_max, y_divs + 1))

#     # Labels and final touches
#     plt.xlabel("Time")
#     plt.ylabel("Voltage")
#     plt.title(title)
#     plt.grid(True, which="both", linestyle="--", linewidth=0.5)
#     plt.legend()
#     plt.tight_layout()
#     plt.show()


# debug = False

# class Plotter:
#     def __init__(self, theme=None):
#         if theme is None:
#             path = r'tools/theme/tokyonight.mplstyle'
#             plt.style.use(path)
#             if debug is True:
#                 print(f'Theme Set: {path}')
#         else:
#             path = theme
#             plt.style.use(path)
#             if debug is True:
#                 print(f'Theme Set: {path}')
                
#         self.channel1 = None
#         self.channel2 = None
#         self.channel3 = None
#         self.channel4 = None
#         self.fig = None
#         self.ax = None
#         self.x = None
#         self.ydata = []
#         self.labels = []
#         self.table_data = None
#         self.table_width = 0.3
        
#     def __tm(self, value, precision=2):
#         """_summary_
        
#         Helper function to conver values into metric notation

#         Args:
#             value (_type_): _description_
#             precision (int, optional): _description_. Defaults to 2.

#         Returns:
#             _type_: _description_
#         """
#         if value == 0:
#             return "0"
#         prefixes = {
#             -24: 'y', -21: 'z', -18: 'a', -15: 'f',
#             -12: 'p', -9: 'n', -6: 'µ', -3: 'm',
#             0: '',   3: 'k',  6: 'M',  9: 'G',
#             12: 'T', 15: 'P', 18: 'E', 21: 'Z', 24: 'Y'
#         }
#         import math
#         exponent = int(math.floor(math.log10(abs(value)) // 3 * 3))
#         exponent = max(min(exponent, 24), -24)
#         scaled = value / (10 ** exponent)
#         prefix = prefixes.get(exponent, f"e{exponent}")
#         return f"{scaled:.{precision}f}{prefix}"
                
#     def __round_to_nice_step(self, raw_step):
#         # Round to 1, 2, 5 * 10^n
#         exponent = np.floor(np.log10(raw_step))
#         base = raw_step / (10 ** exponent)
#         if base < 1.5:
#             nice_base = 1
#         elif base < 3.5:
#             nice_base = 2
#         elif base < 7.5:
#             nice_base = 5
#         else:
#             nice_base = 10
#         return nice_base * (10 ** exponent) 
               
#     def lock_grid(self, ax, x_range=None, y_range=None, x_divs=10, y_divs=8, dynamic=True):
#         """
#         Lock plot grid to fixed division count with nicely rounded steps.
#         - Avoids overriding ticks for log-scale axes.
#         - Keeps consistent formatting and gridlines.
#         """

#         # Determine range
#         if isinstance(x_range, (float, int)):
#             x_min, x_max = 0, x_range
#         elif isinstance(x_range, (tuple, list)):
#             x_min, x_max = x_range
#         else:
#             x_min, x_max = ax.get_xlim()

#         if isinstance(y_range, (float, int)):
#             y_min, y_max = -y_range / 2, y_range / 2
#         elif isinstance(y_range, (tuple, list)):
#             y_min, y_max = y_range
#         else:
#             y_min, y_max = ax.get_ylim()

#         # Calculate span and step
#         x_span = x_max - x_min
#         y_span = y_max - y_min
#         raw_x_step = x_span / x_divs
#         raw_y_step = y_span / y_divs

#         x_step = self.__round_to_nice_step(raw_x_step) if dynamic else raw_x_step
#         y_step = self.__round_to_nice_step(raw_y_step) if dynamic else raw_y_step

#         # Set y limits and ticks always
#         y_min = -y_step * (y_divs / 2)
#         y_max = y_step * (y_divs / 2)
#         ax.set_ylim(y_min, y_max)
#         yticks = np.linspace(y_min, y_max, y_divs + 1)
#         ax.set_yticks(yticks)
#         ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: self.__tm(y, 0)))

#         # Handle x-axis: linear or log separately
#         if ax.get_xscale() == 'log':
#             ax.set_xlim(x_min, x_max)
#             ax.xaxis.set_major_locator(LogLocator(base=10.0, numticks=x_divs))
#             ax.xaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1, numticks=12))
#             ax.xaxis.set_minor_formatter(NullFormatter())
#         else:
#             x_min = 0
#             x_max = x_step * x_divs
#             ax.set_xlim(x_min, x_max)
#             xticks = np.linspace(x_min, x_max, x_divs + 1)
#             ax.set_xticks(xticks)
#             ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: self.__tm(x, 0)))

#         # Grid
#         ax.grid(True, which='major', linestyle='--', linewidth=0.75)

        

#     def create_legend(self, ax=None, labels=None, location='upper right', font_size=12, line_width=2.0, spacing=0.3):
#         """
#         Create a custom legend for the Axes object with optional label overrides.

#         Parameters:
#         - ax: Matplotlib Axes
#         - labels: Optional list of label strings (override default trace labels)
#         - location: Legend location string (e.g., 'upper right')
#         - font_size: Font size of legend text
#         - line_width: Line width for legend handles
#         - spacing: Vertical spacing between legend entries
#         """
#         if ax is None:
#             ax = self.ax
#         handles, current_labels = ax.get_legend_handles_labels()
        
#         # Override labels if given
#         if labels is not None:
#             if len(labels) != len(handles):
#                 raise ValueError("Number of custom labels must match number of plotted lines.")
#             current_labels = labels

#         # Customize handles
#         custom_handles = [
#             plt.Line2D([], [], color=h.get_color(), lw=line_width)
#             for h in handles
#         ]

#         legend = ax.legend(
#             custom_handles, current_labels,
#             loc=location,
#             fontsize=font_size,
#             frameon=True,
#             fancybox=True,
#             edgecolor="#2ac3de",   # Match your theme
#             labelspacing=spacing
#         )
#         return legend
    
#     def add_table(self, fig, ax, table_data, table_width=0.3, font_size=None, row_height=0.05):
#         # Normalize input
#         if isinstance(table_data[0], dict):
#             headers = list(table_data[0].keys())
#             rows = [[str(row.get(h, "")) for h in headers] for row in table_data]
#             data = [headers] + rows
#         elif isinstance(table_data[0], list):
#             data = table_data
#         else:
#             raise ValueError("table_data must be a 2D list or list of dicts.")

#         num_rows = len(data)
#         if font_size is None:
#             font_size = max(6, 14 - num_rows)

#         # Adjust layout to leave room for the table
#         fig.subplots_adjust(left=0.05, right=1 - table_width - 0.05)

#         # Estimate compact table height from row count
#         row_padding = 0.02
#         total_height = (row_height + row_padding) * num_rows

#         # Align table to the top-right of the main plot
#         bbox = ax.get_position()
#         x0 = bbox.x1 + 0.02
#         y0 = bbox.y1 - total_height

#         ax_table = fig.add_axes([x0, y0, table_width, total_height])
#         ax_table.axis("off")
        
#         if font_size is None:
#             font_size = max(6, 14 - num_rows)

#         table = ax_table.table(
#             cellText=data,
#             cellLoc='center',
#             loc='center',
#             bbox=[0, 0, 1, 1]
#         )
#         table.auto_set_font_size(False)
#         table.set_fontsize(font_size)

#         for (row, col), cell in table.get_celld().items():
#             cell.set_edgecolor("#3b4261")
#             cell.set_linewidth(1.0)
#             if row == 0:
#                 cell.set_facecolor("#1f2335")
#                 cell.set_text_props(weight='bold', color="#7aa2f7")
#             else:
#                 cell.set_facecolor("#16161e")
#                 cell.set_text_props(color="#c0caf5")
                
#     def set(self, ydata_list, x_axis, labels=None):
#         self.ydata = ydata_list
#         self.x = x_axis
#         self.labels = labels if labels else [f"Trace {i+1}" for i in range(len(ydata_list))]
        
#     def add_table_data(self, table_data, table_width=0.3):
#         self.table_data = table_data
#         self.table_width = table_width
        
#     def create(self, x_range=None, y_range=None, x_divs=10, y_divs=8, view_params=None, log_x=False):
#         self.fig, self.ax = plt.subplots(figsize=(14 + self.table_width, 8))

#         colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
#         for i, y in enumerate(self.ydata):
#             self.ax.plot(self.x, y, label=self.labels[i], color=colors[i % len(colors)])

#         if view_params:
#             self.view(*view_params, x_divs=x_divs, y_divs=y_divs)
#         else:
#             self.lock_grid(self.ax, x_range, y_range, x_divs, y_divs)

#         if log_x:
#             self.ax.set_xscale('log')

#         self.create_legend()
#         if self.table_data:
#             self.add_table(self.fig, self.ax, self.table_data, self.table_width)

#         plt.show()

        
#     def view(self, y_center: float, y_div: float, x_center: float, x_div: float, x_divs: int = 10, y_divs: int = 8):
#         """
#         Override plot view to a specific centered range and tick grid.
#         """
#         if self.ax is None:
#             raise RuntimeError("Plot not created yet. Call plot.create() first.")

#         # Compute bounds
#         y_span = y_div * y_divs
#         y_min = y_center - y_span / 2
#         y_max = y_center + y_span / 2

#         x_span = x_div * x_divs
#         x_min = x_center - x_span / 2
#         x_max = x_center + x_span / 2

#         # Apply view
#         self.ax.set_xlim(x_min, x_max)
#         self.ax.set_ylim(y_min, y_max)

#         self.ax.set_xticks(np.linspace(x_min, x_max, x_divs + 1))
#         self.ax.set_yticks(np.linspace(y_min, y_max, y_divs + 1))
#         self.ax.grid(True, which='major', linestyle='--', linewidth=0.75)

#         self.ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: self.__tm(x, 0)))
#         self.ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: self.__tm(y, 0)))


