from modules.spice import Spice
from modules.metric_notation import to_metric, from_metric
from modules.themes import PlotTheme
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from dataclasses import dataclass
import numpy as np
import scipy.signal as signal

@dataclass
class Channel:
    name: str
    data: np.ndarray

    def __repr__(self):
        return f"<Channel '{self.name}': {len(self.data)} points>"

    def __add__(self, other):
        if isinstance(other, Channel):
            return Channel(f"({self.name} + {other.name})", self.data + other.data)
        elif isinstance(other, (float, int, np.ndarray)):
            return Channel(f"({self.name} + scalar)", self.data + other)
        return NotImplemented

    def __sub__(self, other):
        if isinstance(other, Channel):
            return Channel(f"({self.name} - {other.name})", self.data - other.data)
        elif isinstance(other, (float, int, np.ndarray)):
            return Channel(f"({self.name} - scalar)", self.data - other)
        return NotImplemented

    def __mul__(self, other):
        if isinstance(other, (float, int, np.ndarray)):
            return Channel(f"({self.name} * scalar)", self.data * other)
        return NotImplemented

    def __truediv__(self, other):
        if isinstance(other, (float, int, np.ndarray)):
            return Channel(f"({self.name} / scalar)", self.data / other)
        return NotImplemented

    def rms(self, reference: str='pp'):
        if reference == 'pp':
            pp = self.peak_to_peak()
            return (1 / (2 * np.sqrt(2))) * pp
        elif reference == 'pk':
            pk = np.max(self.data)
            return (1/np.sqrt(2)) * pk
        elif reference == 'avg':
            avg = np.mean(self.data)
            return (np.pi/(2*np.sqrt(2))) * avg
        else:
            raise ValueError(f'RMS Type not handled: {reference}')

    def peak_to_peak(self):
        return np.max(self.data) - np.min(self.data)

    def mean(self):
        return np.mean(self.data)

    def max(self):
        return np.max(self.data)

    def min(self):
        return np.min(self.data)

class Oscope:
    def __init__(self, sim_path: str, theme: str = 'tokyo_night'):
        self.sim = Spice(sim_path)
        self.sim.parse()
        self.time = self.sim.get_time()
        self.nodes = self.sim.get_nodes()
        self.theme = PlotTheme(theme)
        self.ch1: Channel | None = None
        self.ch2: Channel | None = None
        self.ch3: Channel | None = None
        self.ch4: Channel | None = None
        self.mth: Channel | None = None
        self.fft: Channel | None = None
        prop_cycle = plt.rcParams['axes.prop_cycle']
        colors = prop_cycle.by_key().get('color', [])
        self.colors = {
            "ch1": colors[0] if len(colors) > 0 else "gray",
            "ch2": colors[1] if len(colors) > 1 else "gray",
            "ch3": colors[2] if len(colors) > 2 else "gray",
            "ch4": colors[3] if len(colors) > 3 else "gray",
            "math": colors[4] if len(colors) > 4 else "gray",
            "fft": colors[5] if len(colors) > 5 else "gray",
        }

    def list_signals(self):
        return list(self.nodes.keys())

    def set_channel(self, channel: int, signal: str):
        if signal not in self.nodes:
            raise ValueError(f"Signal '{signal}' not found in simulation.")
        if channel not in [1, 2, 3, 4]:
            raise ValueError("Channel must be 1 through 4.")
        chan = Channel(name=signal, data=self.nodes[signal])
        setattr(self, f"ch{channel}", chan)

    def set_math(self, ch1_invert=False, ch2_invert=False, math: str = "add"):
        if self.ch1 is None or self.ch2 is None:
            raise ValueError("Both ch1 and ch2 must be set before using math.")

        ch1_data = -self.ch1.data if ch1_invert else self.ch1.data
        ch2_data = -self.ch2.data if ch2_invert else self.ch2.data

        if math == "add":
            result_data = ch1_data + ch2_data
            label = f"{'-' if ch1_invert else ''}{self.ch1.name} + {'-' if ch2_invert else ''}{self.ch2.name}"
        elif math == "sub":
            result_data = ch1_data - ch2_data
            label = f"{'-' if ch1_invert else ''}{self.ch1.name} - {'-' if ch2_invert else ''}{self.ch2.name}"
        elif math == "mul":
            result_data = ch1_data * ch2_data
            label = f"{'-' if ch1_invert else ''}{self.ch1.name} * {'-' if ch2_invert else ''}{self.ch2.name}"
        elif math == "div":
            with np.errstate(divide='ignore', invalid='ignore'):
                result_data = np.divide(ch1_data, ch2_data)
                result_data[np.isnan(result_data)] = 0
                result_data[np.isinf(result_data)] = 0
            label = f"{'-' if ch1_invert else ''}{self.ch1.name} / {'-' if ch2_invert else ''}{self.ch2.name}"
        else:
            raise ValueError(f"Unsupported math operation '{math}'")

        self.mth = Channel(name=label, data=result_data)


    def _get_max_voltage(self, signals, x_min, x_max):
        y_max = 0
        for label, mode in signals:
            tag = label.lower()
            channel = getattr(self, tag, None)
            if channel and hasattr(channel, 'data'):
                data = channel.data.copy()
                if mode == "ac":
                    data = data - np.mean(data)
                mask = (self.time >= x_min) & (self.time <= x_max)
                y_max = max(y_max, np.max(np.abs(data[mask])))
        return y_max

    def _choose_voltage_scale(self, peak):
        allowed = [
            50, 20, 10, 5, 2, 1,
            0.5, 0.2, 0.1,
            0.05, 0.02, 0.01,
            0.005, 0.002, 0.001,
            0.0005, 0.0002, 0.0001,
            0.00005, 0.00002, 0.00001
        ]
        for scale in allowed:
            if peak <= scale * 4:
                return scale
        return allowed[-1]
    
    def _create_scope_figure(self):
        fig = plt.figure(figsize=(12, 6))
        ax = fig.add_subplot(1, 1, 1)
        ax.tick_params(axis='x', bottom=True, labelbottom=True)
        ax.tick_params(axis='y', left=True, labelleft=True)
        return fig, None, ax
    
    def _configure_scope_axes(self, ax, x_min, x_max, volts_per_div):
        ax.set_xlim(x_min, x_max)
        ax.set_xticks(np.linspace(x_min, x_max, 11))

        y_range = volts_per_div * 4
        ax.set_ylim(-y_range, y_range)
        ax.set_yticks(np.linspace(-y_range, y_range, 9))
        ax.grid(True, color="#5e6272", linewidth=0.8)

        zero_color = self.theme.colors.get("text", "#c0caf5")
        ax.axhline(0, color=zero_color, linestyle="-", linewidth=1.5, alpha=1.0)
        
    def _plot_signal(self, ax, label, mode, x_min, x_max):
        tag = label.lower()
        tag = "mth" if tag in ["math", "mth"] else tag  # normalize aliases

        channel = getattr(self, tag, None)
        if not channel or not hasattr(channel, 'data'):
            return None

        color = self.theme.colors.get(tag, self.theme.colors.get("math", "gray"))
        marker_color = self.theme.colors.get("marker", "#ff007c")
        data = channel.data.copy()

        if mode == "ac":
            value = channel.peak_to_peak()
            baseline = np.mean(data)
            overlay = [(baseline + value / 2, "+"), (baseline - value / 2, "–")]
            data = data - baseline
            metric = f"{to_metric(value, 2)} pp"

        else:
            value = channel.mean()
            metric = f"{to_metric(value, 2)} avg"
            overlay = [(value, "")]

        mask = (self.time >= x_min) & (self.time <= x_max)
        ax.plot(self.time[mask], data[mask], label=label.upper(), color=color)

        return label.upper(), metric, color, overlay
    
    def _draw_legend_bar(self, legend_ax, items, volts_per_div, time_window):
        legend_ax.set_facecolor(self.theme.colors.get('background', '#1f2335'))
        legend_ax.axis("off")

        scope_str = f"{to_metric(volts_per_div, 0)} V/Div\n{to_metric(time_window, 0)} s/Div"
        items += [("", scope_str, self.theme.colors.get("text", "#c0caf5"))]

        for i, (name, value, color) in enumerate(items):
            text = f"{name}\n{value}" if name else value
            legend_ax.text(
                (i + 0.5) / len(items), 0.5,
                text,
                ha='center', va='center',
                fontsize=10,
                color=color,
                transform=legend_ax.transAxes,
                bbox=dict(
                    facecolor=self.theme.colors.get('background', '#1f2335'),
                    edgecolor=color,
                    linewidth=1,
                    boxstyle="round,pad=0.3"
                )
            )


            
    def plot(self, signals: list[tuple[str, str]], center: str = None, xscale: str = "1", yscale: str = "10m"):
        fig, _, ax = self._create_scope_figure()

        full_time = self.time
        center_val = from_metric(center) if center else full_time[len(full_time) // 2]
        time_window = from_metric(yscale)
        x_min = center_val - (time_window * 5)
        x_max = center_val + (time_window * 5)
        volts_per_div = from_metric(xscale)

        self._configure_scope_axes(ax, x_min, x_max, volts_per_div)

        legend_items = []
        for label, mode in signals:
            tag = label.lower()
            tag = "mth" if tag in ["math", "mth"] else tag
            channel = getattr(self, tag, None)
            if not channel or not hasattr(channel, "data"):
                continue

            color = self.theme.colors.get(tag, self.theme.colors.get("math", "gray"))
            marker_color = self.theme.colors.get("marker", "#ff007c")
            data = channel.data.copy()

            if mode == "ac":
                value = channel.peak_to_peak()
                baseline = np.mean(data)
                overlay = [(baseline + value / 2, "+"), (baseline - value / 2, "–")]
                data = data - baseline
                metric = f"{to_metric(value, 2)} pp"
            else:
                value = channel.mean()
                overlay = [(value, "")]
                metric = f"{to_metric(value, 2)} avg"

            mask = (self.time >= x_min) & (self.time <= x_max)
            ax.plot(self.time[mask], data[mask], label=label.upper(), color=color, linewidth=1.0)

            for level, prefix in overlay:
                ax.axhline(level, color=marker_color, linestyle="--", alpha=0.6, linewidth=1.2)
                ax.text(
                    x_max, level,
                    f"{prefix}{metric}",
                    color=marker_color,
                    fontsize=9,
                    verticalalignment='bottom' if level < ax.get_ylim()[1] else 'top',
                    horizontalalignment='right',
                    backgroundcolor=self.theme.colors.get('background', '#1f2335'),
                    bbox=dict(facecolor=self.theme.colors.get('background', '#1f2335'), alpha=0.8, edgecolor='none', pad=1)
                )

            legend_items.append(f"{label.upper()} ({mode.upper()})\n{metric}")

        # 🔻 Clear axis labels and ticks
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_title("")

        # 🔳 Custom legend
        if legend_items:
            legend_text = "\n\n".join(legend_items)
            ax.text(
                0.01, 0.99,
                legend_text,
                transform=ax.transAxes,
                fontsize=10,
                verticalalignment='top',
                horizontalalignment='left',
                color=self.theme.colors.get("text", "#c0caf5"),
                bbox=dict(
                    facecolor=self.theme.colors.get('background', '#1f2335'),
                    edgecolor=self.theme.colors.get("text", "#c0caf5"),
                    boxstyle="round,pad=0.4",
                    linewidth=1.0
                )
            )

        # 🔳 Scope-style vertical label block (bottom-right)
        ax.text(
            0.99, 0.01,
            f"{to_metric(volts_per_div, 0)} V/Div\n{to_metric(time_window, 0)} s/Div",
            transform=ax.transAxes,
            fontsize=9,
            verticalalignment='bottom',
            horizontalalignment='right',
            color=self.theme.colors.get("text", "#c0caf5"),
            bbox=dict(
                facecolor=self.theme.colors.get('background', '#1f2335'),
                edgecolor=self.theme.colors.get("text", "#c0caf5"),
                boxstyle="round,pad=0.3",
                linewidth=1.0
            )
        )

        # Keep the plot square
        x_tick_count = 10
        y_tick_count = 8
        aspect_ratio = x_tick_count / y_tick_count
        base_height = 6
        fig.set_size_inches(aspect_ratio * base_height, base_height)

        plt.tight_layout()
        plt.show()


        
    def set_fft(self, channel: str = "ch1", window_type: str = "hann"):
        ch = getattr(self, channel)
        if ch is None:
            raise ValueError(f"{channel} must be set before performing FFT.")

        y = ch.data
        t = self.time

        n_raw = min(len(y), len(t))

        # Clamp to next lower power of 2 for clean FFT
        n = 2 ** int(np.log2(n_raw))
        y = y[:n]
        t = t[:n]

        if n < 2:
            raise ValueError("Not enough data points to perform FFT.")

        # Remove DC offset
        y = y - np.mean(y)

        # Optional windowing
        if window_type == "hann":
            try:
                window = signal.windows.hann(n)
            except AttributeError:
                window = signal.hann(n)  # legacy fallback
            y_win = y * window
        elif window_type == "none":
            y_win = y
        else:
            raise ValueError(f"Unsupported window type: {window_type}")

        # Sampling rate
        dt = t[1] - t[0]
        fs = 1.0 / dt
        df = fs / n

        print(f"[FFT] Fs = {fs:.2f} Hz | dt = {dt:.2e} s | N = {n} | df = {df:.2f} Hz/bin")

        # FFT
        fft_data = np.fft.fft(y_win)
        fft_freq = np.fft.fftfreq(n, d=dt)

        half_n = n // 2
        freqs = fft_freq[:half_n]
        mags = 20 * np.log10(np.abs(fft_data[:half_n]) * 2 / n + 1e-12)  # dB
        # mags -= np.min(mags)
        
        # Normalize: peak becomes 100 dB
        mags -= np.max(mags)
        mags += 100

        # Print top 6 peaks
        top_indices = np.argsort(mags)[-6:][::-1]
        print("\nTop FFT Peaks:")
        for idx in top_indices:
            print(f"  {freqs[idx]:.2f} Hz : {mags[idx]:.2f} dB")

        self.fft = Channel(name=f"FFT({ch.name})", data=np.column_stack((freqs, mags)))


        
    def plot_fft(self, center: str = None, freq_per_div: str = "10k", mag_per_div: str = "10", use_log_freq: bool = False):
        if self.fft is None or not hasattr(self.fft, 'data'):
            raise ValueError("FFT data not available. Call set_fft() first.")

        freq = self.fft.data[:, 0]
        magnitude = self.fft.data[:, 1]

        if len(freq) == 0 or len(magnitude) == 0:
            raise ValueError("FFT data appears to be empty.")

        fig, legend_ax, ax = self._create_scope_figure()

        # Preserve original values for coloring logic
        mags_raw = magnitude.copy()

        # Shift all magnitudes so 0 is bottom of Y-axis
        magnitude -= np.min(magnitude)

        # Frequency centering and scaling
        center_val = from_metric(center) if center else np.median(freq)
        freq_div = from_metric(freq_per_div)
        x_min = center_val - freq_div * 5
        x_max = center_val + freq_div * 5
        mask = (freq >= x_min) & (freq <= x_max)
        freq = freq[mask]
        magnitude = magnitude[mask]
        mags_raw = mags_raw[mask]

        ax.set_xlim(x_min, x_max)
        ax.set_xticks(np.linspace(x_min, x_max, 11))

        # Vertical scale (always 0-bottom)
        mag_div = from_metric(mag_per_div)
        y_range = mag_div * 8  # 8 divisions vertically
        ax.set_ylim(0, y_range)
        ax.set_yticks(np.linspace(0, y_range, 9))

        # Conditional colors: above or below 0 dB (pre-shifted)
        colors = np.where(
            mags_raw >= 0,
            self.theme.colors.get("pos_fft", "#41a6b5"),
            self.theme.colors.get("neg_fft", "#ff9e64")
        )

        # Draw vertical bars
        for f, m, c in zip(freq, magnitude, colors):
            ax.vlines(f, 0, m, color=c, linewidth=1.0)

        # Labels and styling
        ax.set_xlabel("Frequency (Hz)", color=self.theme.colors.get("text", "#c0caf5"))
        ax.set_ylabel("Magnitude (dB)", color=self.theme.colors.get("text", "#c0caf5"))
        ax.tick_params(axis='x', colors=self.theme.colors.get("text", "#c0caf5"))
        ax.tick_params(axis='y', colors=self.theme.colors.get("text", "#c0caf5"))
        ax.grid(True, color=self.theme.colors.get("grid", "#3b4261"), linestyle="--", alpha=0.3)
        ax.axhline(0, color=self.theme.colors.get("text", "#c0caf5"), linestyle="-", linewidth=1.5, alpha=1.0)

        # Aspect ratio tuning
        x_tick_count = len(ax.get_xticks()) - 1
        y_tick_count = len(ax.get_yticks()) - 1
        aspect_ratio = x_tick_count / y_tick_count if y_tick_count else 1
        base_height = 6
        fig.set_size_inches(aspect_ratio * base_height, base_height)

        plt.tight_layout()
        plt.show()



        
    def detect_fft_peak(self):
        if not self.fft:
            raise ValueError("FFT data not set. Run set_fft() first.")
        freqs = self.fft.data[:, 0]
        mags = self.fft.data[:, 1]
        idx = np.argmax(mags)
        return freqs[idx], mags[idx]
    
    def get_channel_freq(self, channel: str = "ch1"):
        """
        Estimate the dominant frequency in the signal assigned to the given channel.
        Returns (freq_hz, peak_db).
        """
        ch = getattr(self, channel, None)
        if ch is None or not hasattr(ch, "data"):
            raise ValueError(f"Channel '{channel}' not found or not set.")

        y = ch.data
        t = self.time

        n = min(len(y), len(t))
        n = 2 ** int(np.log2(n))  # power-of-2 FFT length
        y = y[:n]
        t = t[:n]

        # Remove DC
        y = y - np.mean(y)

        # FFT
        dt = t[1] - t[0]
        fs = 1.0 / dt
        fft_data = np.fft.fft(y * signal.windows.hann(n))
        fft_freq = np.fft.fftfreq(n, d=dt)

        half_n = n // 2
        freqs = fft_freq[:half_n]
        mags = 20 * np.log10(np.abs(fft_data[:half_n]) * 2 / n + 1e-12)

        idx = np.argmax(mags)
        return freqs[idx], mags[idx]











