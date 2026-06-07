#!/usr/bin/env python3
"""Interactive sector/angle backscatter balancer with Lambertian normalization.

Reads the CSV output from ``mbes-bs-table`` (``mbes_tools.backscatter.table``)
and lets you drag sector/swath curves and auto-apply a first-pass Lambertian
normalization. The headless math lives in
:mod:`mbes_tools.backscatter.normalize`; this module is the Tk presentation
layer plus calibration-file / KMALL-patch CSV export.

Expected input CSV columns:
    depthMode, rxFanIndex, sector, beam_angle_deg, avgIntensity_dB

Indexing convention:
    CSV depthMode is shifted +1 when loaded.
    CSV sector is shifted +1 when loaded.

Depth-mode labels after shift: 2=Shallow, 3=Medium, 4=Deep, 5=Deeper,
6=Very Deep, 7=Extra Deep, 8=Extreme Deep.

Usage:
    mbes-bs-gui                      # file dialog to pick CSV
    mbes-bs-gui path/to/table.csv    # load directly
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from mbes_tools.backscatter.normalize import (
    DEPTH_MODE_LABELS,
    fit_lambert_intercept,
    lambert_reference_curve,
    solve_sector_shifts,
    valid_fans_for_mode,
)


REQUIRED_COLUMNS = {"depthMode", "rxFanIndex", "sector", "beam_angle_deg", "avgIntensity_dB"}


class FineTuneApp:
    def __init__(
        self,
        master: tk.Tk,
        df: pd.DataFrame,
        unique_modes: np.ndarray,
        unique_fans: np.ndarray,
    ):
        self.master = master
        self.master.title("Sector Intensity Fine-Tuning GUI")

        self.df = df.copy()
        self.unique_modes = unique_modes
        self.unique_fans = unique_fans

        self.adjustments: dict[tuple[int, int, int], float] = {}
        self.swath_adjustments: dict[tuple[int, int], float] = {}
        self.display_modes = {
            (int(m), int(f)): tk.BooleanVar(value=True)
            for m in self.unique_modes
            for f in self.valid_fans_for_mode(int(m))
        }
        self.dragging = None

        self.y_min = tk.DoubleVar(value=-100.0)
        self.y_max = tk.DoubleVar(value=50.0)

        self.lambert_min_angle = tk.DoubleVar(value=0.0)
        self.lambert_max_angle = tk.DoubleVar(value=75.0)
        self.lambert_clip_shift = tk.DoubleVar(value=15.0)
        self.show_lambert_reference = tk.BooleanVar(value=False)

        self.main_frame = tk.Frame(master)
        self.main_frame.pack(fill=tk.BOTH, expand=True)

        self.left_frame = tk.Frame(self.main_frame)
        self.left_frame.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)

        self.right_frame = tk.Frame(self.main_frame)
        self.right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        self.mode_label_map = {
            str(int(m)): DEPTH_MODE_LABELS.get(int(m), str(int(m)))
            for m in self.unique_modes
        }

        self.selected_mode = tk.StringVar()
        self.selected_mode.set(self.mode_label_map[str(int(self.unique_modes[0]))])

        tk.Label(self.left_frame, text="Select Mode to Adjust:").pack(anchor="w")
        self.mode_dropdown = ttk.Combobox(
            self.left_frame,
            textvariable=self.selected_mode,
            values=[self.mode_label_map[str(int(m))] for m in self.unique_modes],
            state="readonly",
            width=18,
        )
        self.mode_dropdown.bind("<<ComboboxSelected>>", self.update_plot)
        self.mode_dropdown.pack(anchor="w", pady=(0, 8))

        tk.Label(self.left_frame, text="Y-axis limits:").pack(anchor="w")
        y_frame = tk.Frame(self.left_frame)
        y_frame.pack(anchor="w")
        tk.Label(y_frame, text="Min:").pack(side=tk.LEFT)
        tk.Entry(y_frame, textvariable=self.y_min, width=8).pack(side=tk.LEFT)
        tk.Label(y_frame, text="Max:").pack(side=tk.LEFT)
        tk.Entry(y_frame, textvariable=self.y_max, width=8).pack(side=tk.LEFT)
        tk.Button(self.left_frame, text="Apply Y Scale", command=self.update_plot).pack(
            anchor="w", pady=(2, 8)
        )

        tk.Label(self.left_frame, text="Lambert normalize settings:").pack(anchor="w")
        lambert_frame1 = tk.Frame(self.left_frame)
        lambert_frame1.pack(anchor="w")
        tk.Label(lambert_frame1, text="Angle min:").pack(side=tk.LEFT)
        tk.Entry(lambert_frame1, textvariable=self.lambert_min_angle, width=6).pack(side=tk.LEFT)
        tk.Label(lambert_frame1, text="max:").pack(side=tk.LEFT)
        tk.Entry(lambert_frame1, textvariable=self.lambert_max_angle, width=6).pack(side=tk.LEFT)

        lambert_frame2 = tk.Frame(self.left_frame)
        lambert_frame2.pack(anchor="w")
        tk.Label(lambert_frame2, text="Max shift ±dB:").pack(side=tk.LEFT)
        tk.Entry(lambert_frame2, textvariable=self.lambert_clip_shift, width=6).pack(side=tk.LEFT)

        tk.Checkbutton(
            self.left_frame,
            text="Show Lambert reference",
            variable=self.show_lambert_reference,
            command=self.update_plot,
        ).pack(anchor="w", pady=(2, 2))

        tk.Button(
            self.left_frame,
            text="Auto Lambert Normalize Selected Mode",
            command=self.auto_lambert_normalize_selected_mode,
        ).pack(anchor="w", pady=(2, 8))

        tk.Label(self.left_frame, text="Modes/Fans to Display:").pack(anchor="w")
        for m in self.unique_modes:
            for f in self.valid_fans_for_mode(int(m)):
                chk = tk.Checkbutton(
                    self.left_frame,
                    text=f"{DEPTH_MODE_LABELS.get(int(m), str(int(m)))} Fan {int(f)}",
                    variable=self.display_modes[(int(m), int(f))],
                    command=self.update_plot,
                )
                chk.pack(anchor="w")

        tk.Button(
            self.left_frame,
            text="Export Calibration File",
            command=self.export_calibration,
        ).pack(anchor="w", pady=(8, 2))

        tk.Button(
            self.left_frame,
            text="Export KMALL Patch Correction CSV",
            command=self.export_fine_tuned_csv,
        ).pack(anchor="w", pady=(2, 2))

        self.fig, self.ax = plt.subplots(figsize=(9, 7))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.right_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.canvas.mpl_connect("button_press_event", self.on_press)
        self.canvas.mpl_connect("button_release_event", self.on_release)
        self.canvas.mpl_connect("motion_notify_event", self.on_motion)

        self.update_plot()

    def valid_fans_for_mode(self, mode: int) -> np.ndarray:
        return valid_fans_for_mode(int(mode), self.unique_fans)

    def selected_mode_int(self) -> int:
        mode_label = self.mode_dropdown.get()
        return [int(k) for k, v in self.mode_label_map.items() if v == mode_label][0]

    def adjusted_values_for_subset(
        self, sub: pd.DataFrame, mode: int, fan: int
    ) -> np.ndarray:
        intensities = sub["avgIntensity_dB"].astype(float).to_numpy().copy()
        sectors = sub["sector"].astype(int).to_numpy()
        intensities += self.swath_adjustments.get((int(mode), int(fan)), 0.0)
        for sector in np.unique(sectors):
            sector_i = int(sector)
            intensities[sectors == sector_i] += self.adjustments.get(
                (int(mode), int(fan), sector_i), 0.0
            )
        return intensities

    def plot_lambert_reference_for_selected_mode(self):
        if not self.show_lambert_reference.get():
            return

        selected_mode = self.selected_mode_int()

        try:
            min_ang = float(self.lambert_min_angle.get())
            max_ang = float(self.lambert_max_angle.get())
        except Exception:
            min_ang, max_ang = 0.0, 75.0

        for fan in self.valid_fans_for_mode(selected_mode):
            fan_i = int(fan)
            if not self.display_modes.get(
                (selected_mode, fan_i), tk.BooleanVar(value=False)
            ).get():
                continue

            sub = self.df[
                (self.df["depthMode"] == selected_mode) & (self.df["rxFanIndex"] == fan_i)
            ].copy()

            if sub.empty:
                continue

            angles = sub["beam_angle_deg"].astype(float).to_numpy()
            adjusted = self.adjusted_values_for_subset(sub, selected_mode, fan_i)
            intercept = fit_lambert_intercept(angles, adjusted, min_ang, max_ang)
            if intercept is None:
                continue

            angle_min = float(np.nanmin(angles))
            angle_max = float(np.nanmax(angles))
            angle_grid = np.linspace(angle_min, angle_max, 301)
            reference = lambert_reference_curve(angle_grid, intercept)

            self.ax.plot(
                angle_grid,
                reference,
                linestyle="--",
                linewidth=2.0,
                color="black",
                alpha=0.85,
            )

    def update_plot(self, event=None):
        self.ax.clear()
        selected_mode = self.selected_mode_int()
        self.points = []

        for mode in self.unique_modes:
            mode_i = int(mode)
            for fan in self.valid_fans_for_mode(mode_i):
                fan_i = int(fan)
                if not self.display_modes.get(
                    (mode_i, fan_i), tk.BooleanVar(value=False)
                ).get():
                    continue

                swath_adj = self.swath_adjustments.get((mode_i, fan_i), 0.0)
                sub_mf = self.df[
                    (self.df["depthMode"] == mode_i) & (self.df["rxFanIndex"] == fan_i)
                ]
                if sub_mf.empty:
                    continue

                for sector in sorted(np.unique(sub_mf["sector"])):
                    sector_i = int(sector)
                    sub = sub_mf[sub_mf["sector"] == sector_i]
                    if sub.empty:
                        continue

                    angles = sub["beam_angle_deg"].astype(float)
                    orig_vals = sub["avgIntensity_dB"].astype(float)
                    sector_adj = self.adjustments.get((mode_i, fan_i, sector_i), 0.0)
                    adj_vals = orig_vals + sector_adj + swath_adj
                    self.adjustments[(mode_i, fan_i, sector_i)] = sector_adj

                    color = "red" if mode_i == selected_mode else None
                    lw = 2.5 if mode_i == selected_mode else 1.0
                    self.ax.plot(angles, adj_vals, color=color, linewidth=lw)

                    mean_angle = float(np.nanmean(angles))
                    mean_val = float(np.nanmean(adj_vals))
                    self.points.append({
                        "mode": mode_i,
                        "fan": fan_i,
                        "sector": sector_i,
                        "x": mean_angle,
                        "y": mean_val,
                        "swath": False,
                    })
                    self.ax.plot(mean_angle, mean_val, "o", color="black")

                mean_angle_swath = float(np.nanmean(sub_mf["beam_angle_deg"].astype(float)))
                mean_val_swath = (
                    float(np.nanmean(sub_mf["avgIntensity_dB"].astype(float))) + swath_adj
                )
                self.points.append({
                    "mode": mode_i,
                    "fan": fan_i,
                    "sector": None,
                    "x": mean_angle_swath,
                    "y": mean_val_swath,
                    "swath": True,
                })
                self.ax.plot(mean_angle_swath, mean_val_swath, "s", color="blue")

        self.plot_lambert_reference_for_selected_mode()

        self.ax.set_title("Fine-Tune Sector Backscatter Correction")
        self.ax.set_xlabel("Beam Angle (deg)")
        self.ax.set_ylabel("Backscatter / adjusted intensity (dB)")

        try:
            ymin = float(self.y_min.get())
            ymax = float(self.y_max.get())
            if ymin == ymax:
                ymin, ymax = -100.0, 50.0
            self.ax.set_ylim(ymin, ymax)
        except Exception:
            self.ax.set_ylim(-100.0, 50.0)

        self.ax.grid(True)
        self.canvas.draw()

    def on_press(self, event):
        if event.inaxes != self.ax:
            return
        closest = None
        min_dist = float("inf")
        for p in self.points:
            dist = np.hypot(event.xdata - p["x"], event.ydata - p["y"])
            if dist < min_dist and dist < 5:
                min_dist = dist
                closest = p
        self.dragging = closest

    def on_motion(self, event):
        if self.dragging is None or event.inaxes != self.ax:
            return
        mode = int(self.dragging["mode"])
        fan = int(self.dragging["fan"])
        delta = event.ydata - self.dragging["y"]
        if self.dragging.get("swath", False):
            self.swath_adjustments[(mode, fan)] = (
                self.swath_adjustments.get((mode, fan), 0.0) + delta
            )
        else:
            sector = int(self.dragging["sector"])
            self.adjustments[(mode, fan, sector)] = (
                self.adjustments.get((mode, fan, sector), 0.0) + delta
            )
        self.dragging["y"] = event.ydata
        self.update_plot()

    def on_release(self, event):
        self.dragging = None

    def auto_lambert_normalize_selected_mode(self):
        selected_mode = self.selected_mode_int()
        try:
            min_ang = float(self.lambert_min_angle.get())
            max_ang = float(self.lambert_max_angle.get())
            max_shift = abs(float(self.lambert_clip_shift.get()))
        except Exception:
            messagebox.showerror("Lambert Normalize", "Invalid Lambert settings.")
            return

        if min_ang < 0:
            min_ang = 0.0
        if max_ang <= min_ang:
            messagebox.showerror(
                "Lambert Normalize", "Angle max must be greater than angle min."
            )
            return

        applied_rows = []

        for fan in self.valid_fans_for_mode(selected_mode):
            fan_i = int(fan)
            sub = self.df[
                (self.df["depthMode"] == selected_mode) & (self.df["rxFanIndex"] == fan_i)
            ].copy()
            if sub.empty:
                continue

            angles = sub["beam_angle_deg"].astype(float).to_numpy()
            adjusted = self.adjusted_values_for_subset(sub, selected_mode, fan_i)
            sectors = sub["sector"].astype(int).to_numpy()

            shifts = solve_sector_shifts(
                angles, adjusted, sectors,
                min_angle_deg=min_ang,
                max_angle_deg=max_ang,
                max_shift_db=max_shift,
            )

            for sector_i, sector_shift in shifts.items():
                key = (selected_mode, fan_i, sector_i)
                self.adjustments[key] = self.adjustments.get(key, 0.0) + sector_shift
                applied_rows.append((selected_mode, fan_i, sector_i, sector_shift))

        self.update_plot()

        if applied_rows:
            msg_lines = ["Applied Lambertian sector shifts:"]
            for mode, fan, sector, shift in applied_rows[:30]:
                msg_lines.append(f"Mode {mode}, Fan {fan}, Sector {sector}: {shift:+.2f} dB")
            if len(applied_rows) > 30:
                msg_lines.append(f"...and {len(applied_rows) - 30} more")
            messagebox.showinfo("Lambert Normalize", "\n".join(msg_lines))
        else:
            messagebox.showwarning(
                "Lambert Normalize",
                "No valid sector shifts were computed for the selected mode.",
            )

    def export_calibration(self):
        save_path = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Text files", ".txt")],
            title="Save Adjusted Calibration File",
        )
        if not save_path:
            return

        template_path = filedialog.askopenfilename(
            title="Select Example Calibration File",
            filetypes=[("Text files", ".txt")],
        )
        if not template_path:
            return

        with open(template_path, "r") as f:
            template_lines = f.readlines()

        mode_header_re = re.compile(r"^\s*(\d+)\s+(\d+)\s+(\d+)\s*$")
        sector_header_re = re.compile(r"^# TX sector\s+(\d+):")
        count_re = re.compile(r"^\s*(\d+)\s*$")
        angle_line_re = re.compile(r"^\s*(-?\d+)\s+[-\d\.]+")

        out_lines = []
        current_mode = None
        current_fan = None
        current_sector = None

        for line in template_lines:
            m_h = mode_header_re.match(line)
            if m_h:
                current_mode, current_fan, _ = map(int, m_h.groups())
                out_lines.append(line)
                continue

            s_h = sector_header_re.match(line)
            if s_h:
                current_sector = int(s_h.group(1))
                out_lines.append(line)
                continue

            c_h = count_re.match(line)
            if c_h and current_sector is not None:
                out_lines.append(line)
                continue

            a_h = angle_line_re.match(line)
            if a_h and None not in (current_mode, current_fan, current_sector):
                angle = int(a_h.group(1))
                swath_adj = self.swath_adjustments.get((current_mode, current_fan), 0.0)
                sector_adj = self.adjustments.get(
                    (current_mode, current_fan, current_sector), 0.0
                )
                out_lines.append(f"{angle} {swath_adj + sector_adj:.2f}\n")
                continue

            out_lines.append(line)

        with open(save_path, "w") as f:
            f.writelines(out_lines)

        try:
            subprocess.run(["unix2dos", save_path], check=True)
        except Exception as e:
            messagebox.showwarning(
                "Conversion Warning",
                f"Calibration file was written, but unix2dos failed:\n\n{e}",
            )

        messagebox.showinfo("Export Complete", f"Calibration file saved to {save_path}")

    def export_fine_tuned_csv(self):
        rows = []
        for mode in self.unique_modes:
            mode_i = int(mode)
            for fan in self.valid_fans_for_mode(mode_i):
                fan_i = int(fan)
                sub_mf = self.df[
                    (self.df["depthMode"] == mode_i) & (self.df["rxFanIndex"] == fan_i)
                ]
                if sub_mf.empty:
                    continue
                for sector in sorted(np.unique(sub_mf["sector"])):
                    sector_i = int(sector)
                    swath_adj = self.swath_adjustments.get((mode_i, fan_i), 0.0)
                    sector_adj = self.adjustments.get((mode_i, fan_i, sector_i), 0.0)
                    rows.append({
                        "depthMode": mode_i,
                        "rxFanIndex": fan_i,
                        "txSectorNumb": sector_i,
                        "correction_dB": float(swath_adj + sector_adj),
                    })

        out_df = pd.DataFrame(
            rows, columns=["depthMode", "rxFanIndex", "txSectorNumb", "correction_dB"]
        )

        save_path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", ".csv")],
            title="Save KMALL Patch Correction CSV",
        )
        if not save_path:
            return

        out_df.to_csv(save_path, index=False)
        messagebox.showinfo(
            "Export Complete", f"KMALL patch correction CSV saved to {save_path}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Interactive sector/angle backscatter balancing GUI with Lambertian normalization."
    )
    parser.add_argument(
        "csv",
        nargs="?",
        default=None,
        help="Sector/angle intensity CSV from mbes-bs-table. "
             "Opens a file dialog if not provided.",
    )
    args = parser.parse_args()

    # Resolve the input CSV — CLI arg or file dialog.
    if args.csv:
        file_path = args.csv
    else:
        # A hidden root is needed to drive the file dialog before the main window exists.
        _temp = tk.Tk()
        _temp.withdraw()
        file_path = filedialog.askopenfilename(
            title="Select Sector/Angle Intensity CSV",
            filetypes=[("CSV files", "*.csv")],
        )
        _temp.destroy()

    if not file_path:
        print("No file selected. Exiting.")
        sys.exit(0)

    raw_df = pd.read_csv(file_path)

    missing = REQUIRED_COLUMNS - set(raw_df.columns)
    if missing:
        print(f"Error: input CSV is missing required columns: {sorted(missing)}", file=sys.stderr)
        sys.exit(1)

    df = raw_df.copy()
    df["depthMode"] = df["depthMode"].astype(int) + 1
    df["sector"] = df["sector"].astype(int) + 1

    unique_modes = np.array(sorted(np.unique(df["depthMode"])))
    unique_fans = np.array(sorted(np.unique(df["rxFanIndex"])))

    root = tk.Tk()
    FineTuneApp(root, df, unique_modes, unique_fans)
    root.mainloop()


if __name__ == "__main__":
    main()
