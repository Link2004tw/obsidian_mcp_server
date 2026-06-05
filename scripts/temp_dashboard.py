"""Live terminal dashboard — GPU, CPU, and NVMe temperatures with color coding.

Usage:
    python scripts/temp_dashboard.py

Requires: rich>=13.0  (pip install rich)
"""

import json
import subprocess
import time
from datetime import datetime
from functools import lru_cache

from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich import box


# ── Data sources ──────────────────────────────────────────────────────────

def get_gpu_info() -> dict | None:
    """GPU temperature (°C), VRAM (MiB), utilization (%), and model name."""
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=temperature.gpu,memory.used,memory.total,utilization.gpu,name",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode != 0:
            return None
        parts = [p.strip() for p in result.stdout.strip().split(", ")]
        if len(parts) < 4:
            return None
        return {
            "temp": int(parts[0]),
            "vram_used": int(parts[1]),
            "vram_total": int(parts[2]),
            "util": int(parts[3]),
            "name": parts[4] if len(parts) > 4 else "Unknown",
        }
    except Exception:
        return None


def get_disk_temps() -> list[dict]:
    """NVMe/SSD temperatures via PowerShell (slow — cache externally)."""
    try:
        script = (
            'Get-PhysicalDisk | Where-Object { $_.MediaType -eq "SSD" -or $_.BusType -eq "NVMe" } '
            "| ForEach-Object {"
            "    $rel = $_ | Get-StorageReliabilityCounter;"
            "    if ($rel -and $rel.Temperature -gt 0) {"
            "        [PSCustomObject]@{"
            "            Name = $_.FriendlyName;"
            "            Temp = $rel.Temperature;"
            "        } | ConvertTo-Json -Compress"
            "    }"
            "}"
        )
        result = subprocess.run(
            ["powershell", "-NoProfile", "-Command", script],
            capture_output=True, text=True, timeout=15,
        )
        if result.returncode != 0 or not result.stdout.strip():
            return []
        disks = []
        for line in result.stdout.strip().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                if isinstance(data, list):
                    for d in data:
                        disks.append({"name": d["Name"], "temp": d["Temp"]})
                else:
                    disks.append({"name": data["Name"], "temp": data["Temp"]})
            except (json.JSONDecodeError, KeyError, TypeError):
                continue
        return disks
    except Exception:
        return []


def get_cpu_temp() -> float | None:
    """CPU temperature (°C) via WMI."""
    try:
        script = (
            "$t = Get-CimInstance -Namespace root/wmi -ClassName MSAcpi_ThermalZoneTemperature; "
            "if ($t) { $t | ForEach-Object { $c = [math]::round(($_.CurrentTemperature - 2732) / 10, 1); "
            "if ($c -gt 0) { $c } } | Where-Object { $_ -gt 0 } | Select-Object -First 1 }"
        )
        result = subprocess.run(
            ["powershell", "-NoProfile", "-Command", script],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0 and result.stdout.strip():
            return float(result.stdout.strip())
    except Exception:
        pass
    return None


# ── Helpers ───────────────────────────────────────────────────────────────

def temp_style(temp: float | None, *, bold: bool = False) -> str:
    if temp is None:
        return "grey50"
    if temp < 60:
        return "bold green" if bold else "green"
    if temp < 75:
        return "bold yellow" if bold else "yellow"
    return "bold red1" if bold else "red1"


def bar(temp: float | None, width: int = 20) -> Text:
    if temp is None:
        return Text("─" * width, style="grey50")
    filled = max(0, min(width, int(temp / 100 * width)))
    t = Text()
    t.append("█" * filled, style=temp_style(temp))
    t.append("░" * (width - filled), style="grey37")
    return t


# ── Rendering ─────────────────────────────────────────────────────────────

def make_gpu_panel(gpu: dict | None, ts: str) -> Panel:
    if gpu is None:
        return Panel(Text("No GPU detected", style="grey50"), title="GPU", border_style="grey50")

    vram_pct = gpu["vram_used"] / gpu["vram_total"] * 100 if gpu["vram_total"] > 0 else 0
    vram_gb = f"{gpu['vram_used'] / 1024:.1f}/{gpu['vram_total'] / 1024:.1f} GB"

    lines = [
        Text.assemble(
            ("Temp  ", "bold"),
            (f"{gpu['temp']}°C  ", temp_style(gpu["temp"], bold=True)),
            bar(gpu["temp"]),
        ),
        Text.assemble(
            ("VRAM  ", "bold"),
            (f"{vram_gb} ({vram_pct:.0f}%)  ", temp_style(vram_pct)),
            bar(vram_pct),
        ),
        Text.assemble(
            ("Util  ", "bold"),
            (f"{gpu['util']}%", temp_style(gpu['util'])),
        ),
    ]
    return Panel(Text("\n").join(lines), title=f"GPU  ({gpu['name']})")


def make_cpu_panel(cpu_temp: float | None) -> Panel:
    if cpu_temp is None:
        return Panel(Text("No CPU data", style="grey50"), title="CPU", border_style="grey50")

    lines = [
        Text.assemble(
            ("Temp  ", "bold"),
            (f"{cpu_temp:.0f}°C  ", temp_style(cpu_temp, bold=True)),
            bar(cpu_temp),
        ),
    ]
    return Panel(Text("\n").join(lines), title="CPU")


def make_disks_panel(disks: list[dict]) -> Panel:
    if not disks:
        return Panel(Text("No NVMe data", style="grey50"), title="NVMe Drives", border_style="grey50")

    rows = []
    for d in disks:
        name = d["name"][:35]
        rows.append(
            Text.assemble(
                (f"{name:35s}  ", ""),
                (f"{d['temp']:2d}°C  ", temp_style(d["temp"], bold=True)),
                bar(d["temp"], 25),
            )
        )
    return Panel(Text("\n").join(rows), title="NVMe Drives")


def make_footer(ts: str, disk_age: int | None) -> Panel:
    parts = [Text(f"Updated: {ts}", style="grey50")]
    if disk_age is not None and disk_age > 0:
        parts.append(Text(f"  |  Disks: {disk_age}s ago", style="grey50"))
    return Panel(Text("  ").join(parts), style="grey50")


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    layout = Layout()
    layout.split_column(
        Layout(name="header", size=3),
        Layout(name="body"),
        Layout(name="footer", size=3),
    )
    layout["body"].split_row(
        Layout(name="gpu"),
        Layout(name="cpu"),
    )

    header = Panel(
        Text("System Temperature Monitor  —  Ctrl+C to exit", style="bold"),
    )

    # Cache disk data — polling PowerShell every cycle is too slow
    _disk_cache: list[dict] = []
    _disk_ts: float = 0
    DISK_INTERVAL = 10.0

    with Live(layout, refresh_per_second=2, screen=True):
        try:
            while True:
                now = time.time()
                ts = datetime.now().strftime("%H:%M:%S")

                gpu = get_gpu_info()
                cpu_temp = get_cpu_temp()

                if now - _disk_ts >= DISK_INTERVAL:
                    _disk_cache = get_disk_temps()
                    _disk_ts = now

                disk_age = int(now - _disk_ts) if _disk_cache else None

                layout["header"].update(header)
                layout["gpu"].update(make_gpu_panel(gpu, ts))
                layout["cpu"].update(make_cpu_panel(cpu_temp))
                layout["footer"].update(make_footer(ts, disk_age))

                # Rebuild disks panel into the body as a third row
                # (We split body into two rows: top row = gpu|cpu, bottom = disks)
                if not hasattr(main, "_body_split"):
                    layout["body"].split_column(
                        Layout(name="top"),
                        Layout(name="disks", size=6),
                    )
                    layout["body"]["top"].split_row(
                        Layout(name="gpu"),
                        Layout(name="cpu"),
                    )
                    main._body_split = True

                layout["body"]["disks"].update(make_disks_panel(_disk_cache))

                time.sleep(2)
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
