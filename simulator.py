"""
=============================================================================
 ATTACK INJECTION TOOL  (scapy)  --  demo companion for the threat detector
=============================================================================

 A standalone Tkinter GUI that GENERATES attack-style network traffic so you
 can prove the detector (network_threat_gui.py, in LIVE mode) actually flags
 real malicious patterns. It crafts real packets with scapy:

     * TCP SYN Scan        -> MITRE T1046 (Network Service Discovery)
     * SYN Flood           -> MITRE T1498 (Network Denial of Service)
     * UDP Flood           -> MITRE T1498
     * Data Exfiltration   -> MITRE T1048 (Exfil over alternative protocol)
     * C2 Beacon           -> MITRE T1071 (regular low-jitter callbacks)

 HOW TO DEMO
   1. Start network_threat_gui.py in LIVE mode on the interface that carries
      traffic to your TARGET.
   2. Run this tool, set the TARGET IP (a machine you own / a lab VM), pick an
      attack, press "Launch Attack".
   3. After the flow goes idle (~FLOW_TIMEOUT, ~10s) the detector closes the
      flow and you should see it appear as MEDIUM/HIGH risk with the matching
      MITRE technique.

 !!! ETHICS / SAFETY !!!
   Only run this against systems and networks you OWN or are explicitly
   authorized to test. Generating scan/flood traffic against third parties is
   illegal. Default target is your own host. Raw packet sending needs
   Administrator privileges + npcap. If scapy is unavailable, the tool runs in
   DRY-RUN mode (it only logs what it WOULD send).

 USAGE
   python attack_simulator.py
=============================================================================
"""

import socket
import threading
import time
from queue import Queue, Empty

import tkinter as tk
from tkinter import ttk, scrolledtext

# --- scapy is optional: fall back to dry-run if missing / no admin ---
try:
    from scapy.all import IP, TCP, UDP, Raw, send, RandShort, RandIP
    _HAS_SCAPY = True
except Exception:
    _HAS_SCAPY = False


def local_ip():
    """Best-effort primary LAN IP of this machine."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"


def default_target():
    """
    Best default attack target = the DEFAULT GATEWAY.

    IMPORTANT: never default to our own IP / loopback -- packets to ourselves
    are not captured by npcap, so the detector would see nothing. The gateway
    is a real host whose traffic egresses the NIC and IS captured.
    """
    try:
        from scapy.all import conf
        gw = conf.route.route("0.0.0.0")[2]
        if gw and gw not in ("0.0.0.0", "127.0.0.1"):
            return gw
    except Exception:
        pass
    # fallback: guess the gateway as x.x.x.1 of our subnet
    parts = local_ip().split(".")
    return ".".join(parts[:3] + ["1"]) if len(parts) == 4 else "192.168.1.1"


# =========================================================
# ATTACK ENGINE  (threaded packet generation)
# =========================================================

class AttackEngine:
    def __init__(self, log_cb, stat_cb, done_cb):
        self.log_cb = log_cb        # callback(str)
        self.stat_cb = stat_cb      # callback(sent:int, kind:str)
        self.done_cb = done_cb      # callback() when finished/stopped
        self.running = False
        self.sent = 0
        self._thread = None
        self.dry_run = not _HAS_SCAPY

    # ---- lifecycle ----
    def launch(self, params):
        if self.running:
            return
        self.running = True
        self.sent = 0
        self._thread = threading.Thread(target=self._run, args=(params,), daemon=True)
        self._thread.start()

    def stop(self):
        self.running = False

    # ---- helpers ----
    def _emit(self, pkt, desc):
        """Send one packet (or log it in dry-run) and update counters."""
        if self.dry_run:
            if self.sent < 3:        # don't spam the log
                self.log_cb(f"[dry-run] would send: {desc}")
        else:
            try:
                send(pkt, verbose=0)
            except Exception as e:
                self.log_cb(f"[send error] {e}")
                self.running = False
                return
        self.sent += 1

    def _pace(self, rate):
        """Sleep to honor the target packets-per-second rate."""
        if rate > 0:
            time.sleep(1.0 / rate)

    # ---- main dispatcher ----
    def _run(self, p):
        kind = p["attack"]
        target = p["target"]
        rate = p["rate"]
        count = p["count"]
        self.log_cb(f"=== Launching {kind} -> {target} "
                    f"({'DRY-RUN' if self.dry_run else 'LIVE scapy'}) ===")
        try:
            if kind == "TCP SYN Scan":
                self._syn_scan(p)
            elif kind == "SYN Flood":
                self._syn_flood(p)
            elif kind == "UDP Flood":
                self._udp_flood(p)
            elif kind == "Data Exfiltration":
                self._exfil(p)
            elif kind == "C2 Beacon":
                self._c2_beacon(p)
        finally:
            self.running = False
            self.log_cb(f"=== Finished: {self.sent} packets sent ===")
            self.done_cb()

    # ---- attack patterns ----
    def _syn_scan(self, p):
        """Sweep many destination ports with single SYN packets (T1046)."""
        target, rate = p["target"], p["rate"]
        start, end = p["port_start"], p["port_end"]
        for port in range(start, end + 1):
            if not self.running:
                break
            pkt = (IP(dst=target) / TCP(dport=port, sport=RandShort(), flags="S")
                   if not self.dry_run else None)
            self._emit(pkt, f"SYN -> {target}:{port}")
            self.stat_cb(self.sent, "scan")
            self._pace(rate)

    def _sport(self, p):
        """Source port: fixed (one aggregated flow) or random (many tiny flows)."""
        if p.get("fixed_sport", True) and not self.dry_run:
            return p.get("src_port", 40000)
        return RandShort() if not self.dry_run else 0

    def _syn_flood(self, p):
        """Flood one port with SYNs (T1498). Fixed sport => one big flow."""
        target, rate, count = p["target"], p["rate"], p["count"]
        port = p["port_start"]
        spoof = p["spoof"]
        for _ in range(count):
            if not self.running:
                break
            if self.dry_run:
                pkt = None
            else:
                ip = IP(dst=target, src=RandIP()) if spoof else IP(dst=target)
                pkt = ip / TCP(dport=port, sport=self._sport(p), flags="S")
            self._emit(pkt, f"SYN flood -> {target}:{port}")
            self.stat_cb(self.sent, "flood")
            self._pace(rate)

    def _udp_flood(self, p):
        """Flood a UDP port with payload packets (T1498)."""
        target, rate, count = p["target"], p["rate"], p["count"]
        port = p["port_start"]
        size = p["payload"]
        for _ in range(count):
            if not self.running:
                break
            pkt = (IP(dst=target) / UDP(dport=port, sport=self._sport(p)) / Raw(b"X" * size)
                   if not self.dry_run else None)
            self._emit(pkt, f"UDP -> {target}:{port} ({size}B)")
            self.stat_cb(self.sent, "flood")
            self._pace(rate)

    def _exfil(self, p):
        """Simulate bulk outbound transfer: large TCP payloads (T1048)."""
        target, rate, count = p["target"], p["rate"], p["count"]
        port = p["port_start"]
        size = max(p["payload"], 1000)
        sport = p.get("src_port", 40000)
        for i in range(count):
            if not self.running:
                break
            pkt = (IP(dst=target) / TCP(dport=port, sport=sport, flags="PA",
                                        seq=i * size) / Raw(b"D" * size)
                   if not self.dry_run else None)
            self._emit(pkt, f"exfil chunk -> {target}:{port} ({size}B)")
            self.stat_cb(self.sent, "exfil")
            self._pace(rate)

    def _c2_beacon(self, p):
        """Send small packets at a fixed interval (low jitter) -> C2 (T1071)."""
        target = p["target"]
        port = p["port_start"]
        count = p["count"]
        sport = p.get("src_port", 40000)
        interval = max(p.get("beacon_interval", 1.0), 0.05)
        for _ in range(count):
            if not self.running:
                break
            pkt = (IP(dst=target) / TCP(dport=port, sport=sport, flags="PA") / Raw(b"ping")
                   if not self.dry_run else None)
            self._emit(pkt, f"beacon -> {target}:{port}")
            self.stat_cb(self.sent, "c2")
            time.sleep(interval)


# =========================================================
# GUI
# =========================================================

ATTACKS = ["TCP SYN Scan", "SYN Flood", "UDP Flood", "Data Exfiltration", "C2 Beacon"]

# sensible defaults per attack: (rate pkts/s, count, port_start, port_end, payload)
PRESETS = {
    "TCP SYN Scan":      dict(rate=200, count=0,    port_start=1,   port_end=1024, payload=0),
    "SYN Flood":         dict(rate=500, count=2000, port_start=80,  port_end=80,   payload=0),
    "UDP Flood":         dict(rate=500, count=2000, port_start=53,  port_end=53,   payload=64),
    "Data Exfiltration": dict(rate=300, count=1500, port_start=443, port_end=443,  payload=1400),
    "C2 Beacon":         dict(rate=1,   count=30,   port_start=8080, port_end=8080, payload=0),
}


class AttackGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Attack Injection Tool (scapy)")
        self.geometry("760x600")
        self.configure(bg="#1e1e2e")

        self.log_queue = Queue()
        self.engine = AttackEngine(
            log_cb=lambda s: self.log_queue.put(s),
            stat_cb=self._on_stat,
            done_cb=lambda: self.log_queue.put("__DONE__"))
        self._sent = 0
        self._t0 = None

        self._build()
        self.after(150, self._poll)

    # ---------- layout ----------
    def _build(self):
        # warning banner
        warn = tk.Frame(self, bg="#5c1a1a")
        warn.pack(fill="x")
        mode = "LIVE (scapy)" if _HAS_SCAPY else "DRY-RUN (scapy not available)"
        tk.Label(warn, bg="#5c1a1a", fg="#ffd7d7", justify="left",
                 font=("Segoe UI", 9, "bold"),
                 text=("  AUTHORIZED USE ONLY - target machines you own / are allowed "
                       f"to test.   Mode: {mode}")).pack(anchor="w", pady=4, padx=6)

        tk.Label(self, text="ATTACK INJECTION TOOL", bg="#1e1e2e", fg="#f38ba8",
                 font=("Segoe UI", 15, "bold")).pack(anchor="w", padx=10, pady=(8, 2))
        tk.Label(self, text="Generates attack-style traffic so the detector can flag it.",
                 bg="#1e1e2e", fg="#a6adc8", font=("Segoe UI", 9)).pack(anchor="w", padx=10)
        tk.Label(self, bg="#1e1e2e", fg="#f9e2af", justify="left", font=("Segoe UI", 8),
                 text=("Setup: target = your GATEWAY or another LAN device (NOT your own IP). "
                       "In the detector pick your Wi-Fi/Ethernet interface and watch "
                       "'captured pkts' rise.")).pack(anchor="w", padx=10)

        form = tk.Frame(self, bg="#1e1e2e")
        form.pack(fill="x", padx=10, pady=10)

        def field(row, label, var, width=18):
            tk.Label(form, text=label, bg="#1e1e2e", fg="#cdd6f4",
                     font=("Segoe UI", 9)).grid(row=row, column=0, sticky="w", pady=3)
            e = tk.Entry(form, textvariable=var, width=width, bg="#2a2a3c", fg="#e0e0e0",
                         insertbackground="#e0e0e0")
            e.grid(row=row, column=1, sticky="w", padx=8)
            return e

        self.target_var = tk.StringVar(value=default_target())
        self.attack_var = tk.StringVar(value=ATTACKS[0])
        self.rate_var = tk.IntVar(value=200)
        self.count_var = tk.IntVar(value=0)
        self.pstart_var = tk.IntVar(value=1)
        self.pend_var = tk.IntVar(value=1024)
        self.payload_var = tk.IntVar(value=0)
        self.sport_var = tk.IntVar(value=40000)
        self.fixed_sport_var = tk.BooleanVar(value=True)
        self.spoof_var = tk.BooleanVar(value=False)

        # attack type
        tk.Label(form, text="Attack type", bg="#1e1e2e", fg="#cdd6f4",
                 font=("Segoe UI", 9)).grid(row=0, column=0, sticky="w", pady=3)
        cb = ttk.Combobox(form, textvariable=self.attack_var, values=ATTACKS,
                          state="readonly", width=20)
        cb.grid(row=0, column=1, sticky="w", padx=8)
        cb.bind("<<ComboboxSelected>>", self._apply_preset)

        field(1, "Target IP", self.target_var)
        field(2, "Rate (pkts/sec)", self.rate_var)
        field(3, "Count (0 = scan range)", self.count_var)
        field(4, "Port start", self.pstart_var)
        field(5, "Port end (scan)", self.pend_var)
        field(6, "Payload bytes", self.payload_var)
        field(7, "Source port", self.sport_var)
        tk.Checkbutton(form, text="Fixed source port (one aggregated flow - needed for HIGH)",
                       variable=self.fixed_sport_var, bg="#1e1e2e", fg="#cdd6f4",
                       selectcolor="#313244", activebackground="#1e1e2e",
                       activeforeground="#cdd6f4").grid(row=8, column=1, sticky="w", pady=3)
        tk.Checkbutton(form, text="Spoof random source IP (fragments flows - leave OFF for demo)",
                       variable=self.spoof_var, bg="#1e1e2e", fg="#cdd6f4",
                       selectcolor="#313244", activebackground="#1e1e2e",
                       activeforeground="#cdd6f4").grid(row=9, column=1, sticky="w", pady=3)

        # buttons
        btns = tk.Frame(self, bg="#1e1e2e")
        btns.pack(fill="x", padx=10)
        self.launch_btn = tk.Button(btns, text="Launch Attack", width=14,
                                    bg="#f38ba8", fg="#1e1e2e", relief="flat",
                                    command=self.launch)
        self.launch_btn.pack(side="left")
        self.stop_btn = tk.Button(btns, text="Stop", width=10, state="disabled",
                                  bg="#fab387", fg="#1e1e2e", relief="flat",
                                  command=self.stop)
        self.stop_btn.pack(side="left", padx=6)
        self.stat_var = tk.StringVar(value="idle")
        tk.Label(btns, textvariable=self.stat_var, bg="#1e1e2e", fg="#89b4fa",
                 font=("Segoe UI", 10, "bold")).pack(side="right")

        # log
        tk.Label(self, text="Activity log", bg="#1e1e2e", fg="#cdd6f4",
                 font=("Segoe UI", 10, "bold")).pack(anchor="w", padx=10, pady=(10, 0))
        self.log = scrolledtext.ScrolledText(self, bg="#11111b", fg="#e0e0e0",
                                             font=("Consolas", 9), height=14)
        self.log.pack(fill="both", expand=True, padx=10, pady=(2, 10))

        self._apply_preset()

    # ---------- preset fill ----------
    def _apply_preset(self, _e=None):
        p = PRESETS.get(self.attack_var.get())
        if not p:
            return
        self.rate_var.set(p["rate"])
        self.count_var.set(p["count"])
        self.pstart_var.set(p["port_start"])
        self.pend_var.set(p["port_end"])
        self.payload_var.set(p["payload"])

    # ---------- actions ----------
    def launch(self):
        params = {
            "attack": self.attack_var.get(),
            "target": self.target_var.get().strip(),
            "rate": max(self.rate_var.get(), 0),
            "count": max(self.count_var.get(), 0),
            "port_start": self.pstart_var.get(),
            "port_end": self.pend_var.get(),
            "payload": max(self.payload_var.get(), 0),
            "src_port": self.sport_var.get(),
            "fixed_sport": self.fixed_sport_var.get(),
            "spoof": self.spoof_var.get(),
            "beacon_interval": 1.0,
        }
        if not params["target"]:
            self._log("Set a target IP first.")
            return
        self._sent = 0
        self._t0 = time.time()
        self.launch_btn.config(state="disabled")
        self.stop_btn.config(state="normal")
        self.engine.launch(params)

    def stop(self):
        self.engine.stop()
        self.stop_btn.config(state="disabled")

    # ---------- callbacks ----------
    def _on_stat(self, sent, kind):
        self._sent = sent

    def _log(self, text):
        self.log.insert("end", text + "\n")
        self.log.see("end")

    def _poll(self):
        try:
            while True:
                msg = self.log_queue.get_nowait()
                if msg == "__DONE__":
                    self.launch_btn.config(state="normal")
                    self.stop_btn.config(state="disabled")
                else:
                    self._log(msg)
        except Empty:
            pass
        if self.engine.running and self._t0:
            dur = time.time() - self._t0
            pps = self._sent / dur if dur > 0 else 0
            self.stat_var.set(f"sent {self._sent} pkts | {pps:.0f}/s | {dur:.1f}s")
        self.after(150, self._poll)


if __name__ == "__main__":
    AttackGUI().mainloop()
