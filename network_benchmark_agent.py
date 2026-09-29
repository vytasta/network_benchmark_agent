"""
Network Benchmark AI Agent — iperf3, netperf, mtr, wrk
=========================================================
An AI agent that takes a natural language prompt describing a network path
to test — a proxy, a web server, a switch/router hop, a load balancer — and
autonomously:
  1. Translates it into iperf3, netperf, mtr, and wrk commands
  2. Runs each against the target
  3. Parses each tool's own output for throughput / latency / loss
  4. Reports results grouped by what they actually measure

These four tools measure different things and are reported that way rather
than forced into one misleading table:
  - iperf3 and netperf both measure raw TCP throughput — same unit
    (Mbps), so these two ARE compared side by side.
  - mtr measures path latency and packet loss hop by hop — a diagnostic,
    not a throughput number.
  - wrk measures HTTP-layer request throughput against a specific web
    endpoint — meaningful for a web server or reverse proxy, not
    comparable to raw TCP bandwidth.

IMPORTANT — iperf3 and netperf both need a matching server daemon already
running on the target (`iperf3 -s` / `netserver`) before this agent's
client-side commands can connect. This agent does not, and cannot, start a
server on a remote host for you (no SSH/remote-exec here) — if the target
isn't running one, the command will fail with a clear connection error
rather than a fabricated result. mtr and wrk need no server-side setup;
they work against any reachable host / HTTP endpoint.

Supports: any host reachable from this machine — proxies, web servers,
routers/switches (via their management or data IP), load balancers.

Set your Gemini key:
    export GOOGLE_API_KEY="AIza..."

Dry-run mode (no iperf3/netperf/mtr/wrk/Gemini needed):
    export DRY_RUN=true
    python network_benchmark_agent.py

Author : vytasta — https://vytasta.github.io/uberzunn/benchmark_agent
License: MIT
"""

import os
import json
import time
import random
import subprocess
import re
import functools
import socket
import shutil
import threading
from typing import TypedDict
from datetime import datetime

import requests

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, ToolMessage, AIMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

# ─────────────────────────────────────────────────────────────
# AGENTLEASE CONTROL-PLANE WIRING
# Set these to register this run as a data plane against a real
# AgentLease lease and heartbeat while the benchmark runs. If
# CONTROL_PLANE_URL / PAIRING_TOKEN aren't set, this is a no-op —
# the agent still runs exactly as it did standalone.
#
#   export CONTROL_PLANE_URL="https://your-control-plane.run.app"
#   export PAIRING_TOKEN="uz_live_..."     # from your lease
#   export DATA_PLANE_NAME="my-benchmark-host"   # optional, defaults to hostname
#   export DATA_PLANE_ENV="bare-metal"           # optional: kubernetes|openstack|bare-metal|other
# ─────────────────────────────────────────────────────────────

CONTROL_PLANE_URL = os.environ.get("CONTROL_PLANE_URL", "").rstrip("/")
PAIRING_TOKEN     = os.environ.get("PAIRING_TOKEN", "")
DATA_PLANE_NAME   = os.environ.get("DATA_PLANE_NAME", socket.gethostname())
DATA_PLANE_ENV    = os.environ.get("DATA_PLANE_ENV", "other")
HEARTBEAT_SECONDS = int(os.environ.get("HEARTBEAT_SECONDS", "30"))

_heartbeat_stop = threading.Event()


def _control_plane_enabled():
    return bool(CONTROL_PLANE_URL and PAIRING_TOKEN)


def register_data_plane():
    """Register this run against the control plane. Returns the data_plane
    id on success, or None if disabled/unreachable (the benchmark still
    runs either way — a control plane outage shouldn't block your work)."""
    if not _control_plane_enabled():
        return None
    try:
        resp = requests.post(
            f"{CONTROL_PLANE_URL}/dataplanes/register",
            json={
                "pairing_token": PAIRING_TOKEN,
                "name": DATA_PLANE_NAME,
                "environment": DATA_PLANE_ENV,
            },
            timeout=10,
        )
        resp.raise_for_status()
        data_plane_id = resp.json()["id"]
        print(f"[control-plane] registered as data plane {data_plane_id} ({DATA_PLANE_NAME})")
        return data_plane_id
    except Exception as e:
        print(f"[control-plane] registration failed, continuing without it: {e}")
        return None


def _heartbeat_loop(data_plane_id):
    while not _heartbeat_stop.wait(HEARTBEAT_SECONDS):
        try:
            requests.post(
                f"{CONTROL_PLANE_URL}/dataplanes/{data_plane_id}/heartbeat",
                json={"pairing_token": PAIRING_TOKEN},
                timeout=10,
            )
        except Exception as e:
            print(f"[control-plane] heartbeat failed (will retry): {e}")


def start_heartbeat(data_plane_id):
    if not data_plane_id:
        return None
    t = threading.Thread(target=_heartbeat_loop, args=(data_plane_id,), daemon=True)
    t.start()
    print(f"[control-plane] heartbeating every {HEARTBEAT_SECONDS}s")
    return t


def stop_heartbeat():
    _heartbeat_stop.set()

# ─────────────────────────────────────────────────────────────
# DRY-RUN FLAG
# Set DRY_RUN=true to simulate the full agent without running
# iperf3/netperf/mtr/wrk or consuming Gemini tokens. Great for CI and demos.
# ─────────────────────────────────────────────────────────────

DRY_RUN = os.environ.get("DRY_RUN", "false").lower() == "true"

if DRY_RUN:
    print("=" * 65)
    print("DRY-RUN MODE — no real tools or LLM calls will be made")
    print("=" * 65)


# ─────────────────────────────────────────────────────────────
# RATE LIMIT PROTECTION — throttle only
# A fixed delay between every LLM call keeps us safely under
# Gemini free tier (15 RPM). No retry loops needed when you
# stay under the limit in the first place.
# ─────────────────────────────────────────────────────────────

THROTTLE_SECONDS = 5        # 5s gap = max 12 RPM — safely under 15 RPM limit
_last_call_time  = 0.0


def throttle():
    """Enforce minimum gap between LLM calls."""
    global _last_call_time
    if DRY_RUN:
        return                  # no throttle needed in dry-run
    elapsed = time.time() - _last_call_time
    if elapsed < THROTTLE_SECONDS:
        wait = THROTTLE_SECONDS - elapsed
        print(f"  [throttle] waiting {wait:.1f}s to stay under rate limit...")
        time.sleep(wait)
    _last_call_time = time.time()


MAX_RATE_RETRIES = 5


def rate_limited(fn):
    """Decorator: throttle before every LLM call, and retry with exponential
    backoff on rate-limit (429) or transient server overload (503) errors."""
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        for attempt in range(MAX_RATE_RETRIES + 1):
            throttle()
            try:
                return fn(*args, **kwargs)
            except Exception as e:
                err = str(e).lower()
                is_retryable = (
                    "429" in err
                    or "503" in err
                    or "rate limit" in err
                    or "resource_exhausted" in err
                    or "quota" in err
                    or "unavailable" in err
                    or "high demand" in err
                )
                if is_retryable and attempt < MAX_RATE_RETRIES:
                    wait = (2 ** attempt) * 10 + random.uniform(0, 2)
                    print(f"  [retry] {e.__class__.__name__} — retry {attempt + 1}/{MAX_RATE_RETRIES} in {wait:.1f}s...")
                    time.sleep(wait)
                else:
                    raise
    return wrapper


# ─────────────────────────────────────────────────────────────
# AGENT STATE
# ─────────────────────────────────────────────────────────────

class BenchmarkState(TypedDict):
    messages:           list   # conversation history
    workload_params:    dict   # parsed workload parameters
    iperf3_command:     str    # generated iperf3 client command
    netperf_command:    str    # generated netperf client command
    mtr_command:        str    # generated mtr command
    wrk_command:        str    # generated wrk command
    iperf3_raw_output:  str    # raw JSON stdout from iperf3
    netperf_raw_output: str    # raw stdout from netperf
    mtr_raw_output:     str    # raw JSON stdout from mtr
    wrk_raw_output:     str    # raw stdout from wrk
    iperf3_results:     dict   # parsed iperf3 throughput
    netperf_results:    dict   # parsed netperf throughput/transaction rate
    mtr_results:        dict   # parsed mtr latency/loss
    wrk_results:        dict   # parsed wrk HTTP throughput/latency
    comparison:         str    # grouped results summary
    error:              str    # last error if any
    final_answer:        str   # clean output for the user


# ─────────────────────────────────────────────────────────────
# TOOLS
# ─────────────────────────────────────────────────────────────

@tool
def generate_iperf3_command(
    target_host: str,
    port: int = 5201,
    duration_seconds: int = 10,
    reverse: bool = False,
    udp: bool = False,
) -> str:
    """
    Generate an iperf3 client command for a raw TCP/UDP throughput test.

    Requires an iperf3 server already running on target_host (`iperf3 -s
    -p <port>`) — this agent cannot start one on a remote host for you.

    Args:
        target_host:      host running iperf3 -s
        port:              port the iperf3 server is listening on
        duration_seconds:  test duration in seconds
        reverse:           measure server->client instead of client->server
        udp:               use UDP instead of TCP

    Returns:
        The iperf3 command line as a string (JSON output via -J, for
        reliable parsing).
    """
    if DRY_RUN:
        return f"[DRY-RUN] iperf3 command for {target_host}:{port} | {duration_seconds}s | reverse={reverse} | udp={udp}"

    flags = f"-c {target_host} -p {port} -t {duration_seconds} -J"
    if reverse:
        flags += " -R"
    if udp:
        flags += " -u"
    return f"iperf3 {flags}"


@tool
def run_iperf3(command: str) -> str:
    """
    Execute an iperf3 command generated by generate_iperf3_command.

    Args:
        command: the iperf3 command line

    Returns:
        Raw iperf3 JSON stdout (iperf3 -J prints valid JSON on both success
        and connection failure — a failure comes back as {"error": "..."}).
    """
    if DRY_RUN or command.startswith("[DRY-RUN]"):
        return json.dumps({
            "end": {
                "sum_sent":     {"seconds": 10.0, "bytes": 22548578304, "bits_per_second": 18038862643.2, "retransmits": 210},
                "sum_received": {"seconds": 10.0, "bytes": 22540123456, "bits_per_second": 18032098765.4},
            }
        })

    try:
        print("  [iperf3] running benchmark...")
        result = subprocess.run(command.split(), capture_output=True, text=True, timeout=duration_from(command) + 30)
        # iperf3 -J writes valid JSON to stdout even on failure; prefer stdout.
        return result.stdout or json.dumps({"error": result.stderr or "iperf3 exited non-zero with no output"})
    except FileNotFoundError:
        return json.dumps({"error": "iperf3 not found. Install: sudo apt install iperf3 (Debian/Ubuntu) or brew install iperf3 (macOS)"})
    except subprocess.TimeoutExpired:
        return json.dumps({"error": "iperf3 timed out"})
    except Exception as e:
        return json.dumps({"error": str(e)})


def duration_from(command: str) -> int:
    """Best-effort extraction of -t/-l duration from a command string, used
    only to size the subprocess timeout generously — falls back to 60s if
    it can't find one rather than guessing wrong in either direction."""
    m = re.search(r"-t\s+(\d+)", command) or re.search(r"-l\s+(\d+)", command)
    return int(m.group(1)) if m else 60


@tool
def parse_iperf3_output(raw_output: str) -> str:
    """
    Parse iperf3's JSON output for sent/received throughput.

    Args:
        raw_output: JSON stdout from run_iperf3

    Returns:
        JSON string with sent_mbps, received_mbps, retransmits, duration_s,
        or {"tool": "iperf3", "error": "..."} if iperf3 itself reported an
        error (e.g. connection refused because no server is running).
    """
    try:
        data = json.loads(raw_output)
        if "error" in data:
            return json.dumps({"tool": "iperf3", "error": data["error"]})
        sent = data["end"]["sum_sent"]
        recv = data["end"]["sum_received"]
        return json.dumps({
            "tool":           "iperf3",
            "sent_mbps":      round(sent["bits_per_second"] / 1e6, 2),
            "received_mbps":  round(recv["bits_per_second"] / 1e6, 2),
            "retransmits":    sent.get("retransmits"),
            "duration_s":     round(sent["seconds"], 2),
        })
    except Exception as e:
        return json.dumps({"tool": "iperf3", "error": f"could not parse iperf3 output: {e}", "raw_snippet": raw_output[:300]})


@tool
def generate_netperf_command(
    target_host: str,
    port: int = 12865,
    duration_seconds: int = 10,
    test_type: str = "TCP_STREAM",
) -> str:
    """
    Generate a netperf client command for throughput (TCP_STREAM) or
    request/response latency-under-load (TCP_RR) testing.

    Requires a netserver daemon already running on target_host (`netserver
    -p <port>`) — this agent cannot start one on a remote host for you.

    Args:
        target_host:       host running netserver
        port:               port netserver is listening on
        duration_seconds:   test duration in seconds
        test_type:          "TCP_STREAM" (raw throughput) or "TCP_RR"
                             (transaction rate — small request/response pairs,
                             a proxy for latency under concurrent load)

    Returns:
        The netperf command line as a string.
    """
    if DRY_RUN:
        return f"[DRY-RUN] netperf {test_type} for {target_host}:{port} | {duration_seconds}s"

    return f"netperf -H {target_host} -p {port} -l {duration_seconds} -t {test_type}"


@tool
def run_netperf(command: str) -> str:
    """
    Execute a netperf command generated by generate_netperf_command.

    Args:
        command: the netperf command line

    Returns:
        Raw netperf stdout.
    """
    if DRY_RUN or command.startswith("[DRY-RUN]"):
        if "TCP_RR" in command:
            return (
                "MIGRATED TCP REQUEST/RESPONSE TEST\n"
                "Local /Remote\nSocket Size   Request  Resp.   Elapsed  Trans.\n"
                "Send   Recv   Size     Size    Time     Rate\n"
                "bytes  Bytes  bytes    bytes   secs.    per sec\n\n"
                "16384  131072 1        1       10.00    29604.42\n"
            )
        return (
            "MIGRATED TCP STREAM TEST\n"
            "Recv   Send    Send\nSocket Socket  Message  Elapsed\n"
            "Size   Size    Size     Time     Throughput\n"
            "bytes  bytes   bytes    secs.    10^6bits/sec\n\n"
            "131072  16384  16384    10.00     20847.64\n"
        )

    try:
        print("  [netperf] running benchmark...")
        result = subprocess.run(command.split(), capture_output=True, text=True, timeout=duration_from(command) + 30)
        if result.returncode != 0 and not result.stdout:
            return json.dumps({"error": result.stderr or "netperf exited non-zero"})
        return result.stdout
    except FileNotFoundError:
        return json.dumps({"error": "netperf not found. Install: sudo apt install netperf (Debian/Ubuntu) or brew install netperf (macOS)"})
    except subprocess.TimeoutExpired:
        return json.dumps({"error": "netperf timed out"})
    except Exception as e:
        return json.dumps({"error": str(e)})


@tool
def parse_netperf_output(raw_output: str) -> str:
    """
    Parse netperf's TCP_STREAM or TCP_RR result line.

    Args:
        raw_output: stdout from run_netperf

    Returns:
        JSON string with either throughput_mbps (TCP_STREAM) or
        transactions_per_sec (TCP_RR), detected from the output's own
        test-name header rather than a separate parameter — netperf prints
        which test it ran, so there's no need to pass that in twice.
    """
    try:
        if "error" in raw_output.lower() and "MIGRATED" not in raw_output:
            return json.dumps({"tool": "netperf", "error": "netperf run reported an error", "raw_snippet": raw_output[:300]})

        is_rr = "REQUEST/RESPONSE" in raw_output
        data_lines = [l for l in raw_output.splitlines() if re.match(r"^\s*\d", l)]
        if not data_lines:
            return json.dumps({"tool": "netperf", "error": "no result line found", "raw_snippet": raw_output[:300]})

        value = float(data_lines[0].split()[-1])
        if is_rr:
            return json.dumps({"tool": "netperf", "test": "TCP_RR", "transactions_per_sec": value})
        return json.dumps({"tool": "netperf", "test": "TCP_STREAM", "throughput_mbps": value})
    except Exception as e:
        return json.dumps({"tool": "netperf", "error": str(e), "raw_snippet": raw_output[:300]})


@tool
def generate_mtr_command(target_host: str, count: int = 10) -> str:
    """
    Generate an mtr command for hop-by-hop path latency and packet loss.

    Args:
        target_host:  host or IP to trace
        count:        number of pings to send per hop (more = more stable
                      loss%/latency numbers, slower to finish)

    Returns:
        The mtr command line as a string (JSON output via -j).
    """
    if DRY_RUN:
        return f"[DRY-RUN] mtr command for {target_host} | count={count}"

    return f"mtr -j -c {count} {target_host}"


@tool
def run_mtr(command: str) -> str:
    """
    Execute an mtr command generated by generate_mtr_command. Needs no
    server-side setup — works against any reachable host.

    Args:
        command: the mtr command line

    Returns:
        Raw mtr JSON stdout.
    """
    if DRY_RUN or command.startswith("[DRY-RUN]"):
        return json.dumps({
            "report": {
                "mtr": {"src": "agent-host", "dst": "target", "tests": 10},
                "hubs": [
                    {"count": 1, "host": "10.0.0.1",     "Loss%": 0.0, "Snt": 10, "Last": 0.4, "Avg": 0.5,  "Best": 0.3,  "Wrst": 0.9,  "StDev": 0.1},
                    {"count": 2, "host": "203.0.113.1",  "Loss%": 0.0, "Snt": 10, "Last": 8.2, "Avg": 8.9,  "Best": 7.1,  "Wrst": 12.4, "StDev": 1.2},
                    {"count": 3, "host": "target",       "Loss%": 0.0, "Snt": 10, "Last": 14.1,"Avg": 15.2, "Best": 13.0, "Wrst": 19.8, "StDev": 1.8},
                ]
            }
        })

    try:
        print("  [mtr] tracing path...")
        # mtr's own -c count controls its runtime; timeout is generous headroom.
        result = subprocess.run(command.split(), capture_output=True, text=True, timeout=120)
        if result.returncode != 0 and not result.stdout:
            return json.dumps({"error": result.stderr or "mtr exited non-zero"})
        return result.stdout
    except FileNotFoundError:
        return json.dumps({"error": "mtr not found. Install: sudo apt install mtr (Debian/Ubuntu) or brew install mtr (macOS)"})
    except subprocess.TimeoutExpired:
        return json.dumps({"error": "mtr timed out"})
    except Exception as e:
        return json.dumps({"error": str(e)})


@tool
def parse_mtr_output(raw_output: str) -> str:
    """
    Parse mtr's JSON report for hop count and destination latency/loss.

    Args:
        raw_output: JSON stdout from run_mtr

    Returns:
        JSON string with hop_count and the destination (last) hop's
        loss%/avg/best/worst/stddev latency in ms — the numbers that matter
        for "is the path to this target healthy", not every intermediate hop.
    """
    try:
        data = json.loads(raw_output)
        if "error" in data:
            return json.dumps({"tool": "mtr", "error": data["error"]})
        hubs = data["report"]["hubs"]
        if not hubs:
            return json.dumps({"tool": "mtr", "error": "no hops in mtr report"})
        dest = hubs[-1]
        return json.dumps({
            "tool":            "mtr",
            "hop_count":       len(hubs),
            "dest_host":       dest.get("host"),
            "dest_loss_pct":   dest.get("Loss%"),
            "dest_avg_ms":     dest.get("Avg"),
            "dest_best_ms":    dest.get("Best"),
            "dest_worst_ms":   dest.get("Wrst"),
            "dest_stddev_ms":  dest.get("StDev"),
        })
    except Exception as e:
        return json.dumps({"tool": "mtr", "error": f"could not parse mtr output: {e}", "raw_snippet": raw_output[:300]})


@tool
def generate_wrk_command(url: str, threads: int = 2, connections: int = 10, duration_seconds: int = 10) -> str:
    """
    Generate a wrk command for HTTP-layer load testing against a web
    server, reverse proxy, or load balancer's HTTP endpoint.

    Args:
        url:               full URL to hit, e.g. http://10.0.0.5/health
        threads:           number of wrk worker threads (keep <= CPU cores)
        connections:       total concurrent HTTP connections
        duration_seconds:  test duration in seconds

    Returns:
        The wrk command line as a string.
    """
    if DRY_RUN:
        return f"[DRY-RUN] wrk command for {url} | threads={threads} | connections={connections} | {duration_seconds}s"

    return f"wrk -t{threads} -c{connections} -d{duration_seconds}s {url}"


@tool
def run_wrk(command: str) -> str:
    """
    Execute a wrk command generated by generate_wrk_command. Needs no
    special server-side setup beyond the target actually serving HTTP.

    Args:
        command: the wrk command line

    Returns:
        Raw wrk stdout.
    """
    if DRY_RUN or command.startswith("[DRY-RUN]"):
        return (
            "Running 10s test\n  2 threads and 10 connections\n"
            "  Thread Stats   Avg      Stdev     Max   +/- Stdev\n"
            "    Latency    22.45ms   10.37ms  55.80ms   64.32%\n"
            "    Req/Sec   198.90     57.97   363.00     68.33%\n"
            "  3960 requests in 10.00s, 21.8MB read\n"
            "Requests/sec:    396.00\n"
            "Transfer/sec:      2.18MB\n"
        )

    try:
        print("  [wrk] running HTTP load test...")
        dur_match = re.search(r"-d(\d+)s", command)
        timeout_s = (int(dur_match.group(1)) if dur_match else 60) + 30
        result = subprocess.run(command.split(), capture_output=True, text=True, timeout=timeout_s)
        if result.returncode != 0 and not result.stdout:
            return json.dumps({"error": result.stderr or "wrk exited non-zero"})
        return result.stdout
    except FileNotFoundError:
        return json.dumps({"error": "wrk not found. Install: sudo apt install wrk (Debian/Ubuntu) or brew install wrk (macOS)"})
    except subprocess.TimeoutExpired:
        return json.dumps({"error": "wrk timed out"})
    except Exception as e:
        return json.dumps({"error": str(e)})


@tool
def parse_wrk_output(raw_output: str) -> str:
    """
    Parse wrk's summary block for requests/sec, average latency, and
    transfer rate.

    Args:
        raw_output: stdout from run_wrk

    Returns:
        JSON string with requests_per_sec, avg_latency_ms, total_requests,
        and transfer_per_sec (kept as its original unit string, e.g.
        "2.18MB", since wrk mixes KB/MB/GB and normalizing isn't needed for
        a single-run report).
    """
    try:
        if "Requests/sec" not in raw_output:
            return json.dumps({"tool": "wrk", "error": "could not find wrk summary in output", "raw_snippet": raw_output[:300]})

        def _f(pattern, cast=float):
            m = re.search(pattern, raw_output)
            return cast(m.group(1)) if m else None

        return json.dumps({
            "tool":              "wrk",
            "requests_per_sec":  _f(r"Requests/sec:\s+([\d.]+)"),
            "avg_latency_ms":    _f(r"Latency\s+([\d.]+)ms"),
            "total_requests":    _f(r"(\d+) requests in", int),
            "transfer_per_sec":  _f(r"Transfer/sec:\s+([\d.]+\w+)", str),
        })
    except Exception as e:
        return json.dumps({"tool": "wrk", "error": str(e), "raw_snippet": raw_output[:300]})


ALL_TOOLS = [
    generate_iperf3_command,  run_iperf3,  parse_iperf3_output,
    generate_netperf_command, run_netperf, parse_netperf_output,
    generate_mtr_command,     run_mtr,     parse_mtr_output,
    generate_wrk_command,     run_wrk,     parse_wrk_output,
]
TOOL_MAP = {t.name: t for t in ALL_TOOLS}


# ─────────────────────────────────────────────────────────────
# LLM SETUP
# ─────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a network benchmarking assistant. Given a natural
language description of a network path or endpoint to test, drive it
through iperf3, netperf, mtr, and wrk, in this order:

1. Call generate_iperf3_command, then run_iperf3, then parse_iperf3_output.
2. Call generate_netperf_command, then run_netperf, then parse_netperf_output.
3. Call generate_mtr_command, then run_mtr, then parse_mtr_output.
4. Call generate_wrk_command, then run_wrk, then parse_wrk_output — only if
   the user's prompt mentions an HTTP endpoint (a URL, a web server, a
   proxy, or a load balancer's HTTP frontend). If there's no URL to test,
   skip wrk entirely rather than inventing one.

Call ONE tool at a time, in the order above.

Default values if not specified:
  iperf3/netperf port: 5201 / 12865 | duration: 10s | mtr count: 10
  netperf test_type: TCP_STREAM | wrk threads: 2 | wrk connections: 10

Remember: iperf3 and netperf both need a matching server daemon already
running on the target. If either comes back with a connection error, report
that plainly rather than treating it as a benchmark result.
"""


def _make_llm():
    """Create LLM — returns a stub in dry-run mode."""
    if DRY_RUN:
        return None
    return ChatGoogleGenerativeAI(
        model="gemini-flash-latest",
        temperature=0,
        google_api_key=os.environ.get("GOOGLE_API_KEY"),
    ).bind_tools(ALL_TOOLS)


def _make_llm_plain():
    if DRY_RUN:
        return None
    return ChatGoogleGenerativeAI(
        model="gemini-flash-latest",
        temperature=0,
        google_api_key=os.environ.get("GOOGLE_API_KEY"),
    )


LLM       = _make_llm()
LLM_PLAIN = _make_llm_plain()


# ─────────────────────────────────────────────────────────────
# DRY-RUN SIMULATION — scripted tool sequence, no LLM needed
# ─────────────────────────────────────────────────────────────

DRY_RUN_SEQUENCE = [
    ("generate_iperf3_command",  {"target_host": "10.0.0.5", "port": 5201, "duration_seconds": 10}),
    ("run_iperf3",               {"command": "[DRY-RUN] iperf3 command"}),
    ("parse_iperf3_output",      {"raw_output": json.dumps({"end": {
        "sum_sent":     {"seconds": 10.0, "bytes": 22548578304, "bits_per_second": 18038862643.2, "retransmits": 210},
        "sum_received": {"seconds": 10.0, "bytes": 22540123456, "bits_per_second": 18032098765.4},
    }})}),
    ("generate_netperf_command", {"target_host": "10.0.0.5", "port": 12865, "duration_seconds": 10, "test_type": "TCP_STREAM"}),
    ("run_netperf",              {"command": "[DRY-RUN] netperf command"}),
    ("parse_netperf_output",     {"raw_output": "MIGRATED TCP STREAM TEST\nbytes  bytes   bytes    secs.    10^6bits/sec\n\n131072  16384  16384    10.00     20847.64\n"}),
    ("generate_mtr_command",     {"target_host": "10.0.0.5", "count": 10}),
    ("run_mtr",                  {"command": "[DRY-RUN] mtr command"}),
    ("parse_mtr_output",         {"raw_output": json.dumps({"report": {"hubs": [
        {"count": 1, "host": "10.0.0.1", "Loss%": 0.0, "Avg": 0.5},
        {"count": 2, "host": "10.0.0.5", "Loss%": 0.0, "Avg": 15.2, "Best": 13.0, "Wrst": 19.8, "StDev": 1.8},
    ]}})}),
    ("generate_wrk_command",     {"url": "http://10.0.0.5/health", "threads": 2, "connections": 10, "duration_seconds": 10}),
    ("run_wrk",                  {"command": "[DRY-RUN] wrk command"}),
    ("parse_wrk_output",         {"raw_output": "Requests/sec:    396.00\nLatency    22.45ms\n3960 requests in 10.00s, 21.8MB read\nTransfer/sec:      2.18MB\n"}),
]

_dry_run_step = 0


class _DryRunMessage(AIMessage):
    """Mimics an AIMessage with tool_calls for dry-run mode."""
    def __init__(self, tool_name, tool_args, call_id):
        super().__init__(
            content=f"[DRY-RUN] calling {tool_name}",
            tool_calls=[{"name": tool_name, "args": tool_args, "id": call_id}],
        )


# ─────────────────────────────────────────────────────────────
# GRAPH NODES
# ─────────────────────────────────────────────────────────────

@rate_limited
def _llm_invoke(messages):
    return LLM.invoke(messages)


def planner_node(state: BenchmarkState) -> dict:
    """LLM (or dry-run stub) decides the next tool to call."""
    global _dry_run_step

    if DRY_RUN:
        if _dry_run_step >= len(DRY_RUN_SEQUENCE):
            return {"messages": state["messages"] + [AIMessage(content="[DRY-RUN] all tools complete", tool_calls=[])]}

        tool_name, tool_args = DRY_RUN_SEQUENCE[_dry_run_step]
        _dry_run_step += 1
        print(f"  [dry-run] step {_dry_run_step}/{len(DRY_RUN_SEQUENCE)}: {tool_name}")
        time.sleep(0.3)
        msg = _DryRunMessage(tool_name, tool_args, f"dry-call-{_dry_run_step}")
        return {"messages": state["messages"] + [msg]}

    messages = state["messages"]
    if len(messages) == 1:
        messages = [{"role": "system", "content": SYSTEM_PROMPT}] + messages
    response = _llm_invoke(messages)
    return {"messages": state["messages"] + [response]}


def tool_executor_node(state: BenchmarkState) -> dict:
    """Execute whatever tool call the planner just requested and fold its
    result into state."""
    last = state["messages"][-1]
    new_messages = []
    updates = {}

    for tc in last.tool_calls:
        fn = TOOL_MAP[tc["name"]]
        result = fn.invoke(tc["args"])
        new_messages.append(ToolMessage(content=result, tool_call_id=tc["id"]))

        if tc["name"] == "generate_iperf3_command":
            updates["iperf3_command"] = result
        elif tc["name"] == "generate_netperf_command":
            updates["netperf_command"] = result
        elif tc["name"] == "generate_mtr_command":
            updates["mtr_command"] = result
        elif tc["name"] == "generate_wrk_command":
            updates["wrk_command"] = result
        elif tc["name"] == "run_iperf3":
            updates["iperf3_raw_output"] = result
        elif tc["name"] == "run_netperf":
            updates["netperf_raw_output"] = result
        elif tc["name"] == "run_mtr":
            updates["mtr_raw_output"] = result
        elif tc["name"] == "run_wrk":
            updates["wrk_raw_output"] = result
        elif tc["name"] == "parse_iperf3_output":
            try:    updates["iperf3_results"] = json.loads(result)
            except: updates["iperf3_results"] = {"raw": result}
        elif tc["name"] == "parse_netperf_output":
            try:    updates["netperf_results"] = json.loads(result)
            except: updates["netperf_results"] = {"raw": result}
        elif tc["name"] == "parse_mtr_output":
            try:    updates["mtr_results"] = json.loads(result)
            except: updates["mtr_results"] = {"raw": result}
        elif tc["name"] == "parse_wrk_output":
            try:    updates["wrk_results"] = json.loads(result)
            except: updates["wrk_results"] = {"raw": result}

    return {"messages": new_messages, **updates}


@rate_limited
def _compare_invoke(messages):
    return LLM_PLAIN.invoke(messages)


def comparison_node(state: BenchmarkState) -> dict:
    """Report results grouped by what they actually measure: iperf3 vs
    netperf (both raw TCP throughput, genuinely comparable in Mbps), mtr
    (path latency/loss, a different axis), and wrk (HTTP-layer throughput,
    only meaningful if a URL was tested)."""
    iperf3  = state.get("iperf3_results", {})
    netperf = state.get("netperf_results", {})
    mtr     = state.get("mtr_results", {})
    wrk     = state.get("wrk_results", {})
    wrk_ran = bool(wrk)

    if DRY_RUN:
        wrk_section = (
            f"HTTP throughput (wrk)\n"
            f"  Requests/sec     : {wrk.get('requests_per_sec', 'N/A')}\n"
            f"  Avg latency (ms) : {wrk.get('avg_latency_ms', 'N/A')}\n"
            f"  Transfer/sec     : {wrk.get('transfer_per_sec', 'N/A')}"
            if wrk_ran else
            "HTTP throughput (wrk): not tested (no URL in this run)"
        )
        answer = f"""
NETWORK BENCHMARK SUMMARY  [DRY-RUN SIMULATION]
====================================================
Raw TCP throughput (iperf3 vs netperf)
---------------------------------------
Metric              | iperf3          | netperf
--------------------+-----------------+-----------------
Sent (Mbps)         | {iperf3.get('sent_mbps', 'N/A')} | {netperf.get('throughput_mbps', 'N/A')}
Received (Mbps)     | {iperf3.get('received_mbps', 'N/A')} | -
Retransmits         | {iperf3.get('retransmits', 'N/A')} | -

Path latency / loss (mtr, to {mtr.get('dest_host', 'target')}, {mtr.get('hop_count', '?')} hops)
---------------------------------------
  Loss%            : {mtr.get('dest_loss_pct', 'N/A')}
  Avg latency (ms) : {mtr.get('dest_avg_ms', 'N/A')}
  Best/Worst (ms)  : {mtr.get('dest_best_ms', 'N/A')} / {mtr.get('dest_worst_ms', 'N/A')}

{wrk_section}

ANALYSIS
--------
[DRY-RUN] iperf3 and netperf agree closely on raw throughput, as expected —
both are measuring the same TCP path with similar methodology. mtr's
latency is a path-health signal (hop count, loss, jitter via stddev), not a
throughput number, and shouldn't be read as "slower than the throughput
tools" — they're not measuring the same thing.

RECOMMENDATION
--------------
Use iperf3 as the primary throughput number (broader adoption, easier JSON
output). Use netperf to cross-check, especially TCP_RR for latency-under-
load. Use mtr whenever loss or jitter — not just raw bandwidth — is the
concern (routers, switches, congested links). Use wrk specifically for
web-server/proxy/load-balancer HTTP performance, not as a general network
throughput substitute.
""".strip()
        return {"comparison": answer, "final_answer": answer}

    wrk_note = (
        f"wrk (HTTP throughput): {json.dumps(wrk)}"
        if wrk_ran else
        "wrk was not run (no HTTP endpoint was part of this request)."
    )

    prompt = f"""
You are a network performance expert. Summarize these results for a
network engineer. iperf3 and netperf both measure raw TCP throughput and
CAN be compared directly (both in Mbps). mtr measures path latency/loss, a
different axis — do not merge it into the throughput comparison. wrk
measures HTTP-layer throughput and is only relevant if it was actually run.

iperf3 results: {json.dumps(iperf3, indent=2)}
netperf results: {json.dumps(netperf, indent=2)}
mtr results: {json.dumps(mtr, indent=2)}
{wrk_note}

Format:
NETWORK BENCHMARK SUMMARY
==============================
Raw TCP throughput: iperf3 vs netperf side by side
Path latency/loss: mtr's numbers, described as a separate signal
HTTP throughput: wrk's numbers if run, or a one-line note if not

ANALYSIS
--------
2-3 sentences on what the numbers suggest.

RECOMMENDATION
--------------
Which tool to lean on for which kind of question going forward.
"""
    response = _compare_invoke([{"role": "user", "content": prompt}])
    return {"comparison": response.content, "final_answer": response.content}


# ─────────────────────────────────────────────────────────────
# EDGE ROUTING
# ─────────────────────────────────────────────────────────────

def route_after_planner(state: BenchmarkState) -> str:
    last = state["messages"][-1]
    if hasattr(last, "tool_calls") and last.tool_calls:
        return "execute_tool"
    return "compare"


def route_after_tool(state: BenchmarkState) -> str:
    """Always hand back to the planner after a tool call.

    Unlike the storage/database agents (where every tool is mandatory and
    this function can short-circuit straight to "compare" the moment they're
    all done), wrk here is genuinely optional — it only belongs in the plan
    when the request actually names an HTTP endpoint. This function has no
    way to know that (state doesn't say "a URL was in scope", and the last
    message at this point is always the ToolMessage that was just executed,
    which never carries tool_calls — checking it for tool_calls, as an
    earlier version of this function did, is always false and would send
    every run straight to "compare" right after mtr, silently skipping wrk
    even when it should have run).

    So completion is decided entirely by the planner itself, in
    route_after_planner: the dry-run script naturally runs all 12 steps
    (including wrk) before it stops emitting tool_calls, and the live LLM
    is instructed to skip wrk only when there's truly no URL to test. That
    single signal — no tool_calls left — is what actually means "done" here.
    """
    return "planner"


# ─────────────────────────────────────────────────────────────
# BUILD THE GRAPH
# ─────────────────────────────────────────────────────────────

def build_benchmark_agent():
    graph = StateGraph(BenchmarkState)
    graph.add_node("planner",      planner_node)
    graph.add_node("execute_tool", tool_executor_node)
    graph.add_node("compare",      comparison_node)

    graph.set_entry_point("planner")
    graph.add_conditional_edges("planner", route_after_planner, {
        "execute_tool": "execute_tool",
        "compare":      "compare",
    })
    graph.add_conditional_edges("execute_tool", route_after_tool, {
        "planner": "planner",
        "compare": "compare",
    })
    graph.add_edge("compare", END)
    return graph.compile(checkpointer=MemorySaver())


# ─────────────────────────────────────────────────────────────
# RUNNER
# ─────────────────────────────────────────────────────────────

def run_benchmark_agent(user_prompt: str):
    global _dry_run_step
    _dry_run_step = 0

    agent = build_benchmark_agent()
    config = {"configurable": {"thread_id": "network-benchmark-1"}}

    initial_state = {
        "messages":           [HumanMessage(content=user_prompt)],
        "workload_params":    {},
        "iperf3_command":     "",
        "netperf_command":    "",
        "mtr_command":        "",
        "wrk_command":        "",
        "iperf3_raw_output":  "",
        "netperf_raw_output": "",
        "mtr_raw_output":     "",
        "wrk_raw_output":     "",
        "iperf3_results":     {},
        "netperf_results":    {},
        "mtr_results":        {},
        "wrk_results":        {},
        "comparison":         "",
        "error":              "",
        "final_answer":       "",
    }

    print("\n" + "=" * 65)
    print("NETWORK BENCHMARK AI AGENT")
    print("=" * 65)
    print(f"Prompt : {user_prompt}")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Mode   : {'DRY-RUN (simulated)' if DRY_RUN else 'LIVE'}")
    print("=" * 65 + "\n")

    result = agent.invoke(initial_state, config=config)

    print("\n" + "=" * 65)
    print("RESULTS")
    print("=" * 65)
    print(result.get("final_answer", "(no comparison produced)"))
    return result


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

def check_prerequisites():
    """
    Check for iperf3/netperf/mtr/wrk BEFORE calling the LLM at all — a live
    run with one missing would still spend a real Gemini call just to have
    the model report back that a binary was missing, something
    `shutil.which` answers instantly and for free.
    """
    missing = [t for t in ("iperf3", "netperf", "mtr", "wrk") if shutil.which(t) is None]
    if not missing:
        return
    print(f"ERROR: required benchmark tool(s) not found on this host: {', '.join(missing)}")
    if "iperf3" in missing:
        print("  Install iperf3:  sudo apt install iperf3 (Debian/Ubuntu) | brew install iperf3 (macOS)")
    if "netperf" in missing:
        print("  Install netperf: sudo apt install netperf (Debian/Ubuntu) | brew install netperf (macOS)")
    if "mtr" in missing:
        print("  Install mtr:     sudo apt install mtr (Debian/Ubuntu) | brew install mtr (macOS)")
    if "wrk" in missing:
        print("  Install wrk:     sudo apt install wrk (Debian/Ubuntu) | brew install wrk (macOS)")
    print("Or test the pipeline without real tools: export DRY_RUN=true")


if __name__ == "__main__":
    if not DRY_RUN and not os.environ.get("GOOGLE_API_KEY"):
        print("ERROR: GOOGLE_API_KEY not set.")
        print("Get a free key at https://aistudio.google.com")
        print("Or run in dry-run mode: export DRY_RUN=true")
        exit(1)

    if not DRY_RUN:
        check_prerequisites()
        if any(shutil.which(t) is None for t in ("iperf3", "netperf", "mtr", "wrk")):
            exit(1)

    data_plane_id = register_data_plane()
    start_heartbeat(data_plane_id)

    # ── Pick a prompt ──────────────────────────────────────
    prompt = (
        "Test the network path to 10.0.0.5: raw throughput (iperf3 and "
        "netperf are both running there on default ports), path latency "
        "with mtr, and HTTP load on http://10.0.0.5/health for 10 seconds "
        "with 2 threads and 10 connections."
    )
    # prompt = "Test raw throughput and path latency to 192.168.1.1 (our core switch) — no HTTP endpoint to test there."

    try:
        result = run_benchmark_agent(prompt)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output = {
            "prompt":           prompt,
            "timestamp":        timestamp,
            "dry_run":          DRY_RUN,
            "iperf3_command":   result.get("iperf3_command"),
            "netperf_command":  result.get("netperf_command"),
            "mtr_command":      result.get("mtr_command"),
            "wrk_command":      result.get("wrk_command"),
            "iperf3_results":   result.get("iperf3_results"),
            "netperf_results":  result.get("netperf_results"),
            "mtr_results":      result.get("mtr_results"),
            "wrk_results":      result.get("wrk_results"),
            "comparison":       result.get("comparison"),
        }
        out_path = f"/tmp/network_benchmark_results_{timestamp}.json"
        with open(out_path, "w") as f:
            json.dump(output, f, indent=2)
        print(f"\nFull results saved to: {out_path}")
    finally:
        stop_heartbeat()
