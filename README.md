# Network Benchmark Agent

An AI agent that runs `iperf3`, `netperf`, `mtr`, and `wrk` against a
target from a plain-English description of the network path or endpoint to
test — a proxy, a web server, a switch/router hop, a load balancer — via
LangGraph + Gemini.

> Rename this file to `README.md` in its own repo (`network_benchmark_agent.py`
> lives alongside `storage_benchmark_agent.py` and `database_benchmark_agent.py`
> in this output bundle, but each is meant to be its own GitHub repo).

## Prerequisites — install these BEFORE running for real

```bash
# Ubuntu/Debian
sudo apt install iperf3 netperf mtr wrk

# RHEL/CentOS
sudo yum install iperf3 netperf mtr wrk

# macOS
brew install iperf3 netperf mtr wrk
```

**iperf3 and netperf both need a matching server daemon already running on
the target** before this agent's client-side commands can connect:

```bash
# on the target host
iperf3 -s -p 5201
netserver -p 12865
```

This agent cannot start a server on a remote host for you (no SSH/remote-
exec here) — if the target isn't running one, the command fails with a
clear connection error rather than a fabricated result.

`mtr` and `wrk` need no server-side setup — they work against any reachable
host or HTTP endpoint.

**Want to test the agent itself without any of these installed?** Use
dry-run mode instead (see below).

## What gets compared, and what doesn't

These four tools measure different things, so results are grouped by what
they actually measure rather than forced into one table:

- **iperf3 vs netperf** — both measure raw TCP throughput in the same
  unit (Mbps), so these two genuinely are compared side by side.
- **mtr** — hop-by-hop path latency and packet loss. A path-health signal,
  not a throughput number; not merged into the throughput comparison.
- **wrk** — HTTP-layer request throughput against a specific web endpoint.
  Only run when the request actually names a URL — meaningful for a web
  server, reverse proxy, or load balancer's HTTP frontend, not a general
  network throughput substitute.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Running

**Dry-run (no tools, no target, no API key, no cost):**
```bash
export DRY_RUN=true
python3 network_benchmark_agent.py
```

**Live run (real iperf3/netperf/mtr/wrk, real Gemini calls):**
```bash
export GOOGLE_API_KEY="your-key"   # free tier: https://aistudio.google.com
python3 network_benchmark_agent.py
```

Edit the `prompt` variable near the bottom of the file (in the
`if __name__ == "__main__":` block) to describe your own target — host,
ports for iperf3/netperf, hop count for mtr, and a URL if you want wrk's
HTTP test included.

## AgentLease control-plane integration (optional)

If you're running this agent as part of an AgentLease lease rather than
standalone, set these before running so it registers itself and
heartbeats:

```bash
export CONTROL_PLANE_URL="https://your-control-plane-url"
export PAIRING_TOKEN="uz_live_..."   # from your lease
export DATA_PLANE_NAME="my-host"     # optional, defaults to hostname
```

Leave these unset to run the agent completely standalone — it works
identically either way; the control-plane calls are a no-op without them.

## Rate limits

Gemini's free tier has both per-minute and per-day request limits. This
agent throttles and retries automatically on transient rate-limit (429)
and server-overload (503) errors, but a genuinely exhausted **daily**
quota can't be retried around — check https://ai.dev/rate-limit if a run
keeps failing with `RESOURCE_EXHAUSTED` despite several retries.

## Output

Full results (all four commands, raw outputs, parsed metrics, and the
grouped comparison summary) are saved as JSON to
`/tmp/network_benchmark_results_<timestamp>.json` after every run.
