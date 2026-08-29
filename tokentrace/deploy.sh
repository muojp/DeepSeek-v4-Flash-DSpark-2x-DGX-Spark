#!/usr/bin/env bash
# Deploy / control the tokentrace sampler on both DGX Spark nodes.
#
#   tokentrace/deploy.sh sync            copy this package to both nodes (~/tokentrace-src)
#   tokentrace/deploy.sh start [secs]    start samplers (head + worker); optional --duration
#   tokentrace/deploy.sh stop            SIGTERM both samplers, wait for clean exit
#   tokentrace/deploy.sh status          pid, control-port stat, trace file sizes
#   tokentrace/deploy.sh selftest        run the unit tests on the head node's python
#
# The worker is only reachable through the head (fabric IP), so every worker
# command hops via the head. Nothing here touches the vLLM containers.
set -euo pipefail

HEAD="${TT_HEAD:-192.168.0.100}"                # ssh reachable from the laptop
WORKER_VIA_HEAD="${TT_WORKER:-192.168.100.40}"  # fabric IP, from the head
WORKER_PEER="${TT_WORKER_PEER:-$WORKER_VIA_HEAD}"
REMOTE_SRC="${TT_REMOTE_SRC:-tokentrace-src}"
TRACE_DIR="${TT_TRACE_DIR:-~/tokentrace}"
VLLM_URL="${TT_VLLM_URL:-http://127.0.0.1:8888}"
PORT="${TT_CONTROL_PORT:-47001}"
HERE="$(cd "$(dirname "$0")/.." && pwd)"

h() { ssh -o BatchMode=yes -o ConnectTimeout=8 "$HEAD" "$@"; }
w() { ssh -o BatchMode=yes -o ConnectTimeout=8 "$HEAD" "ssh -o BatchMode=yes -o ConnectTimeout=8 $WORKER_VIA_HEAD $(printf '%q' "$*")"; }

sync_src() {
  echo "== sync → $HEAD:~/$REMOTE_SRC"
  tar -C "$HERE" --exclude __pycache__ -cf - tokentrace tests/conftest.py tests/test_tokentrace_*.py \
    | h "mkdir -p ~/$REMOTE_SRC && tar -C ~/$REMOTE_SRC -xf -"
  echo "== sync → worker via head"
  h "tar -C ~/$REMOTE_SRC -cf - tokentrace tests | ssh -o BatchMode=yes $WORKER_VIA_HEAD 'mkdir -p ~/$REMOTE_SRC && tar -C ~/$REMOTE_SRC -xf -'"
  h "cd ~/$REMOTE_SRC && python3 -m py_compile tokentrace/*.py && echo head: compiled"
  w "cd ~/$REMOTE_SRC && python3 -m py_compile tokentrace/*.py && echo worker: compiled"
}

start_one() {  # $1 = h|w, rest = sampler args
  local fn="$1"; shift
  "$fn" "cd ~/$REMOTE_SRC && mkdir -p $TRACE_DIR && \
    if [ -f $TRACE_DIR/sampler.pid ] && kill -0 \$(cat $TRACE_DIR/sampler.pid) 2>/dev/null; then echo 'already running pid '\$(cat $TRACE_DIR/sampler.pid); exit 0; fi; \
    nohup setsid python3 -m tokentrace sampler --dir $TRACE_DIR $* >> $TRACE_DIR/sampler.log 2>&1 < /dev/null & \
    echo \$! > $TRACE_DIR/sampler.pid; sleep 1; kill -0 \$(cat $TRACE_DIR/sampler.pid) && echo started pid \$(cat $TRACE_DIR/sampler.pid); tail -n 2 $TRACE_DIR/sampler.log"
}

start() {
  local dur="${1:-0}"
  echo "== worker"
  start_one w --role worker --control-port "$PORT" --duration "$dur"
  echo "== head"
  start_one h --role head --vllm-url "$VLLM_URL" --peer "$WORKER_PEER" --control-port "$PORT" --duration "$dur"
}

stop_one() {
  local fn="$1"
  "$fn" "if [ -f $TRACE_DIR/sampler.pid ]; then p=\$(cat $TRACE_DIR/sampler.pid); kill -TERM \$p 2>/dev/null && for i in 1 2 3 4 5 6 7 8 9 10; do kill -0 \$p 2>/dev/null || break; sleep 0.5; done; kill -0 \$p 2>/dev/null && echo 'still alive' || echo stopped; rm -f $TRACE_DIR/sampler.pid; tail -n 1 $TRACE_DIR/sampler.log; else echo no pidfile; fi"
}

stop() { echo "== head"; stop_one h; echo "== worker"; stop_one w; }

status() {
  for pair in "head:h:127.0.0.1" "worker:w:127.0.0.1"; do
    IFS=: read -r name fn addr <<< "$pair"
    echo "== $name"
    "$fn" "if [ -f $TRACE_DIR/sampler.pid ] && kill -0 \$(cat $TRACE_DIR/sampler.pid) 2>/dev/null; then \
        p=\$(cat $TRACE_DIR/sampler.pid); echo pid \$p cpu% \$(ps -o %cpu= -p \$p) rss_kb \$(ps -o rss= -p \$p); else echo 'not running'; fi; \
        cd ~/$REMOTE_SRC 2>/dev/null && python3 -m tokentrace stat $addr:$PORT | cut -c1-400; \
        ls -la $TRACE_DIR/*/ 2>/dev/null | tail -n +2 | grep -v '^total' | awk '{print \$5, \$9}'"
  done
}

selftest() {
  # the nodes' system python is PEP 668 managed → pytest lives in a private venv (system site-packages visible)
  h "cd ~/$REMOTE_SRC && { [ -x ~/tokentrace-venv/bin/python ] || python3 -m venv --system-site-packages ~/tokentrace-venv; } && \
     { ~/tokentrace-venv/bin/python -c 'import pytest' 2>/dev/null || ~/tokentrace-venv/bin/pip install -q pytest; } && \
     ~/tokentrace-venv/bin/python -m pytest -q tests 2>&1 | tail -n 4"
}

case "${1:-}" in
  sync) sync_src ;;
  start) shift; start "${1:-0}" ;;
  stop) stop ;;
  status) status ;;
  selftest) selftest ;;
  *) sed -n 2,12p "$0"; exit 2 ;;
esac
