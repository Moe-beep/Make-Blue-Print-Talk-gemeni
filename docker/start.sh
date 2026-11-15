#!/bin/sh
# Starts the PlanQA pipeline (Python, internal port 8765) and the web server (Node, $PORT).
# If one of them exits, the other is stopped too, so the container restarts as a whole.
set -u

cd /app/pipeline
# shellcheck disable=SC2086
python serve.py --host 127.0.0.1 --port 8765 --out out/service $PLANQA_ARGS &
PY_PID=$!

cd /app
node --import tsx server.ts &
NODE_PID=$!

trap 'kill $PY_PID $NODE_PID 2>/dev/null; exit 0' TERM INT

while kill -0 "$PY_PID" 2>/dev/null && kill -0 "$NODE_PID" 2>/dev/null; do
  sleep 2
done
echo "start.sh: a process exited (pipeline alive: $(kill -0 $PY_PID 2>/dev/null && echo yes || echo no)," \
     "web alive: $(kill -0 $NODE_PID 2>/dev/null && echo yes || echo no)) - stopping container" >&2
kill "$PY_PID" "$NODE_PID" 2>/dev/null
exit 1
