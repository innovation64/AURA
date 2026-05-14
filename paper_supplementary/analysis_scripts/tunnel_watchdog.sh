#!/bin/bash
# Watchdog for the human-eval tunnel. Keeps an SSH reverse tunnel to
# localhost.run alive forever; logs the assigned URL to /tmp/aura-eval-url
# every time it (re)connects. If the SSH process dies for any reason,
# this loop restarts it within ~5 seconds. Cloudflared is intentionally
# avoided here because the local network blocks QUIC and breaks HTTP/2.
#
# Usage:
#   nohup bash scripts/tunnel_watchdog.sh > /tmp/aura-eval-watchdog.log 2>&1 &
#   disown
#
# Stop:
#   pkill -f tunnel_watchdog.sh
#   pkill -f "ssh.*localhost.run"

LOG=/tmp/aura-eval-watchdog.log
URL_FILE=/tmp/aura-eval-url
LOCAL_PORT=5050

while true; do
  ts=$(date '+%F %T')
  echo "[$ts] starting localhost.run SSH tunnel..." >> "$LOG"
  rm -f /tmp/lhr-tunnel.log
  ssh -o StrictHostKeyChecking=no \
      -o UserKnownHostsFile=/dev/null \
      -o ServerAliveInterval=20 \
      -o ServerAliveCountMax=3 \
      -o ExitOnForwardFailure=yes \
      -T \
      -R 80:localhost:${LOCAL_PORT} \
      nokey@localhost.run > /tmp/lhr-tunnel.log 2>&1 &
  ssh_pid=$!

  # Wait up to 30s for URL to appear in the log
  url=""
  for i in 1 2 3 4 5 6; do
    sleep 5
    url=$(grep -oE "https://[a-zA-Z0-9]+\.lhr\.life" /tmp/lhr-tunnel.log | head -1)
    [ -n "$url" ] && break
  done

  if [ -n "$url" ]; then
    echo "$url" > "$URL_FILE"
    echo "[$ts] tunnel up: $url (pid $ssh_pid)" >> "$LOG"
  else
    echo "[$ts] no URL after 30s, killing pid $ssh_pid" >> "$LOG"
    kill "$ssh_pid" 2>/dev/null
    sleep 5
    continue
  fi

  # Wait for the SSH process to die
  wait "$ssh_pid" 2>/dev/null
  exit_code=$?
  ts=$(date '+%F %T')
  echo "[$ts] tunnel pid $ssh_pid exited with $exit_code, restarting in 3s..." >> "$LOG"
  sleep 3
done
