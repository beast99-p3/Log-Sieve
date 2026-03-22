#!/usr/bin/env bash
#
# Log-Sieve — simulate heterogeneous links with Linux tc (traffic control).
#
# Requirements: Linux, iproute2 (`tc`, `ip`), typically sudo/CAP_NET_ADMIN.
# Prefer a dedicated veth / Docker network namespace — do not throttle your SSH NIC.
#
# Example workflow (conceptual):
#   sudo ./simulate_network.sh apply veth-client-a 10mbit 80ms    # slow straggler
#   sudo ./simulate_network.sh apply veth-client-b 1000mbit 2ms   # fast node
#   # run Flower; server uses --round-timeout so slow clients do not block forever
#   sudo ./simulate_network.sh clear veth-client-a
#   sudo ./simulate_network.sh clear veth-client-b

set -euo pipefail

usage() {
  sed -n '1,22p' "$0"
  echo ""
  echo "Usage:"
  echo "  $0 apply <interface> [rate] [latency]"
  echo "  $0 clear <interface>"
  echo ""
  echo "Examples:"
  echo "  sudo $0 apply eth1 10mbit 80ms"
  echo "  sudo $0 apply eth1 1000mbit 2ms"
  echo "  sudo $0 clear eth1"
}

cmd_apply() {
  local iface="$1"
  local rate="${2:-10mbit}"
  local latency="${3:-50ms}"
  if [[ -z "$iface" ]]; then
    echo "error: interface required" >&2
    usage >&2
    exit 1
  fi
  sudo tc qdisc replace dev "$iface" root handle 1: htb default 30
  sudo tc class add dev "$iface" parent 1: classid 1:1 htb rate "$rate" ceil "$rate" 2>/dev/null \
    || sudo tc class change dev "$iface" parent 1: classid 1:1 htb rate "$rate" ceil "$rate"
  sudo tc class add dev "$iface" parent 1: classid 1:30 htb rate "$rate" ceil "$rate" 2>/dev/null \
    || sudo tc class change dev "$iface" parent 1: classid 1:30 htb rate "$rate" ceil "$rate"
  sudo tc qdisc replace dev "$iface" parent 1:30 handle 20: netem delay "$latency"
  echo "Applied on $iface: HTB rate=$rate, netem delay=$latency"
}

cmd_clear() {
  local iface="$1"
  if [[ -z "$iface" ]]; then
    echo "error: interface required" >&2
    usage >&2
    exit 1
  fi
  sudo tc qdisc del dev "$iface" root 2>/dev/null || true
  echo "Cleared qdisc on $iface"
}

case "${1:-}" in
  apply)
    cmd_apply "${2:-}" "${3:-10mbit}" "${4:-50ms}"
    ;;
  clear)
    cmd_clear "${2:-}"
    ;;
  -h|--help|help|"")
    usage
    exit 0
    ;;
  *)
    echo "unknown command: ${1:-}" >&2
    usage >&2
    exit 1
    ;;
esac
