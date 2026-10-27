#!/usr/bin/env bash
# beacon-watch-universal.sh — LLM inference exfiltration monitor
#
# Run this ON the machine hosting your inference server (DGX Spark, GPU box,
# cloud VM, laptop — anywhere you serve a local model). It watches for
# suspicious outbound network activity that could indicate the model is
# beaconing or exfiltrating data.
#
# Requirements: bash, one of { ss, netstat }, and optionally tcpdump (for
# DNS monitoring). That's it. No Python, no Node, no special permissions
# unless you enable packet capture.
#
# Usage:
#   ./beacon-watch-universal.sh                    # basic mode (ss polling)
#   sudo ./beacon-watch-universal.sh --dns         # + DNS packet capture
#   ./beacon-watch-universal.sh --port 8888        # only watch inference port
#   ./beacon-watch-universal.sh --container NAME   # watch a Docker container
#
# Background:
#   Cohere (2026-10-26) found Qwen3.8-Flash-Next generating commands to
#   upload files to Alibaba Cloud OSS in Beijing during agentic tool use.
#   This monitor detects that class of behavior at the network level.
#
# License: MIT

set -euo pipefail

# --- Config ---
POLL_INTERVAL="${BEACON_POLL_INTERVAL:-2}"
LOGFILE="${BEACON_LOGFILE:-/tmp/beacon-watch.log}"
DNS_MODE=false
INFERENCE_PORT=""
CONTAINER=""
ALERT_CMD="${BEACON_ALERT_CMD:-}"  # e.g. "curl -X POST https://your-webhook"

# --- Parse args ---
while [[ $# -gt 0 ]]; do
    case $1 in
        --dns) DNS_MODE=true; shift ;;
        --port) INFERENCE_PORT="$2"; shift 2 ;;
        --container) CONTAINER="$2"; shift 2 ;;
        --log) LOGFILE="$2"; shift 2 ;;
        --alert-cmd) ALERT_CMD="$2"; shift 2 ;;
        --interval) POLL_INTERVAL="$2"; shift 2 ;;
        -h|--help)
            echo "Usage: $0 [--dns] [--port PORT] [--container NAME] [--log FILE] [--alert-cmd CMD] [--interval SEC]"
            echo ""
            echo "Options:"
            echo "  --dns          Enable DNS packet capture (requires sudo/tcpdump)"
            echo "  --port PORT    Only flag connections from processes listening on this port"
            echo "  --container N  Monitor a specific Docker container's network namespace"
            echo "  --log FILE     Log alerts to file (default: /tmp/beacon-watch.log)"
            echo "  --alert-cmd C  Run this command on each alert (gets alert text as \$1)"
            echo "  --interval S   Poll interval in seconds (default: 2)"
            echo ""
            echo "Environment:"
            echo "  BEACON_POLL_INTERVAL  Poll interval (default: 2)"
            echo "  BEACON_LOGFILE        Log file path"
            echo "  BEACON_ALERT_CMD      Alert command"
            exit 0
            ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# --- Detect tools ---
SS_CMD=""
if command -v ss &>/dev/null; then
    SS_CMD="ss"
elif command -v netstat &>/dev/null; then
    SS_CMD="netstat"
else
    echo "ERROR: Need 'ss' or 'netstat'. Install iproute2 or net-tools."
    exit 1
fi

TCPDUMP_CMD=""
if $DNS_MODE; then
    if command -v tcpdump &>/dev/null; then
        TCPDUMP_CMD="tcpdump"
    else
        echo "WARNING: --dns requires tcpdump. Falling back to connection-only monitoring."
        DNS_MODE=false
    fi
fi

DOCKER_CMD=""
if [ -n "$CONTAINER" ]; then
    if command -v docker &>/dev/null; then
        DOCKER_CMD="docker"
    else
        echo "ERROR: --container requires docker."
        exit 1
    fi
fi

# --- Watchlist ---
# Chinese cloud providers
WATCH_DOMAINS=(
    "aliyuncs"          # Alibaba Cloud OSS (Cohere finding)
    "oss-cn-"           # Alibaba Cloud OSS region prefix
    "alibaba"
    "alicdn"
    "taobao"
    "alipay"
    "myqcloud"          # Tencent Cloud
    "tencent"
    "bce\.baidu"        # Baidu Cloud
    "huaweicloud"
    "volcengine"        # ByteDance cloud
    "bytedance"
    "sanmaohe"          # Specific bucket from Cohere finding
    "routify"           # File proxy from Cohere finding
)

# Suspicious IPs
WATCH_IPS=(
    "169\.254\.169\.254"    # Cloud metadata (credential theft)
)

# Known Alibaba Cloud IP ranges (major blocks)
# Source: https://ipinfo.io/AS45102
WATCH_CIDR_PATTERNS=(
    "^47\.74\."
    "^47\.88\."
    "^47\.89\."
    "^47\.90\."
    "^47\.91\."
    "^47\.254\."
    "^8\.208\."
    "^8\.209\."
    "^8\.210\."
    "^8\.211\."
    "^8\.212\."
    "^8\.213\."
    "^8\.214\."
    "^8\.215\."
    "^8\.216\."
    "^8\.217\."
    "^8\.218\."
    "^8\.219\."
    "^47\.236\."
    "^47\.237\."
    "^47\.238\."
    "^47\.239\."
    "^47\.245\."
    "^47\.246\."
    "^47\.250\."
    "^47\.251\."
    "^47\.252\."
    "^47\.253\."
    "^149\.129\."
    "^161\.117\."
)

DOMAIN_PATTERN=$(IFS='|'; echo "${WATCH_DOMAINS[*]}")
IP_PATTERN=$(IFS='|'; echo "${WATCH_IPS[*]}")
CIDR_PATTERN=$(IFS='|'; echo "${WATCH_CIDR_PATTERNS[*]}")
FULL_PATTERN="${DOMAIN_PATTERN}|${IP_PATTERN}"

# --- Functions ---
alert() {
    local msg="$1"
    local ts
    ts="$(date '+%Y-%m-%d %H:%M:%S')"

    # Terminal (red)
    echo -e "\033[1;31m[${ts}] ⚠ BEACON ALERT: ${msg}\033[0m" >&2

    # Log
    echo "[${ts}] ALERT: ${msg}" >> "$LOGFILE"

    # Desktop notification (best-effort)
    notify-send "BEACON ALERT" "$msg" 2>/dev/null || true

    # Custom alert command
    if [ -n "$ALERT_CMD" ]; then
        eval "$ALERT_CMD" "'$msg'" 2>/dev/null || true
    fi
}

get_connections() {
    local output
    if [ -n "$CONTAINER" ]; then
        local pid
        pid=$($DOCKER_CMD inspect --format '{{.State.Pid}}' "$CONTAINER" 2>/dev/null)
        if [ -z "$pid" ] || [ "$pid" = "0" ]; then
            return
        fi
        output=$(nsenter -t "$pid" -n $SS_CMD -tnp 2>/dev/null || true)
    else
        if [ "$SS_CMD" = "ss" ]; then
            output=$(ss -tnp 2>/dev/null || true)
        else
            output=$(netstat -tnp 2>/dev/null || true)
        fi
    fi
    echo "$output"
}

check_connections() {
    local conns
    conns=$(get_connections)

    # Check against domain/IP watchlist (for resolved connections in process names)
    local domain_hits
    domain_hits=$(echo "$conns" | grep -iE "$FULL_PATTERN" || true)
    if [ -n "$domain_hits" ]; then
        alert "Suspicious connection (domain match): $domain_hits"
    fi

    # Check against Alibaba Cloud CIDR ranges
    local peer_ips
    peer_ips=$(echo "$conns" | grep "ESTAB" | awk '{print $5}' | cut -d: -f1 | sort -u || true)
    for ip in $peer_ips; do
        if echo "$ip" | grep -qE "$CIDR_PATTERN"; then
            local proc_info
            proc_info=$(echo "$conns" | grep "$ip" | head -1)
            alert "Connection to Alibaba Cloud IP range: ${ip} — ${proc_info}"
        fi
    done
}

# --- Baseline ---
echo "╔══════════════════════════════════════════════════════╗"
echo "║  BEACON WATCH — LLM Exfiltration Monitor            ║"
echo "╠══════════════════════════════════════════════════════╣"
echo "║  Mode: $([ -n "$CONTAINER" ] && echo "container ($CONTAINER)" || echo "host")"
echo "║  Tool: ${SS_CMD}$(${DNS_MODE} && echo " + tcpdump")"
echo "║  Poll: ${POLL_INTERVAL}s"
echo "║  Log:  ${LOGFILE}"
[ -n "$INFERENCE_PORT" ] && echo "║  Port: ${INFERENCE_PORT}"
echo "╠══════════════════════════════════════════════════════╣"
echo "║  Watching for:                                      ║"
echo "║  • Alibaba Cloud (OSS, CDN, IP ranges)              ║"
echo "║  • Tencent / Baidu / ByteDance / Huawei cloud       ║"
echo "║  • Cloud metadata endpoint (169.254.169.254)        ║"
echo "║  • routify proxy (Cohere finding)                   ║"
echo "╠══════════════════════════════════════════════════════╣"
echo "║  Ctrl+C to stop. Clean output = no beacons.         ║"
echo "╚══════════════════════════════════════════════════════╝"
echo
echo "[$(date '+%H:%M:%S')] Taking connection baseline..." | tee "$LOGFILE"

BASELINE_PEERS=$(get_connections | grep "ESTAB" | awk '{print $5}' | sort -u)
BASELINE_COUNT=$(echo "$BASELINE_PEERS" | grep -c . || true)
echo "[$(date '+%H:%M:%S')] Baseline: ${BASELINE_COUNT} established peers" | tee -a "$LOGFILE"

# --- DNS capture (background, optional) ---
DNS_PID=""
if $DNS_MODE; then
    echo "[$(date '+%H:%M:%S')] Starting DNS capture..."
    $TCPDUMP_CMD -l -n -i any port 53 2>/dev/null | while IFS= read -r line; do
        if echo "$line" | grep -qiE "$DOMAIN_PATTERN"; then
            alert "Suspicious DNS query: $line"
        fi
    done &
    DNS_PID=$!
fi

cleanup() {
    [ -n "$DNS_PID" ] && kill "$DNS_PID" 2>/dev/null || true
    echo
    echo "[$(date '+%H:%M:%S')] Monitor stopped. Alerts logged to: $LOGFILE"
}
trap cleanup EXIT

# --- Main loop ---
TICK=0
while true; do
    check_connections

    # New peer detection (not in baseline)
    CURRENT_PEERS=$(get_connections | grep "ESTAB" | awk '{print $5}' | sort -u)
    NEW_PEERS=$(comm -23 <(echo "$CURRENT_PEERS") <(echo "$BASELINE_PEERS") 2>/dev/null || true)
    if [ -n "$NEW_PEERS" ]; then
        for peer in $NEW_PEERS; do
            PEER_IP=$(echo "$peer" | cut -d: -f1)
            # Check if new peer is in suspicious CIDR
            if echo "$PEER_IP" | grep -qE "$CIDR_PATTERN"; then
                alert "NEW connection to suspicious IP: $peer"
            fi
        done
        # Update baseline with known-good new peers
        BASELINE_PEERS="$CURRENT_PEERS"
    fi

    # Heartbeat every 60 ticks
    TICK=$((TICK + 1))
    if [ $((TICK % 30)) -eq 0 ]; then
        PEER_COUNT=$(echo "$CURRENT_PEERS" | grep -c . || true)
        echo "[$(date '+%H:%M:%S')] ✓ Clean — ${PEER_COUNT} peers, ${TICK} checks" | tee -a "$LOGFILE"
    fi

    sleep "$POLL_INTERVAL"
done
