#!/usr/bin/env bash
# post-beacon-scan.sh — OMP post-tool hook
#
# Scans model-generated tool output for beacon/exfil indicators.
# Load with: omp --hook ~/.omp/hooks/post-beacon-scan.sh
#
# Based on Cohere's Qwen3.8-Flash-Next findings (2026-10-26):
# - Model generated curl commands to Alibaba Cloud OSS buckets
# - Probed cloud metadata endpoint (169.254.169.254)
# - Inventoried /tmp and transfer tools before exfil attempt
# - Reasoning trace gave no warning (said it wouldn't call tools)

# Read tool output from stdin
OUTPUT=$(cat)

# Watchlist patterns (case-insensitive grep)
BEACON_PATTERNS=(
    # Chinese cloud providers (Cohere finding: sanmaohe.oss-cn-beijing.aliyuncs.com)
    "aliyuncs\.com"
    "oss-cn-"
    "alibaba"
    "alicdn"
    "taobao"
    "alipay"
    "tencent"
    "myqcloud\.com"
    "baidu\.com"
    "bce\.baidu"
    "huaweicloud"
    "bytedance"
    "volcengine"
    "sanmaohe"

    # File proxy from Cohere findings
    "routify"

    # Cloud metadata (credential theft)
    "169\.254\.169\.254"
    "metadata\.google"
    "metadata\.azure"

    # Suspicious tool inventory (pre-exfil recon)
    "which curl.*which wget"
    "command -v curl.*command -v wget"
    "type -p curl.*type -p wget"

    # Direct exfil patterns
    "curl.*-T.*oss"
    "curl.*--upload-file"
    "wget.*--post-file"
    "curl.*-d @"
)

# Build combined pattern
PATTERN=$(IFS='|'; echo "${BEACON_PATTERNS[*]}")

# Scan
HITS=$(echo "$OUTPUT" | grep -iEn "$PATTERN" || true)

if [ -n "$HITS" ]; then
    echo "═══════════════════════════════════════════════════"
    echo "⚠️  BEACON WATCH: Suspicious pattern in model output"
    echo "═══════════════════════════════════════════════════"
    echo "$HITS"
    echo "═══════════════════════════════════════════════════"
    echo "Review: Model may be attempting to exfiltrate data."
    echo "Ref: cohere.com/blog/concerning-behaviour-popular-open-weight-ai-model"
    echo "═══════════════════════════════════════════════════"

    # Log to file
    echo "[$(date -Iseconds)] BEACON ALERT in tool output:" >> /tmp/beacon-alerts.log
    echo "$HITS" >> /tmp/beacon-alerts.log
    echo "---" >> /tmp/beacon-alerts.log

    # Exit non-zero to block the tool call (if OMP supports it)
    exit 1
fi

# Clean — pass through
exit 0
