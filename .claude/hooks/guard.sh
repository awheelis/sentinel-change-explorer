#!/usr/bin/env bash
# Guard hook: blocks destructive file operations even under --dangerously-skip-permissions
# Reads the tool input from CLAUDE_TOOL_INPUT env var (JSON)

set -euo pipefail

# Extract the command from the JSON input
COMMAND="$(echo "$CLAUDE_TOOL_INPUT" | python3 -c "import sys,json; print(json.load(sys.stdin).get('command',''))" 2>/dev/null || echo "")"

if [ -z "$COMMAND" ]; then
  exit 0
fi

# Normalize: collapse whitespace, lowercase for matching
NORM="$(echo "$COMMAND" | tr '\n' ' ' | sed 's/  */ /g')"

# --- BLOCKLIST PATTERNS ---
# Each pattern is checked against the normalized command string

BLOCKED=0
REASON=""

# SAFE DIRECTORY ALLOWLIST: if rm -rf targets only known-safe dirs, allow it
SAFE_RM_DIRS='(\.venv|__pycache__|node_modules|\.pytest_cache|\.mypy_cache|build|dist|\.egg-info|\.tox|\.cache|\.ruff_cache|\.coverage|htmlcov)'
if echo "$NORM" | grep -qEi 'rm\s+(-[a-zA-Z]*r)' && \
   echo "$NORM" | grep -qEi "rm\s+(-[a-zA-Z]*r[a-zA-Z]*)\s+${SAFE_RM_DIRS}"; then
  # This is a safe rm, skip all rm checks
  :
else
  # Block direct rm -rf / or rm -rf ~ or rm -rf $HOME
  if echo "$NORM" | grep -qEi 'rm\s+.*-r.*\s+(/\s|/\*|~/|~\s|\$HOME|/Users/[^/]+\s|/Users/[^/]+/\s|/Users/[^/]+/\*)'; then
    BLOCKED=1
    REASON="Blocked: rm targeting home or root directory"
  fi

  # Block rm -rf / rm -r on broad paths (., /, ~, *, etc.)
  if echo "$NORM" | grep -qEi 'rm\s+(-[a-zA-Z]*r[a-zA-Z]*\s+|--recursive\s+)(\.|/|~|\*|\$HOME|\$\(|`)'; then
    BLOCKED=1
    REASON="Blocked: recursive rm on broad path"
  fi

  # Block rm -rf with force on paths under /Users
  if echo "$NORM" | grep -qEi 'rm\s+(-[a-zA-Z]*r[a-zA-Z]*f|--force)\s+.*(/Users|/home)'; then
    BLOCKED=1
    REASON="Blocked: forced recursive rm on user directory"
  fi
fi

# Block git clean -fdx (destroys untracked files)
if echo "$NORM" | grep -qEi 'git\s+clean\s+.*-[a-zA-Z]*f'; then
  BLOCKED=1
  REASON="Blocked: git clean -f can destroy untracked files"
fi

# Block git reset --hard
if echo "$NORM" | grep -qEi 'git\s+reset\s+--hard'; then
  BLOCKED=1
  REASON="Blocked: git reset --hard can destroy uncommitted work"
fi

# Block git push --force to main/master
if echo "$NORM" | grep -qEi 'git\s+push\s+.*--force.*\s+(main|master)'; then
  BLOCKED=1
  REASON="Blocked: force push to main/master"
fi

# Block git checkout . or git restore . (discards all changes)
if echo "$NORM" | grep -qEi 'git\s+(checkout|restore)\s+\.\s*$'; then
  BLOCKED=1
  REASON="Blocked: discarding all working tree changes"
fi

# Block find ... -delete
if echo "$NORM" | grep -qEi 'find\s+(/|~|\.|/Users).*-delete'; then
  BLOCKED=1
  REASON="Blocked: find with -delete on broad path"
fi

# Block truncation of important files via > redirect
if echo "$NORM" | grep -qEi '>\s*(\.env|\.gitignore|Makefile|pyproject\.toml|setup\.py|requirements\.txt|CLAUDE\.md)'; then
  BLOCKED=1
  REASON="Blocked: overwriting critical project file via redirect"
fi

# Block chmod/chown on broad paths
if echo "$NORM" | grep -qEi '(chmod|chown)\s+.*-R\s+(/|~|/Users)'; then
  BLOCKED=1
  REASON="Blocked: recursive permission change on broad path"
fi

# Block dd writing to disk devices
if echo "$NORM" | grep -qEi 'dd\s+.*of=\s*/dev/'; then
  BLOCKED=1
  REASON="Blocked: dd write to device"
fi

# Block mkfs
if echo "$NORM" | grep -qEi 'mkfs'; then
  BLOCKED=1
  REASON="Blocked: filesystem format command"
fi

if [ "$BLOCKED" -eq 1 ]; then
  echo "HOOK_ERROR: $REASON" >&2
  echo "Command was: $COMMAND" >&2
  echo "If this is intentional, remove the guard hook temporarily." >&2
  exit 2
fi

exit 0
