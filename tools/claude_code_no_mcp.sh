#!/usr/bin/env bash

if [ $# -eq 0 ]; then
    echo "Usage: $0 <prompt>"
    echo "Example: $0 'Fix the bug in main.py'"
    exit 1
fi

# TODO - run this in a sandbox environment (e.g. Docker), and use --dangerously-skip-permissions
claude \
    --strict-mcp-config \
    --permission-mode acceptEdits \
    -p "$1"
