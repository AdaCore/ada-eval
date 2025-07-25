#!/usr/bin/env bash

if [ $# -eq 0 ]; then
    echo "Usage: $0 <prompt>"
    echo "Example: $0 'Fix the bug in main.py'"
    exit 1
fi

# TODO - run this in a sandbox environment (e.g. Docker), and use --dangerously-skip-permissions
echo "Dummy script executed with prompt: $1"

sleep_time=$((RANDOM % 3 + 1))
echo "Sleeping for $sleep_time seconds to simulate processing..."
sleep $sleep_time
