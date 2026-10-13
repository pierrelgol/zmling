#!/bin/sh
#
# The opposite of heal.sh: checks that every exercise, as shipped, FAILS.
# An exercise that passes without being fixed teaches nothing.
#
# Usage, from the project root:  ./patches/verify_broken.sh [first [last]]
#
if [ ! -f patches/verify_broken.sh ]; then
    echo "Run me from the project root directory."
    exit 1
fi

total=$(ls exercises/*/[0-9]*.zig | wc -l)
first=${1:-1}
last=${2:-$total}
saved_progress=$(cat .progress.txt 2>/dev/null || true)
bad=""

n=$first
while [ "$n" -le "$last" ]; do
    if out=$(zig build -Dn="$n" 2>&1); then
        echo "NOT BROKEN: exercise $n passes as shipped"
        bad="$bad $n"
    else
        # Show why it fails: a broken build setup must not pass for a broken exercise.
        why=$(echo "$out" | grep -m1 -e ' error: ' -e ' panic: ' -e 'expected this output' | cut -c1-150)
        echo "ok: exercise $n is broken: $why"
    fi
    n=$((n + 1))
done

if [ -n "$saved_progress" ]; then echo "$saved_progress" > .progress.txt; else rm -f .progress.txt; fi
[ -z "$bad" ]
