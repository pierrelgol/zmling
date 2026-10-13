#!/bin/sh
#
# Creates patches/patches/NNN_name.patch from your solution in
# answers/<chapter>/NNN_name.zig (answers/ is git-ignored).
#
# Usage, from the project root:  ./patches/mkpatch.sh [number ...]   (default: all answers)
#
if [ ! -f patches/mkpatch.sh ]; then
    echo "Run me from the project root directory."
    exit 1
fi

if [ $# -eq 0 ]; then set -- $(ls answers/*/[0-9]*.zig | sed 's|.*/0*\([0-9]*\)_.*|\1|'); fi

for n in "$@"; do
    n=${n#0}
    n=${n#0}
    f=$(ls exercises/*/"$(printf '%03d' "$n")"_*.zig)
    rel=${f#exercises/}
    if [ ! -f "answers/$rel" ]; then
        echo "No answer for $rel"
        exit 1
    fi
    # diff exits 1 when files differ, which is what we want.
    if diff -u --label "exercises/$rel" --label "answers/$rel" "$f" "answers/$rel" > "patches/patches/$(basename "$f" .zig).patch"; then
        echo "answers/$rel is identical to the exercise: nothing is broken!"
        exit 1
    fi
done
