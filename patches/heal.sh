#!/bin/sh
#
# Heals every broken exercise with its patch into patches/healed, then runs
# them all. This is how we know each exercise can actually be solved.
#
# Usage, from the project root:
#     ./patches/heal.sh            heal and check everything
#     ./patches/heal.sh 17         heal everything, start checking at exercise 17
#
set -e

if [ ! -f patches/heal.sh ]; then
    echo "Run me from the project root directory."
    exit 1
fi

rm -rf patches/healed
cp -r exercises patches/healed

for broken in exercises/*/[0-9]*.zig; do
    name=$(basename "$broken" .zig)
    chapter=$(basename "$(dirname "$broken")")
    patch_name="patches/patches/$name.patch"
    if [ -f "$patch_name" ]; then
        patch -s -i "$patch_name" "patches/healed/$chapter/$name.zig"
    else
        echo "Cannot heal $name. No patch found."
        exit 1
    fi
done

zig fmt --check patches/healed

saved_progress=$(cat .progress.txt 2>/dev/null || true)
zig build -Dhealed -Ds="${1:-1}" && status=0 || status=$?

# Point Bazel back at the real exercises, restore the learner's progress.
ln -sfn ../exercises zml/zmling
if [ -n "$saved_progress" ]; then echo "$saved_progress" > .progress.txt; else rm -f .progress.txt; fi
exit $status
