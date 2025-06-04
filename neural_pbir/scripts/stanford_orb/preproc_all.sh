# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

dataroot="data/stanford_orb/blender_HDR"
command="python scripts/preprocess/stanford_orb.py"

# Process all scenes under dataroot
for datadir in $(ls -d "$dataroot"/*)
do
    scene=$(basename "$datadir")
    echo "Processing: $scene"
    $command "$dataroot"/"$scene"/
done
