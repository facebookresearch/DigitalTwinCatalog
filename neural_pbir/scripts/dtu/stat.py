# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np

avg = []
for scanid in [24, 37, 40, 55, 63, 65, 69, 83, 97, 105, 106, 110, 114, 118, 122]:
    cd = np.loadtxt(f"results/dtu/dtu_scan{scanid:d}/neural_surface_recon/results.txt")[
        -1
    ]
    avg.append(cd)
    print(f"{scanid:3d}: {cd:.2f}")

print(f"avg: {np.mean(avg):.2f}")
