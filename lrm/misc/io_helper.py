# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os

from .dist_helper import get_rank


def mkdirs(dirpath):
    if get_rank() == 0 and (not os.path.isdir(dirpath)):
        os.makedirs(dirpath)
    return dirpath
