# SPDX-FileCopyrightText: 2025-present Matt Chandler <mc16535@bristol.ac.uk>
#
# SPDX-License-Identifier: MIT

__version__ = "0.0.1.a"

from .hypothesis_testing import mahalanobis, transform_observations
from .stats import cov

__all__ = [
    'cov', 'mahalanobis', 'transform_observations'
]