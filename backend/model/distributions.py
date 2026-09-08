"""Shared distribution contract: published *_sd fields are standard deviations."""

import numpy as np


def student_t_scale(sd, degrees_of_freedom: float):
    if not np.isfinite(degrees_of_freedom) or degrees_of_freedom <= 2:
        raise ValueError("Student-t degrees of freedom must exceed 2")
    if np.any(~np.isfinite(sd)) or np.any(np.asarray(sd) <= 0):
        raise ValueError("Standard deviations must be finite and positive")
    return sd * np.sqrt((degrees_of_freedom - 2) / degrees_of_freedom)
