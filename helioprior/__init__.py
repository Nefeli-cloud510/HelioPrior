"""HelioPrior: LHAASO solar-wind forecast strategy evaluation and iteration.

The package keeps the competitive-hypothesis loop as the backend mechanism
that screens reliable space-weather precursor signals and forecast
strategies.
"""

from helioprior.service import HelioPriorService

__all__ = ["HelioPriorService"]
