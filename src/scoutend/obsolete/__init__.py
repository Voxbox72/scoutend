"""The numpy implementations, kept as the original reference.

These are the estimators written first, by hand, to understand the maths. Most
of them are no longer used -- the pipeline takes Gamma and Ridge from
scikit-learn -- with one exception: ``PoissonRegressor`` still ships, because
scikit-learn's Poisson regression has no exposure offset (see
``scoutend/models/glm.py`` for the measurement behind that decision).

Kept because they are correct, short, and the clearest documentation of what the
library estimators compute.
"""
