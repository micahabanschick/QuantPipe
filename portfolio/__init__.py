"""Portfolio construction module — signals → target weights.

Phase 4 work. Core function signature:
    construct_portfolio(
        signals: pl.DataFrame,
        cov_matrix: np.ndarray,
        constraints: PortfolioConstraints,
    ) -> pl.DataFrame  # columns: [date, symbol, weight]

Starts with vol-scaled equal weighting; adds cvxpy/PyPortfolioOpt later.
"""
