import numpy as np


class RandomDaysSplitter:
    def __init__(self, days_of_year: np.ndarray, n_splits: int, n_intervals_per_split: int, n_days_per_interval: int,
                 random_state: int = None):
        """
        Split input data `n_splits` times. Each split will draw `n_intervals_per_split` of length `n_days_per_interval`
        from input data. In other words, each split contains `n_intervals` * `n_days` days in total.
        """
        # Get number of unique days
        self.days = np.unique(days_of_year)
        self.days_idxs = np.arange(len(self.days))
        self.days_of_year = days_of_year
        self.n_samples = len(days_of_year)

        # Reproducible numpy randomness
        self.rng = np.random.default_rng(seed=random_state)

        self.n_splits = n_splits
        self.n_intervals = n_intervals_per_split
        self.n_days = n_days_per_interval

    def split(self, X, y=None, groups=None):
        def get_random_interval() -> np.ndarray:
            # Draw from day index. Don't draw days directly because they can be non-continuous
            random_day_idx = self.rng.choice(self.days_idxs[:-self.n_days], size=1)[0]
            interval_idxs = [random_day_idx + i for i in range(self.n_days)]
            return self.days[interval_idxs]

        # Input has to match days of years provided in constructor
        assert len(X) == self.n_samples

        # Indices of X
        X_ind = np.arange(len(X))

        for _ in range(self.n_splits):
            selected_days = []
            for _ in range(self.n_intervals):
                # Draw random interval of length `n_days` and repeat if any of the days in interval was already selected
                days = get_random_interval()
                while np.any(np.isin(selected_days, days)):
                    days = get_random_interval()
                selected_days += days.tolist()

            is_selected = np.isin(self.days_of_year, selected_days)

            train = X_ind[~is_selected]
            test = X_ind[is_selected]

            yield train, test

