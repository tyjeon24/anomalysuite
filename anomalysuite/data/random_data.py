"""Random data generator module."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class RandomDataGenerator:
    """Example data generator."""

    def __init__(
        self,
        length: int = 10000,
        anomaly_length: int = 50,
        features: int = 3,
        minmax_scale: bool = True,
        seed: int | None = None,
    ) -> None:
        """Random data generator for testing model.

        generator sinusoidal wave datasets with anomalies.

        Args:
            length: length of total data. Defaults to 10000.
            anomaly_length: anomaly length for each dimension.. Defaults to 50.
            features: number of features. Defaults to 3.
            minmax_scale: minmax scale if True. Defaults to True.
            seed: random seed to repeat. Defaults to None.

        Examples:
            data_generator = RandomDataGenerator(
                length=1000,
                anomaly_length=30,
                features=5,
                seed=22,
            )
            df = data_generator.get_data()
            data_generator.plot()

        """
        self.length = length
        self.anomaly_length = anomaly_length
        self.features = features
        self.minmax_scale = minmax_scale
        self.seed = seed
        self.random_number_generator = np.random.default_rng(seed=self.seed)
        self.anomaly_spans: list[tuple[int, int]] = []

    def get_data(self) -> pd.DataFrame:
        """Generate data.

        generate arbitrary sinusodial wave data.

        Returns:
            pandas dataframe.

        """
        data = {}
        for i in range(self.features):
            x = np.arange(0, self.length)
            phase = self.random_number_generator.uniform(0, 2 * np.pi)
            amplitude = self.random_number_generator.uniform(0, 10)
            frequency = self.random_number_generator.uniform(0.02, 0.05)
            y = amplitude * np.sin(2 * np.pi * frequency * x - phase)

            start = self.random_number_generator.integers(int(self.length * 0.8), self.length - self.anomaly_length)
            end = start + self.anomaly_length
            y[start:end] = y[start:end] + (self.random_number_generator.random(self.anomaly_length) - 0.5)
            self.anomaly_spans.append((start, end))
            if self.minmax_scale:
                y = (y - y.min()) / (y.max() - y.min())
            data[f"feature_{i + 1}"] = y

        df = pd.DataFrame(data)
        return df

    def plot(self) -> None:
        """Plot data.

        plot data and anomaly spans.
        """
        df = self.get_data()

        plt.figure(figsize=(21, 9))
        for column in df.columns:
            plt.plot(df[column], label=column)
        for anomaly_start, anomaly_end in self.anomaly_spans:
            plt.axvspan(anomaly_start, anomaly_end, color="red", alpha=0.2, label="Anomaly Section")
        plt.legend()
        plt.show()
