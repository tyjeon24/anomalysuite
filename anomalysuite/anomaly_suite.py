"""Main module for anomaly suite project."""

import lightning as L


class AnomalySuite:
    """Meta model to use all-in-one anomaly detection."""

    def __init__(
        self,
        reconstructor: L.LightningModule,
        scorer: L.LightningModule,
        classifier: L.LightningModule,
    ) -> None:
        """Integrate three models for training and prediction.

        Args:
            reconstructor : the reconstruction based model.
            scorer : the model which converts error = x - x_hat to score.
            classifier : the model to check the data is whether anomaly(1) or not(0).

        """
        self.reconstructor = reconstructor
        self.scorer = scorer
        self.classifier = classifier
