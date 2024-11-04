from torchmetrics import Metric
import torch


class BestShotAccuracy(Metric):
    def __init__(self):
        """
        Best shot accuracy metric. The best shot is the shot with the highest confidence score. The metric is calculated
        as the ratio of correct best shots to the total number of samples.
        """
        super().__init__()
        self.add_state("correct", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds, target):
        assert preds.shape == target.shape
        assert preds.ndim == 2
        B, C = preds.shape

        preds = preds.argmax(1)
        # check if the best shot is correct
        for b in range(B):
            print(preds[b], target[b].nonzero())
            self.correct += int(preds[b] in target[b].nonzero())

        self.total += B

    def compute(self):
        return self.correct / self.total


if __name__ == '__main__':
    metric = BestShotAccuracy()
    preds = torch.tensor([[0.1, 0.9, 0.1], [0.1, 0.1, 0.9]])
    target = torch.tensor([[0, 1, 1], [0, 0, 1]])
    print(metric(preds, target))