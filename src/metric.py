from torchmetrics import Metric
import torch


class MyAccuracy(Metric):
    def __init__(self):
        super().__init__()
        self.add_state('total', default=torch.tensor(0), dist_reduce_fx='sum')
        self.add_state('correct', default=torch.tensor(0), dist_reduce_fx='sum')

    def update(self, preds, target):
        # [TODO] The preds (B x C tensor), so take argmax to get index with highest confidence
        pred_value, pred_index = torch.max(preds, dim=1)

        # [TODO] check if preds and target have equal shape
        if pred_index.shape != target.shape:
            raise("Shape missmatch")

        # [TODO] Cound the number of correct prediction
        correct = (pred_index == target)

        # Accumulate to self.correct
        self.correct += torch.sum(correct)

        # Count the number of elements in target
        self.total += target.numel()

    def compute(self):
        return self.correct.float() / self.total.float()


# [TODO] Implement this!
class MyF1Score(Metric):
    def __init__(self, num_classes=200):
        super().__init__()
        self.num_classes = num_classes
        self.add_state('true_positive', default=torch.zeros(num_classes), dist_reduce_fx='sum')
        self.add_state('false_positive', default=torch.zeros(num_classes), dist_reduce_fx='sum')
        self.add_state('false_negative', default=torch.zeros(num_classes), dist_reduce_fx='sum')

    def update(self, preds, target):
        preds = torch.argmax(preds, dim=1)
        # class 별로 true_positive, false_positive, false_negative 모두 계산
        for cls in range(self.num_classes):
            tp_map = (preds == cls) & (target == cls)
            fp_map = (preds == cls) & (target != cls)
            fn_map = (preds != cls) & (target == cls)
            
            self.true_positive[cls] += torch.sum(tp_map)
            self.false_positive[cls] += torch.sum(fp_map)
            self.false_negative[cls] += torch.sum(fn_map)

    def compute(self):
        # ppt 17p 구현, zero div 방지
        precision = self.true_positive / (self.true_positive + self.false_positive + 1e-16)
        recall = self.true_positive / (self.true_positive + self.false_negative + 1e-16)
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-16)
        return torch.mean(f1_scores)