from typing import List

import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix, f1_score, accuracy_score, classification_report
from IPython.display import clear_output


# TODO: save plots
# TODO: quality metrics
# TODO: optional losses
# TODO: custom evaluation?
class MetricsMonitor:
    def __init__(
        self,
        label_names: List[str],
        empty_id: int,
        padding_id: int,
        print_clear_output: bool = False,
    ):
        self.label_names = label_names

        self.train_losses = []
        self.val_losses = []

        self.acc_scores = []
        self.f1_scores = []

        self.report = None

        self.empty_id = empty_id
        self.padding_id = padding_id
        
        all_labels = set(range(len(self.label_names)))

        self.labels_no_padding = list(sorted(all_labels - {self.padding_id}))
        self.labels_selected = list(sorted(all_labels - {self.padding_id, self.empty_id}))
        self.label_names = label_names

        self.print_clear_output = print_clear_output


    def update_quality(self, y_true, y_pred):
        confusion_labels = [self.label_names[i] for i in self.labels_no_padding]

        self.report = str(confusion_labels) + '\n' +\
                str(confusion_matrix(y_true, y_pred, labels=self.labels_no_padding)) + '\n' +\
                classification_report(y_true, y_pred, labels=self.labels_no_padding, target_names=self.label_names)

        selected_idx = y_true[self.labels_selected]

        acc = accuracy_score(y_true[selected_idx], y_pred[selected_idx])
        f1 = f1_score(y_true[selected_idx], y_pred[selected_idx], average="macro")

        self.acc_scores.append(acc)
        self.f1_scores.append(f1)

    
    def update_losses(self, train_loss, val_loss):
        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)
        
    def print_report(self):
        if self.print_clear_output:
            clear_output()
        
        if self.report is not None:
            print(self.report)

        print(f'Validation:\nf1_score: {self.f1_scores[-1]}')
        print(f'accuracy: {self.acc_scores[-1]}')

        if len(self.train_losses) > 0:
            print(f'Epoch {len(self.train_losses)}:')
            print("_________")
            print(f'train_loss = {self.train_losses[-1]}, val_loss = {self.val_losses[-1]}')

            plt.plot(self.train_losses, label='train_loss')
            plt.plot(self.val_losses, label='val_loss')
            plt.show()
