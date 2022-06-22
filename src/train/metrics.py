import os
from datetime import datetime
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score, classification_report
from IPython.display import clear_output

from utils.paths import RESULTS_DIR


class MetricsMonitor:
    def __init__(
            self,
            label_names: List[str],
            empty_id: int,
            padding_id: int,
            experiment_name: Optional[str] = None,
            print_clear_output: bool = False,
            show_plots: bool = True,
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
        self.show_plots = show_plots

        if experiment_name is None:
            experiment_name = datetime.now().strftime("%D_%m_%H_%M_%S").replace('/', '_')
        self.target_dir = RESULTS_DIR / experiment_name
        os.makedirs(self.target_dir, exist_ok=False)

    def update_quality(self, y_true, y_pred):
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        confusion_labels = [self.label_names[i] for i in self.labels_no_padding]

        self.report = str(confusion_labels) + '\n' + \
                      str(confusion_matrix(y_true, y_pred, labels=self.labels_no_padding)) + '\n' + \
                      classification_report(y_true, y_pred, labels=self.labels_no_padding,
                                            target_names=self.label_names)

        selected_idx = np.where(lambda i: i in self.labels_selected, np.repeat(True, y_true.shape),
                                np.repeat(False, y_true.shape))
        print(selected_idx)
        print(y_true[selected_idx])

        acc = accuracy_score(y_true[selected_idx], y_pred[selected_idx])
        f1 = f1_score(y_true[selected_idx], y_pred[selected_idx], average="macro")

        self.acc_scores.append(acc)
        self.f1_scores.append(f1)

    def update_losses(self, train_loss, val_loss):
        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)

    def print_report(self, print_figure=True):
        if self.print_clear_output:
            clear_output()

        if self.report is not None:
            print(self.report)

        print(f'Validation:\nf1_score: {self.f1_scores[-1]}')
        print(f'accuracy: {self.acc_scores[-1]}')

        if not print_figure:
            return

        plt.figure(figsize=(8, 6))
        plt.grid(alpha=0.4)
        plt.plot(self.acc_scores, label='accuracy')
        plt.plot(self.f1_scores, label='f1_macro')
        plt.legend()
        plt.xlabel("Epoch")
        plt.ylabel("Metric")

        plt.savefig(str(self.target_dir / 'scores.png'))
        if self.show_plots:
            plt.show()

        if len(self.train_losses) > 0:
            print(f'Epoch {len(self.train_losses)}:')
            print("_________")
            print(f'train_loss = {self.train_losses[-1]}, val_loss = {self.val_losses[-1]}')

            plt.figure(figsize=(8, 6))
            plt.grid(alpha=0.4)
            plt.plot(self.train_losses, label='train')
            plt.plot(self.val_losses, label='val')
            plt.legend()
            plt.xlabel("Epoch")
            plt.ylabel("Loss")

            plt.savefig(str(self.target_dir / 'losses.png'))
            if self.show_plots:
                plt.show()
