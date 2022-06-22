import sys

from data_utils.predict import predict_multiwindow
from metrics import MetricsMonitor
from models import load_model_checkpoint
from train.train_utils import get_test
from utils import get_default_device
from utils.mappings import NAME2ID

def main(model_name="base-256"):
    device = get_default_device()
    model = load_model_checkpoint(model_name)

    for i in range(8):
    # for i in range(0, 1):
    # for i in range(1, 2):
    # for i in range(2, 3):
    # for i in range(3, 4):
    # for i in range(4, 5):
    # for i in range(3, 8):
        n = 2 ** i
        print("Calc for", n, "predictions per token")
        test_dl = get_test(window_n=n)

        _xs, y_pred, y_true = predict_multiwindow(model, test_dl, device, windows_n=n, return_true=True)

        label_names = list(sorted(NAME2ID.keys(), key=lambda k: NAME2ID[k]))
        metrics_monitor = MetricsMonitor(label_names, empty_id=NAME2ID['_EMPTY'], padding_id=NAME2ID['_PAD'])
        metrics_monitor.update_quality(y_pred, y_true)
        metrics_monitor.print_report(False)


if __name__ == '__main__':
    main()
