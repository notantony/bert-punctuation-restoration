import sys

from data_utils.predict import predict
from metrics import MetricsMonitor
from models import load_model_checkpoint
from train.train_utils import get_test
from utils import get_default_device
from utils.mappings import NAME2ID

def main():
    device = get_default_device()
    model = load_model_checkpoint(sys.argv[1])
    test_dl = get_test()

    _xs, y_pred, y_true = predict(model, test_dl, device, return_true=True)

    label_names = list(sorted(NAME2ID.keys(), key=lambda k: NAME2ID[k]))
    metrics_monitor = MetricsMonitor(label_names, empty_id=NAME2ID['_EMPTY'], padding_id=NAME2ID['_PAD'])
    metrics_monitor.update_quality(y_pred, y_true)
    metrics_monitor.print_report()


if __name__ == '__main__':
    main()
