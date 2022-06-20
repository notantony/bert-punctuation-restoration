import torch
from torch import nn
from tqdm import tqdm

from models import get_model_by_name, load_bert_classifier
from train.utils import get_train_dev
from train.metrics import MetricsMonitor
from utils import set_seed, get_default_device
from utils.mappings import NAME2ID



def train(
        model,
        optimizer,
        loss_fn,
        train_iter,
        valid_iter,
        device,
        metrics_monitor,
        n_epochs=10,
        grad_clipping=1.5,
        early_stopping_patience=None,
        seed=42,
):
    print(f'Running training on the device: {device}')
    print(f'Random seed: {seed}')
    set_seed(seed)

    best_loss = float('inf')

    # TODO: early stopping refactor
    early_stopping_patience_cnt = 0

    model.to(device)

    for _ in range(n_epochs):
        # Train loop
        model.train()
        train_loss = 0
        train_len = 0

        for x_batch, y_batch in tqdm(train_iter):
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            optimizer.zero_grad()
            out = model(x_batch)
            loss = loss_fn(out.flatten(end_dim=1), y_batch.flatten())
            train_loss += loss.item() * len(y_batch)
            train_len += len(y_batch)
            if grad_clipping is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clipping)

            loss.backward()
            optimizer.step()

        train_loss = train_loss / train_len

        # Validation loop
        model.eval()
        val_loss = 0
        val_len = 0
        val_out = []
        val_target = []

        with torch.no_grad():
            optimizer.zero_grad()

            for x_batch, y_batch in tqdm(valid_iter):
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)
                out = model(x_batch)
                out = out.flatten(end_dim=1)
                y_batch = y_batch.flatten()
                val_loss += loss_fn(out, y_batch).item() * len(y_batch)
                val_len += len(y_batch)
                out, target = out.cpu().detach().numpy(), y_batch.cpu().detach().numpy()

                out = np.argmax(out, axis=1)
                val_out.append(out)
                val_target.append(target)

        out = np.concatenate(val_out)
        target = np.concatenate(val_target)

        val_loss /= val_len

        metrics_monitor.update_losses(train_loss, val_loss)
        metrics_monitor.update_quality(target, out)
        metrics_monitor.print_report()

        if best_loss > val_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), 'best_model.pth')
            early_stopping_patience_cnt = 0
        else:
            early_stopping_patience_cnt += 1
            if early_stopping_patience is not None and early_stopping_patience_cnt >= early_stopping_patience:
                break


def main():
    device = get_default_device()
    train_dl, dev_dl = get_train_dev()

    model = load_bert_classifier('base-256')

    loss_fn = nn.CrossEntropyLoss(ignore_index=NAME2ID['_PAD'])

    optimizer = torch.optim.Adam([
        {'params': model.classifier_model.parameters(), 'lr': 1e-4},
        {'params': model.bert_model.parameters(), 'lr': 1e-5},
    ])

    label_names = list(sorted(NAME2ID.keys(), key=lambda k: NAME2ID[k]))

    metrics_monitor = MetricsMonitor(
        label_names,
        empty_id=NAME2ID['_EMPTY'],
        padding_id=NAME2ID['_PAD'],
        print_clear_output=False,
    )

    train(
        model,
        optimizer,
        loss_fn,
        train_dl,
        dev_dl,
        device=device,
        metrics_monitor=metrics_monitor,
        n_epochs=5,
        grad_clipping=1.5,
    )


if __name__ == '__main__':
    main()
