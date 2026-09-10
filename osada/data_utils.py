import os
import pickle

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset


SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_random_seed(seed=SEED):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def to_tensor(data, device=DEVICE):
    if not isinstance(data, torch.Tensor):
        data = torch.tensor(data, dtype=torch.float32)
    return data.to(device, non_blocking=True)


def unit_energy_normalize_iq(data, eps=1e-12):
    arr = np.asarray(data, dtype=np.float32)
    power = np.mean(arr ** 2, axis=(1, 2), keepdims=True)
    scale = np.sqrt(np.maximum(power, eps)).astype(np.float32)
    return arr / scale


def load_source_data(pkl_path, modulation="QAM16", snr_list=None, normalize_iq=True):
    with open(pkl_path, "rb") as f:
        dataset = pickle.load(f, encoding="latin1")

    all_iq = []
    all_label = []
    all_snr = []

    snr_set = set(int(v) for v in snr_list) if snr_list is not None else None

    for (mod, snr), samples in dataset.items():
        if mod != modulation:
            continue
        if snr_set is not None and int(snr) not in snr_set:
            continue
        all_iq.append(samples["iq"])
        all_label.append(samples["label"])
        all_snr.extend([snr] * len(samples["iq"]))

    x_total = np.vstack(all_iq)
    y_total = np.vstack(all_label)
    snr_total = np.array(all_snr, dtype=np.int16)

    if normalize_iq:
        x_total = unit_energy_normalize_iq(x_total)

    shuffle_idx = np.random.permutation(x_total.shape[0])
    return x_total[shuffle_idx], y_total[shuffle_idx], snr_total[shuffle_idx]


def load_target_data(pkl_path, snr_list=None, normalize_iq=True):
    with open(pkl_path, "rb") as f:
        dataset = pickle.load(f, encoding="latin1")

    snr_set = set(int(v) for v in snr_list) if snr_list is not None else None

    all_iq = []
    for (_, snr), samples in dataset.items():
        if snr_set is not None and int(snr) not in snr_set:
            continue
        all_iq.append(samples["iq"])

    target_iq = np.vstack(all_iq)
    if normalize_iq:
        target_iq = unit_energy_normalize_iq(target_iq)

    shuffle_idx = np.random.permutation(target_iq.shape[0])
    return target_iq[shuffle_idx]


def split_source_train_val_test(data, labels, snrs, val_size=0.1, test_size=0.2, device=DEVICE, seed=SEED):
    x_train_val, x_test, y_train_val, y_test, snr_train_val, snr_test = train_test_split(
        data, labels, snrs, test_size=test_size, random_state=seed
    )
    x_train, x_val, y_train, y_val, snr_train, snr_val = train_test_split(
        x_train_val,
        y_train_val,
        snr_train_val,
        test_size=val_size / (1 - test_size),
        random_state=seed,
    )

    return (
        to_tensor(x_train, device),
        to_tensor(x_val, device),
        to_tensor(x_test, device),
        to_tensor(y_train, device),
        to_tensor(y_val, device),
        to_tensor(y_test, device),
        snr_train,
        snr_val,
        snr_test,
    )


def build_dataloader(data, labels=None, batch_size=32, shuffle=True, drop_last=True):
    if labels is not None:
        dataset = TensorDataset(data, labels)
    else:
        dataset = TensorDataset(data)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        pin_memory=False,
    )


