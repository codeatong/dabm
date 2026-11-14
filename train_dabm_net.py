import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

try:
    from dabm.cbmc import ContinuousBehaviorMatrixConverter
    from dabm.dsff import (
        DualScaleFeatureFusion,
        fit_dsff_statistics,
    )
    from dabm.dtti_net import DualTopologyTemporalInteractionNetwork
except ImportError:
    from cbmc import ContinuousBehaviorMatrixConverter
    from dsff import (
        DualScaleFeatureFusion,
        fit_dsff_statistics,
    )
    from dtti_net import DualTopologyTemporalInteractionNetwork


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TEMPORAL_PATH = (
    PROJECT_ROOT / "dabm" / "calendar_temporal_tensor.npy"
)
DEFAULT_METADATA_PATH = (
    PROJECT_ROOT / "dabm" / "calendar_metadata.npz"
)
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "dabm" / "dabm_net_output"
EXPECTED_SCHEMA_VERSION = "tsgcm_raw_voc_sms_compact_global_v3"


class TelecomFraudDataset(Dataset):
    """Load original users and optional coordinate-preserving views."""

    def __init__(
        self,
        temporal_matrices,
        global_features,
        labels,
        indices,
        target_size=None,
        augmentation_seed=0,
        event_dropout_rate=0.0,
        value_jitter_std=0.0,
        day_mask_probability=0.0,
        max_masked_days=0,
    ):
        self.temporal_matrices = temporal_matrices
        self.global_features = global_features
        self.labels = labels
        self.indices = np.asarray(indices, dtype=np.int64)
        self.original_size = int(self.indices.size)
        self.event_dropout_rate = float(event_dropout_rate)
        self.value_jitter_std = float(value_jitter_std)
        self.day_mask_probability = float(day_mask_probability)
        self.max_masked_days = int(max_masked_days)

        requested_size = (
            self.original_size
            if target_size is None
            else int(target_size)
        )
        if requested_size < self.original_size:
            raise ValueError(
                "target_size cannot be smaller than the original split."
            )
        if self.original_size == 0 and requested_size > 0:
            raise ValueError("Cannot augment an empty split.")

        extra_count = requested_size - self.original_size
        random_state = np.random.RandomState(int(augmentation_seed))
        extra_indices = []
        while extra_count > 0:
            shuffled = self.indices.copy()
            random_state.shuffle(shuffled)
            take = min(extra_count, self.original_size)
            extra_indices.append(shuffled[:take])
            extra_count -= take

        if extra_indices:
            extra_indices = np.concatenate(extra_indices)
            self.sample_indices = np.concatenate(
                [self.indices, extra_indices]
            )
            self.is_augmented = np.concatenate(
                [
                    np.zeros(self.original_size, dtype=bool),
                    np.ones(extra_indices.size, dtype=bool),
                ]
            )
        else:
            self.sample_indices = self.indices.copy()
            self.is_augmented = np.zeros(
                self.original_size,
                dtype=bool,
            )

    def __len__(self):
        return len(self.sample_indices)

    def augment_temporal(self, temporal):
        active = temporal > 0
        if self.event_dropout_rate > 0:
            keep = torch.rand_like(temporal) >= self.event_dropout_rate
            temporal = temporal * (keep | ~active).to(temporal.dtype)

        if self.value_jitter_std > 0:
            jitter = torch.exp(
                torch.randn_like(temporal) * self.value_jitter_std
            )
            temporal = torch.where(active, temporal * jitter, temporal)

        if (
            self.max_masked_days > 0
            and temporal.shape[-1] > 0
            and torch.rand(()).item() < self.day_mask_probability
        ):
            maximum_width = min(
                self.max_masked_days,
                int(temporal.shape[-1]),
            )
            width = int(
                torch.randint(1, maximum_width + 1, (1,)).item()
            )
            maximum_start = int(temporal.shape[-1]) - width + 1
            start = int(
                torch.randint(0, maximum_start, (1,)).item()
            )
            temporal[:, :, start : start + width] = 0.0

        return torch.clamp(temporal, min=0.0)

    def __getitem__(self, position):
        user_index = int(self.sample_indices[position])

        # Copy one user at a time so PyTorch receives writable arrays while
        # the complete temporal tensor remains memory mapped.
        temporal = np.array(
            self.temporal_matrices[user_index],
            dtype=np.float32,
            copy=True,
        )
        global_vector = np.array(
            self.global_features[user_index],
            dtype=np.float32,
            copy=True,
        )
        label = int(self.labels[user_index])

        temporal_tensor = torch.from_numpy(temporal)
        if bool(self.is_augmented[position]):
            temporal_tensor = self.augment_temporal(temporal_tensor)

        return (
            temporal_tensor,
            torch.from_numpy(global_vector),
            torch.tensor(label, dtype=torch.long),
            torch.tensor(user_index, dtype=torch.long),
        )


class DABMNet(nn.Module):
    """DSFF -> CBMC -> DTTI-Net -> classifier."""

    def __init__(self, dsff, cbmc, dtti):
        super().__init__()
        self.dsff = dsff
        self.cbmc = cbmc
        self.dtti = dtti

    def forward(self, temporal_features, global_features):
        dsff_outputs = self.dsff(
            temporal_features,
            global_features,
        )
        cbmc_outputs = self.cbmc(
            dsff_outputs["weighted_features"],
            return_components=True,
        )
        behavior_matrix = cbmc_outputs["behavior_matrix"]
        dtti_outputs = self.dtti(
            behavior_matrix,
            return_components=True,
        )

        return {
            "logits": dtti_outputs["logits"],
            "behavior_matrix": behavior_matrix,
            "base_behavior_matrix": cbmc_outputs[
                "base_behavior_matrix"
            ],
            "anomaly_intensity": cbmc_outputs[
                "anomaly_intensity"
            ],
            "anomaly_scale": cbmc_outputs["anomaly_scale"],
            "feature_weights": dsff_outputs["feature_weights"],
            "encoded_features": dtti_outputs["encoded_features"],
            "last_routing_weights": dtti_outputs[
                "last_routing_weights"
            ],
            "mean_topology_weights": dtti_outputs[
                "mean_topology_weights"
            ],
        }


def parse_arguments():
    parser = argparse.ArgumentParser(
        description=(
            "Train DSFF, CBMC, and DTTI-Net end to end for telecom "
            "fraud detection."
        )
    )

    parser.add_argument(
        "--temporal-path",
        default=str(DEFAULT_TEMPORAL_PATH),
    )
    parser.add_argument(
        "--metadata-path",
        default=str(DEFAULT_METADATA_PATH),
    )
    parser.add_argument(
        "--allow-legacy-data",
        action="store_true",
        help=(
            "Allow metadata without the current raw-VOC/SMS schema. "
            "Use only for explicit legacy ablations."
        ),
    )
    parser.add_argument(
        "--allow-partial-data",
        action="store_true",
        help=(
            "Allow a smoke-test dataset created with row or user limits. "
            "Never use this option for reported experiments."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
    )
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--test-ratio", type=float, default=0.2)
    parser.add_argument("--validation-ratio", type=float, default=0.2)
    parser.add_argument(
        "--protocol",
        choices=("all", "learning-curve", "train80"),
        default="all",
        help=(
            "Run all experiments, only the 10/20/40/60 learning "
            "curve, or only the 80/20 comparison experiment."
        ),
    )
    parser.add_argument(
        "--train-ratios",
        type=float,
        nargs="+",
        default=(0.1, 0.2, 0.4, 0.6),
        help=(
            "Fractions of all users used for the learning-curve "
            "training subsets. The default is 0.1 0.2 0.4 0.6."
        ),
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--split-seed",
        type=int,
        default=42,
        help=(
            "Seed used only for train/validation/test partitioning. "
            "Keep this fixed when repeating experiments with different "
            "model seeds."
        ),
    )
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--target-training-instances",
        type=int,
        default=None,
        help=(
            "Expand only the training loader to this many instances "
            "using coordinate-preserving temporal augmentation."
        ),
    )
    parser.add_argument(
        "--test-time-views",
        type=int,
        default=1,
        help=(
            "Number of views evaluated per original test user. "
            "Predictions are averaged before user-level metrics are "
            "calculated; 1 disables test-time augmentation."
        ),
    )
    parser.add_argument(
        "--augmentation-event-dropout",
        type=float,
        default=0.05,
    )
    parser.add_argument(
        "--augmentation-jitter-std",
        type=float,
        default=0.05,
    )
    parser.add_argument(
        "--augmentation-day-mask-probability",
        type=float,
        default=0.25,
    )
    parser.add_argument(
        "--augmentation-max-masked-days",
        type=int,
        default=3,
    )
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--cbmc-hidden-channels", type=int, default=16)
    parser.add_argument(
        "--cbmc-local-kernel-size",
        type=int,
        nargs=2,
        default=(3, 7),
    )
    parser.add_argument(
        "--cbmc-initial-anomaly-scale",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--intensity-sparsity-weight",
        type=float,
        default=1e-3,
    )
    parser.add_argument(
        "--intensity-smoothness-weight",
        type=float,
        default=1e-3,
    )
    parser.add_argument(
        "--dtti-channels",
        type=int,
        nargs="+",
        default=(32, 32, 64, 64, 128),
    )
    parser.add_argument(
        "--dtti-chronological-dilations",
        type=int,
        nargs="+",
        default=(1, 2, 4, 8, 16),
    )
    parser.add_argument(
        "--dtti-periodic-dilations",
        type=int,
        nargs="+",
        default=(1, 2, 4, 7, 14),
    )
    parser.add_argument(
        "--dtti-router-hidden-channels",
        type=int,
        default=32,
    )
    parser.add_argument(
        "--dtti-pool-size",
        type=int,
        nargs=2,
        default=(3, 8),
    )
    parser.add_argument(
        "--dtti-classifier-hidden-dimension",
        type=int,
        default=256,
    )
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument(
        "--disable-class-weights",
        action="store_true",
        help="Use an unweighted cross-entropy loss.",
    )
    parser.add_argument(
        "--export-test-matrices",
        action="store_true",
        help=(
            "Export learned test matrices for visualization only. "
            "The exported file is not needed to compute metrics."
        ),
    )

    return parser.parse_args()


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def select_device(requested_device):
    if requested_device != "auto":
        return torch.device(requested_device)

    if torch.cuda.is_available():
        return torch.device("cuda")

    if (
        hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()
    ):
        return torch.device("mps")

    return torch.device("cpu")


def resolve_input_path(path_value):
    """Resolve explicit relative paths from either cwd or project root."""
    path = Path(path_value).expanduser()

    if path.is_absolute():
        return path

    cwd_candidate = Path.cwd() / path
    if cwd_candidate.exists():
        return cwd_candidate.resolve()

    return (PROJECT_ROOT / path).resolve()


def create_fixed_partitions(
    labels,
    test_ratio,
    validation_ratio,
    seed,
):
    """
    Create one fixed 60/20/20 partition.

    The 60% partition is the largest learning-curve training pool.
    Combining it with the 20% validation partition produces the 80%
    training partition used by the separate 80/20 experiment. The test
    users are therefore identical in every experiment.
    """
    if test_ratio <= 0 or validation_ratio <= 0:
        raise ValueError("Split ratios must be greater than zero.")

    if test_ratio + validation_ratio >= 1:
        raise ValueError(
            "test_ratio + validation_ratio must be less than one."
        )

    all_indices = np.arange(len(labels), dtype=np.int64)

    development_indices, test_indices = train_test_split(
        all_indices,
        test_size=test_ratio,
        random_state=seed,
        stratify=labels,
    )

    relative_validation_ratio = (
        validation_ratio / (1.0 - test_ratio)
    )

    maximum_train_indices, validation_indices = train_test_split(
        development_indices,
        test_size=relative_validation_ratio,
        random_state=seed,
        stratify=labels[development_indices],
    )

    return (
        np.asarray(development_indices, dtype=np.int64),
        np.asarray(maximum_train_indices, dtype=np.int64),
        np.asarray(validation_indices, dtype=np.int64),
        np.asarray(test_indices, dtype=np.int64),
    )


def allocate_class_counts(target_size, pool_counts):
    """Allocate an exact target size while preserving class ratios."""
    pool_counts = np.asarray(pool_counts, dtype=np.int64)
    expected = target_size * pool_counts / pool_counts.sum()
    allocated = np.floor(expected).astype(np.int64)
    remaining = int(target_size - allocated.sum())

    fractional_order = np.argsort(
        -(expected - allocated),
        kind="stable",
    )

    for class_position in fractional_order:
        if remaining == 0:
            break
        if allocated[class_position] < pool_counts[class_position]:
            allocated[class_position] += 1
            remaining -= 1

    if remaining != 0:
        raise ValueError(
            "Unable to allocate the requested stratified subset size."
        )

    return allocated


def create_nested_training_subsets(
    labels,
    maximum_train_indices,
    train_ratios,
    total_user_count,
    seed,
):
    """
    Build deterministic, stratified, nested learning-curve subsets.

    A user included at a lower training ratio remains included at every
    larger ratio. This removes resampling differences from comparisons
    between training-set sizes.
    """
    ratios = sorted(set(float(ratio) for ratio in train_ratios))
    if not ratios:
        raise ValueError("At least one training ratio is required.")

    maximum_ratio = len(maximum_train_indices) / float(total_user_count)
    rounding_tolerance = 1.0 / float(total_user_count) + 1e-12
    for ratio in ratios:
        if ratio <= 0:
            raise ValueError("Training ratios must be greater than zero.")
        if ratio > maximum_ratio + rounding_tolerance:
            raise ValueError(
                f"Training ratio {ratio:.4f} exceeds the available "
                f"learning-curve pool ({maximum_ratio:.4f})."
            )

    class_values = np.unique(labels[maximum_train_indices])
    if len(class_values) < 2:
        raise ValueError(
            "The maximum training pool must contain both classes."
        )

    rng = np.random.RandomState(seed)
    shuffled_by_class = []
    for class_value in class_values:
        class_indices = np.asarray(
            maximum_train_indices[
                labels[maximum_train_indices] == class_value
            ],
            dtype=np.int64,
        ).copy()
        rng.shuffle(class_indices)
        shuffled_by_class.append(class_indices)

    pool_counts = np.asarray(
        [len(indices) for indices in shuffled_by_class],
        dtype=np.int64,
    )
    previous_counts = np.zeros_like(pool_counts)
    subsets = {}

    for ratio in ratios:
        target_size = int(round(total_user_count * ratio))
        target_size = min(target_size, len(maximum_train_indices))

        # The largest requested ratio must use every member of the fixed
        # maximum training pool, even when integer rounding differs by one.
        if abs(ratio - maximum_ratio) <= rounding_tolerance:
            target_size = len(maximum_train_indices)

        class_counts = allocate_class_counts(target_size, pool_counts)

        if np.any(class_counts < previous_counts):
            raise RuntimeError(
                "The requested ratios could not form nested class-wise "
                "subsets. Use increasing, well-separated ratios."
            )

        selected = np.concatenate(
            [
                class_indices[:class_count]
                for class_indices, class_count in zip(
                    shuffled_by_class,
                    class_counts,
                )
            ]
        )

        # DataLoader shuffles training batches, but a deterministic index
        # order also makes the saved split directly reproducible.
        subset_rng = np.random.RandomState(
            seed + int(round(ratio * 10000))
        )
        subset_rng.shuffle(selected)
        subsets[ratio] = selected.astype(np.int64, copy=False)
        previous_counts = class_counts

    return subsets


def create_data_loader(
    temporal_matrices,
    global_features,
    labels,
    indices,
    batch_size,
    shuffle,
    num_workers,
    pin_memory,
    target_size=None,
    augmentation_seed=0,
    event_dropout_rate=0.0,
    value_jitter_std=0.0,
    day_mask_probability=0.0,
    max_masked_days=0,
):
    dataset = TelecomFraudDataset(
        temporal_matrices=temporal_matrices,
        global_features=global_features,
        labels=labels,
        indices=indices,
        target_size=target_size,
        augmentation_seed=augmentation_seed,
        event_dropout_rate=event_dropout_rate,
        value_jitter_std=value_jitter_std,
        day_mask_probability=day_mask_probability,
        max_masked_days=max_masked_days,
    )

    loader_arguments = {
        "dataset": dataset,
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
    }

    if num_workers > 0:
        loader_arguments["persistent_workers"] = True

    return DataLoader(**loader_arguments)


def calculate_class_weights(train_labels, device):
    counts = np.bincount(train_labels, minlength=2).astype(np.float64)

    if np.any(counts == 0):
        raise ValueError(
            "Both classes must occur in the training partition."
        )

    weights = len(train_labels) / (2.0 * counts)
    return torch.tensor(
        weights,
        dtype=torch.float32,
        device=device,
    )


def calculate_metrics(labels, predictions, probabilities):
    metrics = {
        "accuracy": float(accuracy_score(labels, predictions)),
        "precision": float(
            precision_score(
                labels,
                predictions,
                zero_division=0,
            )
        ),
        "recall": float(
            recall_score(
                labels,
                predictions,
                zero_division=0,
            )
        ),
        "f1": float(
            f1_score(
                labels,
                predictions,
                zero_division=0,
            )
        ),
        "auc": float(roc_auc_score(labels, probabilities)),
        "confusion_matrix": confusion_matrix(
            labels,
            predictions,
            labels=[0, 1],
        ).tolist(),
    }
    return metrics


def aggregate_test_views(results):
    """Average repeated-view predictions for each original test user."""
    user_indices = np.asarray(results["user_indices"], dtype=np.int64)
    unique_indices, inverse = np.unique(
        user_indices,
        return_inverse=True,
    )

    def grouped_mean(values):
        values = np.asarray(values)
        output = np.zeros(
            (len(unique_indices),) + values.shape[1:],
            dtype=np.float64,
        )
        counts = np.bincount(inverse, minlength=len(unique_indices))
        for view_index, user_position in enumerate(inverse):
            output[user_position] += values[view_index]
        reshape = (len(unique_indices),) + (1,) * (values.ndim - 1)
        return (output / counts.reshape(reshape)).astype(
            values.dtype,
            copy=False,
        )

    labels = np.empty(len(unique_indices), dtype=np.int64)
    for user_position in range(len(unique_indices)):
        group_labels = results["labels"][inverse == user_position]
        if np.any(group_labels != group_labels[0]):
            raise ValueError("Repeated test views have inconsistent labels.")
        labels[user_position] = int(group_labels[0])

    probabilities = grouped_mean(results["probabilities"])
    predictions = (probabilities >= 0.5).astype(np.int64)
    metrics = calculate_metrics(labels, predictions, probabilities)
    for name in ("loss", "mean_anomaly_intensity", "anomaly_scale"):
        if name in results["metrics"]:
            metrics[name] = results["metrics"][name]

    aggregated = {
        "metrics": metrics,
        "user_indices": unique_indices,
        "labels": labels,
        "predictions": predictions,
        "probabilities": probabilities,
        "feature_weights": grouped_mean(results["feature_weights"]),
        "topology_weights": grouped_mean(results["topology_weights"]),
        "view_counts": np.bincount(
            inverse,
            minlength=len(unique_indices),
        ),
        "raw_view_count": int(len(user_indices)),
    }

    optional_arrays = (
        "behavior_matrices",
        "base_behavior_matrices",
        "anomaly_intensity_maps",
        "last_routing_maps",
    )
    for name in optional_arrays:
        if name in results:
            aggregated[name] = grouped_mean(results[name])

    return aggregated


def calculate_intensity_regularization(anomaly_intensity):
    sparsity = anomaly_intensity.mean()
    smoothness_terms = []

    if anomaly_intensity.shape[2] > 1:
        smoothness_terms.append(
            torch.abs(
                anomaly_intensity[:, :, 1:, :]
                - anomaly_intensity[:, :, :-1, :]
            ).mean()
        )

    if anomaly_intensity.shape[3] > 1:
        smoothness_terms.append(
            torch.abs(
                anomaly_intensity[:, :, :, 1:]
                - anomaly_intensity[:, :, :, :-1]
            ).mean()
        )

    if smoothness_terms:
        smoothness = torch.stack(smoothness_terms).mean()
    else:
        smoothness = anomaly_intensity.new_zeros(())

    return sparsity, smoothness


def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer,
    device,
    intensity_sparsity_weight=0.0,
    intensity_smoothness_weight=0.0,
):
    model.train()

    total_classification_loss = 0.0
    total_objective_loss = 0.0
    total_intensity_sparsity = 0.0
    total_intensity_smoothness = 0.0
    total_anomaly_scale = 0.0
    total_samples = 0
    collected_labels = []
    collected_predictions = []
    collected_probabilities = []

    for temporal, global_features, labels, _ in loader:
        temporal = temporal.to(device, non_blocking=True)
        global_features = global_features.to(
            device,
            non_blocking=True,
        )
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()
        outputs = model(temporal, global_features)
        logits = outputs["logits"]
        classification_loss = criterion(logits, labels)
        intensity_sparsity, intensity_smoothness = (
            calculate_intensity_regularization(
                outputs["anomaly_intensity"]
            )
        )
        objective_loss = (
            classification_loss
            + intensity_sparsity_weight * intensity_sparsity
            + intensity_smoothness_weight * intensity_smoothness
        )
        objective_loss.backward()
        optimizer.step()

        probabilities = torch.softmax(
            logits.detach(),
            dim=1,
        )[:, 1]
        predictions = logits.detach().argmax(dim=1)
        batch_size = labels.size(0)
        total_classification_loss += (
            classification_loss.item() * batch_size
        )
        total_objective_loss += objective_loss.item() * batch_size
        total_intensity_sparsity += (
            intensity_sparsity.item() * batch_size
        )
        total_intensity_smoothness += (
            intensity_smoothness.item() * batch_size
        )
        total_anomaly_scale += (
            outputs["anomaly_scale"].detach().item() * batch_size
        )
        total_samples += batch_size
        collected_labels.append(labels.detach().cpu().numpy())
        collected_predictions.append(
            predictions.cpu().numpy()
        )
        collected_probabilities.append(
            probabilities.cpu().numpy()
        )

    epoch_labels = np.concatenate(collected_labels)
    epoch_predictions = np.concatenate(collected_predictions)
    epoch_probabilities = np.concatenate(collected_probabilities)

    metrics = calculate_metrics(
        epoch_labels,
        epoch_predictions,
        epoch_probabilities,
    )
    metrics["loss"] = total_classification_loss / total_samples
    metrics["objective_loss"] = total_objective_loss / total_samples
    metrics["intensity_sparsity"] = (
        total_intensity_sparsity / total_samples
    )
    metrics["intensity_smoothness"] = (
        total_intensity_smoothness / total_samples
    )
    metrics["anomaly_scale"] = total_anomaly_scale / total_samples
    return metrics


def evaluate_model(
    model,
    loader,
    criterion,
    device,
    collect_matrices=False,
    aggregate_repeated_users=False,
):
    model.eval()

    total_loss = 0.0
    total_samples = 0
    total_intensity = 0.0
    total_anomaly_scale = 0.0

    collected_indices = []
    collected_labels = []
    collected_predictions = []
    collected_probabilities = []
    collected_feature_weights = []
    collected_matrices = []
    collected_base_matrices = []
    collected_intensity_maps = []
    collected_topology_weights = []
    collected_last_routing_maps = []

    with torch.no_grad():
        for temporal, global_features, labels, user_indices in loader:
            temporal = temporal.to(device, non_blocking=True)
            global_features = global_features.to(
                device,
                non_blocking=True,
            )
            labels_device = labels.to(device, non_blocking=True)

            outputs = model(temporal, global_features)
            logits = outputs["logits"]
            probabilities = torch.softmax(logits, dim=1)[:, 1]
            predictions = logits.argmax(dim=1)
            loss = criterion(logits, labels_device)

            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size
            total_intensity += (
                outputs["anomaly_intensity"].mean().item()
                * batch_size
            )
            total_anomaly_scale += (
                outputs["anomaly_scale"].item() * batch_size
            )

            collected_indices.append(user_indices.numpy())
            collected_labels.append(labels.numpy())
            collected_predictions.append(predictions.cpu().numpy())
            collected_probabilities.append(probabilities.cpu().numpy())
            collected_feature_weights.append(
                outputs["feature_weights"].cpu().numpy()
            )
            collected_topology_weights.append(
                outputs["mean_topology_weights"].cpu().numpy()
            )

            if collect_matrices:
                collected_matrices.append(
                    outputs["behavior_matrix"].cpu().numpy()
                )
                collected_base_matrices.append(
                    outputs["base_behavior_matrix"].cpu().numpy()
                )
                collected_intensity_maps.append(
                    outputs["anomaly_intensity"].cpu().numpy()
                )
                collected_last_routing_maps.append(
                    outputs["last_routing_weights"].cpu().numpy()
                )

    user_indices = np.concatenate(collected_indices)
    labels = np.concatenate(collected_labels)
    predictions = np.concatenate(collected_predictions)
    probabilities = np.concatenate(collected_probabilities)
    feature_weights = np.concatenate(collected_feature_weights)
    topology_weights = np.concatenate(collected_topology_weights)

    metrics = calculate_metrics(
        labels,
        predictions,
        probabilities,
    )
    metrics["loss"] = total_loss / total_samples
    metrics["mean_anomaly_intensity"] = (
        total_intensity / total_samples
    )
    metrics["anomaly_scale"] = total_anomaly_scale / total_samples

    results = {
        "metrics": metrics,
        "user_indices": user_indices,
        "labels": labels,
        "predictions": predictions,
        "probabilities": probabilities,
        "feature_weights": feature_weights,
        "topology_weights": topology_weights,
    }

    if collect_matrices:
        results["behavior_matrices"] = np.concatenate(
            collected_matrices
        )
        results["base_behavior_matrices"] = np.concatenate(
            collected_base_matrices
        )
        results["anomaly_intensity_maps"] = np.concatenate(
            collected_intensity_maps
        )
        results["last_routing_maps"] = np.concatenate(
            collected_last_routing_maps
        )

    if aggregate_repeated_users:
        return aggregate_test_views(results)
    return results


def print_metrics(prefix, metrics):
    print(
        f"{prefix} | "
        f"Loss: {metrics['loss']:.4f} | "
        f"Accuracy: {metrics['accuracy']:.4f} | "
        f"Precision: {metrics['precision']:.4f} | "
        f"Recall: {metrics['recall']:.4f} | "
        f"F1: {metrics['f1']:.4f} | "
        f"AUC: {metrics['auc']:.4f}"
    )


def save_predictions(
    output_path,
    evaluation_results,
    user_ids,
    temporal_feature_names,
):
    user_indices = evaluation_results["user_indices"]

    prediction_table = pd.DataFrame(
        {
            "user_id": user_ids[user_indices],
            "true_label": evaluation_results["labels"],
            "predicted_label": evaluation_results["predictions"],
            "fraud_probability": evaluation_results["probabilities"],
        }
    )
    prediction_table.to_csv(output_path, index=False)

    weight_table = pd.DataFrame(
        evaluation_results["feature_weights"],
        columns=[
            "weight_" + str(name)
            for name in temporal_feature_names
        ],
    )
    weight_table.insert(
        0,
        "user_id",
        user_ids[user_indices],
    )

    weight_path = output_path.with_name(
        "test_feature_weights.csv"
    )
    weight_table.to_csv(weight_path, index=False)


def build_model(
    temporal_matrices,
    global_features,
    statistics,
    args,
    device,
):
    num_temporal_features = temporal_matrices.shape[1]
    num_global_features = global_features.shape[1]

    dsff = DualScaleFeatureFusion(
        num_temporal_features=num_temporal_features,
        num_global_features=num_global_features,
        temporal_center=statistics["temporal_center"],
        temporal_scale=statistics["temporal_scale"],
        global_center=statistics["global_center"],
        global_scale=statistics["global_scale"],
        temperature=args.temperature,
        dropout=args.dropout,
    )
    cbmc = ContinuousBehaviorMatrixConverter(
        num_features=num_temporal_features,
        hidden_channels=args.cbmc_hidden_channels,
        dropout=args.dropout,
        local_kernel_size=tuple(args.cbmc_local_kernel_size),
        initial_anomaly_scale=args.cbmc_initial_anomaly_scale,
    )
    dtti = DualTopologyTemporalInteractionNetwork(
        in_channels=1,
        channels=tuple(args.dtti_channels),
        chronological_dilations=tuple(
            args.dtti_chronological_dilations
        ),
        periodic_dilations=tuple(args.dtti_periodic_dilations),
        router_hidden_channels=args.dtti_router_hidden_channels,
        pooled_size=tuple(args.dtti_pool_size),
        classifier_hidden_dimension=(
            args.dtti_classifier_hidden_dimension
        ),
        dropout=args.dropout,
    )

    return DABMNet(
        dsff=dsff,
        cbmc=cbmc,
        dtti=dtti,
    ).to(device)


def class_distribution(labels, indices):
    values, counts = np.unique(labels[indices], return_counts=True)
    return {
        str(int(value)): int(count)
        for value, count in zip(values, counts)
    }


def find_independent_validation_peaks(training_history):
    """
    Find the best observed value of each validation metric.

    These values are diagnostic only because different metrics may peak
    at different epochs. Model selection still uses one checkpoint: the
    epoch with the highest validation AUC.
    """
    if not training_history or "validation_auc" not in training_history[0]:
        return {}

    metric_objectives = {
        "validation_loss": "min",
        "validation_accuracy": "max",
        "validation_precision": "max",
        "validation_recall": "max",
        "validation_f1": "max",
        "validation_auc": "max",
    }
    peaks = {}

    for metric_name, objective in metric_objectives.items():
        if objective == "min":
            best_row = min(
                training_history,
                key=lambda row: row[metric_name],
            )
        else:
            best_row = max(
                training_history,
                key=lambda row: row[metric_name],
            )

        peaks[metric_name] = {
            "value": float(best_row[metric_name]),
            "epoch": int(best_row["epoch"]),
        }

    return peaks


def best_metrics_summary_row(result_summary):
    """Flatten one best-model record for the aggregate CSV file."""
    best_model = result_summary["best_model_metrics"]
    validation_metrics = best_model[
        "validation_metrics_at_selected_epoch"
    ]
    test_metrics = best_model["test_metrics_of_selected_checkpoint"]

    return {
        "experiment": result_summary["experiment"],
        "protocol": result_summary["protocol"],
        "actual_train_ratio": result_summary["actual_train_ratio"],
        "selection_metric": best_model["selection_metric"],
        "selected_epoch": best_model["selected_epoch"],
        "validation_accuracy": validation_metrics.get("accuracy"),
        "validation_precision": validation_metrics.get("precision"),
        "validation_recall": validation_metrics.get("recall"),
        "validation_f1": validation_metrics.get("f1"),
        "validation_auc": validation_metrics.get("auc"),
        "test_accuracy": test_metrics["accuracy"],
        "test_precision": test_metrics["precision"],
        "test_recall": test_metrics["recall"],
        "test_f1": test_metrics["f1"],
        "test_auc": test_metrics["auc"],
    }


def _checkpoint_payload(
    model,
    optimizer,
    epoch,
    args,
    temporal_feature_names,
    global_feature_names,
    class_weights,
    validation_metrics=None,
):
    return {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "validation_metrics": validation_metrics,
        "arguments": vars(args),
        "temporal_feature_names": temporal_feature_names,
        "global_feature_names": global_feature_names,
        "class_weights": (
            None
            if class_weights is None
            else class_weights.detach().cpu()
        ),
    }


def run_experiment(
    experiment_name,
    protocol_name,
    requested_train_ratio,
    train_indices,
    validation_indices,
    test_indices,
    temporal_matrices,
    global_features,
    labels,
    user_ids,
    temporal_feature_names,
    global_feature_names,
    output_root,
    args,
    device,
):
    """Train and evaluate one fully independent experimental run."""
    set_random_seed(args.seed)

    experiment_dir = output_root / experiment_name
    experiment_dir.mkdir(parents=True, exist_ok=True)
    has_validation = (
        validation_indices is not None
        and len(validation_indices) > 0
    )
    saved_validation_indices = (
        np.asarray(validation_indices, dtype=np.int64)
        if has_validation
        else np.empty(0, dtype=np.int64)
    )

    np.savez_compressed(
        experiment_dir / "split_indices.npz",
        train_indices=np.asarray(train_indices, dtype=np.int64),
        validation_indices=saved_validation_indices,
        test_indices=np.asarray(test_indices, dtype=np.int64),
    )

    print("\n" + "=" * 72)
    print("Experiment:", experiment_name)
    print("Protocol:", protocol_name)
    print(
        "Split sizes:",
        f"train={len(train_indices)}",
        f"validation={len(saved_validation_indices)}",
        f"test={len(test_indices)}",
    )
    print("Fitting DSFF statistics on this training subset only...")

    statistics = fit_dsff_statistics(
        temporal_data=temporal_matrices,
        global_data=global_features,
        train_indices=train_indices,
    )
    np.savez_compressed(
        experiment_dir / "dsff_statistics.npz",
        **statistics,
    )

    pin_memory = device.type == "cuda"
    train_loader = create_data_loader(
        temporal_matrices,
        global_features,
        labels,
        train_indices,
        args.batch_size,
        True,
        args.num_workers,
        pin_memory,
        target_size=(
            max(
                int(args.target_training_instances),
                int(len(train_indices)),
            )
            if (
                args.target_training_instances is not None
                and protocol_name == "train_test_comparison"
            )
            else None
        ),
        augmentation_seed=args.split_seed + args.seed * 1009,
        event_dropout_rate=args.augmentation_event_dropout,
        value_jitter_std=args.augmentation_jitter_std,
        day_mask_probability=(
            args.augmentation_day_mask_probability
        ),
        max_masked_days=args.augmentation_max_masked_days,
    )
    validation_loader = None
    if has_validation:
        validation_loader = create_data_loader(
            temporal_matrices,
            global_features,
            labels,
            validation_indices,
            args.batch_size,
            False,
            args.num_workers,
            pin_memory,
        )
    test_loader = create_data_loader(
        temporal_matrices,
        global_features,
        labels,
        test_indices,
        args.batch_size,
        False,
        args.num_workers,
        pin_memory,
    )
    final_test_loader = test_loader
    if args.test_time_views > 1:
        final_test_loader = create_data_loader(
            temporal_matrices,
            global_features,
            labels,
            test_indices,
            args.batch_size,
            False,
            args.num_workers,
            pin_memory,
            target_size=len(test_indices) * args.test_time_views,
            augmentation_seed=(
                args.split_seed + args.seed * 1009 + 7919
            ),
            event_dropout_rate=args.augmentation_event_dropout,
            value_jitter_std=args.augmentation_jitter_std,
            day_mask_probability=(
                args.augmentation_day_mask_probability
            ),
            max_masked_days=args.augmentation_max_masked_days,
        )

    training_instance_count = len(train_loader.dataset)
    augmented_training_views = (
        training_instance_count - len(train_indices)
    )
    final_test_instance_count = len(final_test_loader.dataset)
    print(
        "Training users / effective instances / augmented views:",
        len(train_indices),
        "/",
        training_instance_count,
        "/",
        augmented_training_views,
    )
    print(
        "Test users / evaluated views:",
        len(test_indices),
        "/",
        final_test_instance_count,
    )

    model = build_model(
        temporal_matrices,
        global_features,
        statistics,
        args,
        device,
    )

    if args.disable_class_weights:
        criterion = nn.CrossEntropyLoss()
        class_weights = None
    else:
        class_weights = calculate_class_weights(
            labels[train_indices],
            device,
        )
        criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scheduler = None
    if has_validation:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=0.5,
            patience=3,
        )

    checkpoint_path = experiment_dir / (
        "best_model.pt" if has_validation else "final_model.pt"
    )
    best_validation_auc = None
    selected_epoch = 0
    training_history = []
    training_start = time.time()

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        train_metrics = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            intensity_sparsity_weight=(
                args.intensity_sparsity_weight
            ),
            intensity_smoothness_weight=(
                args.intensity_smoothness_weight
            ),
        )
        current_lr = optimizer.param_groups[0]["lr"]
        history_row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
            "train_precision": train_metrics["precision"],
            "train_recall": train_metrics["recall"],
            "train_f1": train_metrics["f1"],
            "train_auc": train_metrics["auc"],
            "train_objective_loss": train_metrics[
                "objective_loss"
            ],
            "train_intensity_sparsity": train_metrics[
                "intensity_sparsity"
            ],
            "train_intensity_smoothness": train_metrics[
                "intensity_smoothness"
            ],
            "anomaly_scale": train_metrics["anomaly_scale"],
            "learning_rate": current_lr,
        }

        if has_validation:
            validation_results = evaluate_model(
                model,
                validation_loader,
                criterion,
                device,
            )
            validation_metrics = validation_results["metrics"]
            scheduler.step(validation_metrics["auc"])
            current_lr = optimizer.param_groups[0]["lr"]

            history_row.update(
                {
                    "validation_loss": validation_metrics["loss"],
                    "validation_accuracy": validation_metrics["accuracy"],
                    "validation_precision": validation_metrics["precision"],
                    "validation_recall": validation_metrics["recall"],
                    "validation_f1": validation_metrics["f1"],
                    "validation_auc": validation_metrics["auc"],
                    "learning_rate": current_lr,
                }
            )

            if (
                best_validation_auc is None
                or validation_metrics["auc"] > best_validation_auc
            ):
                best_validation_auc = validation_metrics["auc"]
                selected_epoch = epoch
                torch.save(
                    _checkpoint_payload(
                        model,
                        optimizer,
                        epoch,
                        args,
                        temporal_feature_names,
                        global_feature_names,
                        class_weights,
                        validation_metrics,
                    ),
                    checkpoint_path,
                )
        else:
            selected_epoch = epoch

        epoch_test_results = evaluate_model(
            model,
            test_loader,
            criterion,
            device,
        )
        epoch_test_metrics = epoch_test_results["metrics"]
        history_row.update(
            {
                "test_loss": epoch_test_metrics["loss"],
                "test_accuracy": epoch_test_metrics["accuracy"],
                "test_precision": epoch_test_metrics["precision"],
                "test_recall": epoch_test_metrics["recall"],
                "test_f1": epoch_test_metrics["f1"],
                "test_auc": epoch_test_metrics["auc"],
            }
        )

        print(f"Epoch {epoch:03d} | LR: {current_lr:.2e}")
        print_metrics("  Train", train_metrics)
        print(
            "  CBMC | "
            f"Objective: {train_metrics['objective_loss']:.4f} | "
            f"Sparsity: {train_metrics['intensity_sparsity']:.4f} | "
            f"Smoothness: {train_metrics['intensity_smoothness']:.4f} | "
            f"Scale: {train_metrics['anomaly_scale']:.4f}"
        )
        if has_validation:
            print_metrics("  Validation", validation_metrics)
        print_metrics("  Test (monitor only)", epoch_test_metrics)

        history_row["seconds"] = time.time() - epoch_start
        training_history.append(history_row)

    if not has_validation:
        # The 80/20 protocol uses all non-test users for fitting. It has no
        # validation partition, so the final epoch is selected by definition;
        # the monitored test metrics are never used for checkpoint selection.
        torch.save(
            _checkpoint_payload(
                model,
                optimizer,
                selected_epoch,
                args,
                temporal_feature_names,
                global_feature_names,
                class_weights,
            ),
            checkpoint_path,
        )

    pd.DataFrame(training_history).to_csv(
        experiment_dir / "training_history.csv",
        index=False,
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )
    model.load_state_dict(checkpoint["model_state_dict"])

    # Reload the selected checkpoint and compute its final reported metrics.
    test_results = evaluate_model(
        model,
        final_test_loader,
        criterion,
        device,
        collect_matrices=args.export_test_matrices,
        aggregate_repeated_users=args.test_time_views > 1,
    )

    selected_validation_metrics = (
        checkpoint.get("validation_metrics") or {}
    )
    selection_metric = (
        "validation_auc"
        if has_validation
        else "fixed_final_epoch"
    )
    best_model_metrics = {
        "selection_metric": selection_metric,
        "selected_epoch": int(selected_epoch),
        "validation_metrics_at_selected_epoch": (
            selected_validation_metrics
        ),
        "independent_validation_metric_peaks": (
            find_independent_validation_peaks(training_history)
        ),
        "diagnostic_note": (
            "Independent validation peaks may come from different epochs; "
            "the reported test metrics use only the checkpoint selected "
            "by validation AUC."
            if has_validation
            else "The 80/20 protocol uses a fixed final epoch and does not "
            "select a model using the test set."
        ),
        "test_metrics_of_selected_checkpoint": test_results["metrics"],
    }

    print_metrics("Test", test_results["metrics"])
    if args.test_time_views > 1:
        print(
            "Test-time augmentation:",
            args.test_time_views,
            "views per user; probabilities averaged before metrics.",
        )
    print(
        "Confusion matrix [[TN, FP], [FN, TP]]:",
        test_results["metrics"]["confusion_matrix"],
    )

    save_predictions(
        experiment_dir / "test_predictions.csv",
        test_results,
        user_ids,
        temporal_feature_names,
    )

    mean_feature_weights = test_results["feature_weights"].mean(axis=0)
    print("Mean test feature weights:")
    for name, weight in zip(
        temporal_feature_names,
        mean_feature_weights,
    ):
        print(f"  {name}: {weight:.6f}")
    print(
        "Mean test anomaly intensity:",
        f"{test_results['metrics']['mean_anomaly_intensity']:.6f}",
    )
    print(
        "Learned anomaly scale:",
        f"{test_results['metrics']['anomaly_scale']:.6f}",
    )
    mean_topology_weights = test_results["topology_weights"].mean(axis=0)
    print(
        "Mean DTTI routing weights "
        "[chronological, period-aligned, difference]:",
        np.array2string(mean_topology_weights, precision=6),
    )

    print("\nBest-model summary:")
    print(
        f"  Selection: {selection_metric}, epoch={selected_epoch}"
    )
    if selected_validation_metrics:
        print_metrics(
            "  Validation at selected epoch",
            selected_validation_metrics,
        )
    print_metrics(
        "  Test of selected checkpoint",
        test_results["metrics"],
    )

    if args.export_test_matrices:
        exported_matrices = np.transpose(
            test_results["behavior_matrices"],
            (0, 2, 3, 1),
        )
        exported_base_matrices = np.transpose(
            test_results["base_behavior_matrices"],
            (0, 2, 3, 1),
        )
        exported_intensity_maps = np.transpose(
            test_results["anomaly_intensity_maps"],
            (0, 2, 3, 1),
        )
        np.savez_compressed(
            experiment_dir / "test_behavior_matrices.npz",
            X=exported_matrices,
            base_behavior_matrix=exported_base_matrices,
            anomaly_intensity=exported_intensity_maps,
            dtti_last_routing=test_results["last_routing_maps"],
            dtti_mean_topology_weights=test_results[
                "topology_weights"
            ],
            Y=test_results["labels"],
            user_ids=user_ids[test_results["user_indices"]],
        )

    total_seconds = time.time() - training_start
    total_users = len(labels)
    result_summary = {
        "experiment": experiment_name,
        "protocol": protocol_name,
        "requested_train_ratio": requested_train_ratio,
        "actual_train_ratio": len(train_indices) / float(total_users),
        "actual_validation_ratio": (
            len(saved_validation_indices) / float(total_users)
        ),
        "actual_test_ratio": len(test_indices) / float(total_users),
        "selection_method": (
            "best validation AUC"
            if has_validation
            else "fixed epoch count; test set not used for selection"
        ),
        "selected_epoch": selected_epoch,
        "best_validation_auc": best_validation_auc,
        "training_seconds": total_seconds,
        "device": str(device),
        "split_sizes": {
            "train": len(train_indices),
            "validation": len(saved_validation_indices),
            "test": len(test_indices),
        },
        "augmentation": {
            "target_training_instances": (
                args.target_training_instances
            ),
            "effective_training_instances": training_instance_count,
            "augmented_training_views": augmented_training_views,
            "test_time_views_per_user": args.test_time_views,
            "test_evaluation_views": final_test_instance_count,
            "event_dropout_rate": args.augmentation_event_dropout,
            "value_jitter_std": args.augmentation_jitter_std,
            "day_mask_probability": (
                args.augmentation_day_mask_probability
            ),
            "max_masked_days": args.augmentation_max_masked_days,
            "test_metrics_remain_user_level": True,
        },
        "class_distribution": {
            "train": class_distribution(labels, train_indices),
            "validation": (
                class_distribution(labels, saved_validation_indices)
                if has_validation
                else {}
            ),
            "test": class_distribution(labels, test_indices),
        },
        "temporal_shape": list(temporal_matrices.shape),
        "global_shape": list(global_features.shape),
        "mean_test_feature_weights": {
            str(name): float(weight)
            for name, weight in zip(
                temporal_feature_names,
                mean_feature_weights,
            )
        },
        "mean_test_topology_weights": {
            "chronological": float(mean_topology_weights[0]),
            "period_aligned": float(mean_topology_weights[1]),
            "difference": float(mean_topology_weights[2]),
        },
        "best_model_metrics": best_model_metrics,
        "test_metrics": test_results["metrics"],
    }

    with open(
        experiment_dir / "test_metrics.json",
        "w",
        encoding="utf-8",
    ) as output_file:
        json.dump(
            result_summary,
            output_file,
            indent=2,
            ensure_ascii=True,
        )

    with open(
        experiment_dir / "best_metrics.json",
        "w",
        encoding="utf-8",
    ) as output_file:
        json.dump(
            best_model_metrics,
            output_file,
            indent=2,
            ensure_ascii=True,
        )

    print("Selected checkpoint:", checkpoint_path)
    print("Outputs saved to:", experiment_dir)

    summary_row = {
        "experiment": experiment_name,
        "protocol": protocol_name,
        "requested_train_ratio": requested_train_ratio,
        "actual_train_ratio": result_summary["actual_train_ratio"],
        "train_users": len(train_indices),
        "training_instances": training_instance_count,
        "validation_users": len(saved_validation_indices),
        "test_users": len(test_indices),
        "test_evaluation_views": final_test_instance_count,
        "selected_epoch": selected_epoch,
        "best_validation_auc": best_validation_auc,
        "test_accuracy": test_results["metrics"]["accuracy"],
        "test_precision": test_results["metrics"]["precision"],
        "test_recall": test_results["metrics"]["recall"],
        "test_f1": test_results["metrics"]["f1"],
        "test_auc": test_results["metrics"]["auc"],
        "training_seconds": total_seconds,
    }
    return result_summary, summary_row


def _ratio_directory_name(prefix, ratio):
    percentage = ratio * 100.0
    if abs(percentage - round(percentage)) < 1e-9:
        ratio_text = str(int(round(percentage)))
    else:
        ratio_text = (f"{percentage:.2f}").rstrip("0").rstrip(".")
        ratio_text = ratio_text.replace(".", "p")
    return prefix + ratio_text


def main():
    args = parse_arguments()
    if args.epochs <= 0:
        raise ValueError("epochs must be greater than zero.")
    if args.batch_size <= 0:
        raise ValueError("batch-size must be greater than zero.")
    if args.intensity_sparsity_weight < 0:
        raise ValueError(
            "intensity-sparsity-weight must be nonnegative."
        )
    if args.intensity_smoothness_weight < 0:
        raise ValueError(
            "intensity-smoothness-weight must be nonnegative."
        )
    if (
        args.target_training_instances is not None
        and args.target_training_instances <= 0
    ):
        raise ValueError("target-training-instances must be positive.")
    if args.test_time_views <= 0:
        raise ValueError("test-time-views must be positive.")
    for argument_name in (
        "augmentation_event_dropout",
        "augmentation_day_mask_probability",
    ):
        value = float(getattr(args, argument_name))
        if value < 0 or value > 1:
            raise ValueError(
                argument_name.replace("_", "-")
                + " must be between zero and one."
            )
    if args.augmentation_jitter_std < 0:
        raise ValueError("augmentation-jitter-std must be nonnegative.")
    if args.augmentation_max_masked_days < 0:
        raise ValueError(
            "augmentation-max-masked-days must be nonnegative."
        )
    if not (
        len(args.dtti_channels)
        == len(args.dtti_chronological_dilations)
        == len(args.dtti_periodic_dilations)
    ):
        raise ValueError(
            "DTTI channels and both dilation lists must have equal length."
        )
    if any(
        int(value) <= 0
        for value in (
            list(args.dtti_channels)
            + list(args.dtti_chronological_dilations)
            + list(args.dtti_periodic_dilations)
            + list(args.dtti_pool_size)
        )
    ):
        raise ValueError("All DTTI dimensions must be positive.")

    set_random_seed(args.seed)
    temporal_path = resolve_input_path(args.temporal_path)
    metadata_path = resolve_input_path(args.metadata_path)

    output_root = Path(args.output_dir).expanduser()
    if not output_root.is_absolute():
        output_root = PROJECT_ROOT / output_root
    output_root = output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    device = select_device(args.device)
    print("Device:", device)
    print("Temporal data:", temporal_path)
    print("Metadata:", metadata_path)

    if not temporal_path.exists():
        raise FileNotFoundError(
            "Temporal matrix file not found: " + str(temporal_path)
        )
    if not metadata_path.exists():
        raise FileNotFoundError(
            "Metadata file not found: " + str(metadata_path)
        )

    temporal_matrices = np.load(temporal_path, mmap_mode="r")

    # The legacy metadata stores user_ids as an object array. This is a
    # trusted local file, so pickle loading is enabled for compatibility.
    with np.load(metadata_path, allow_pickle=True) as metadata:
        labels = np.asarray(metadata["labels"], dtype=np.int64)
        global_features = np.asarray(
            metadata["global_features"],
            dtype=np.float32,
        )
        user_ids = np.asarray(metadata["user_ids"])
        temporal_feature_names = np.asarray(
            metadata["temporal_feature_names"]
        )
        global_feature_names = np.asarray(
            metadata["global_feature_names"]
        )
        if "schema_version" in metadata.files:
            schema_version = str(
                np.asarray(metadata["schema_version"]).item()
            )
        else:
            schema_version = None
        if "is_partial_build" in metadata.files:
            is_partial_build = bool(
                np.asarray(metadata["is_partial_build"]).item()
            )
        else:
            is_partial_build = False
        if "global_feature_set" in metadata.files:
            global_feature_set = str(
                np.asarray(metadata["global_feature_set"]).item()
            )
        else:
            global_feature_set = "unknown"

    if (
        schema_version != EXPECTED_SCHEMA_VERSION
        and not args.allow_legacy_data
    ):
        raise ValueError(
            "The temporal data use an old or unknown schema. Run "
            "`python dabm/data_process.py` to rebuild the K=8 raw-call/SMS "
            "tensor with the compact global context, or pass "
            "--allow-legacy-data only for a deliberate "
            "legacy ablation. Found schema: " + repr(schema_version)
        )
    if is_partial_build and not args.allow_partial_data:
        raise ValueError(
            "The metadata describe a partial smoke-test build. Rebuild "
            "without --max-users/--max-voc-rows/--max-sms-rows before "
            "training reported experiments."
        )

    if temporal_matrices.shape[0] != len(labels):
        raise ValueError(
            "Temporal matrices and metadata have different user counts."
        )
    if global_features.shape[0] != len(labels):
        raise ValueError(
            "Global features and labels have different user counts."
        )
    if temporal_matrices.ndim != 4:
        raise ValueError(
            "Temporal matrices must have shape [N, K, H, D]."
        )
    if temporal_matrices.shape[1] != len(temporal_feature_names):
        raise ValueError(
            "The temporal channel count does not match the metadata."
        )
    if global_features.shape[1] != len(global_feature_names):
        raise ValueError(
            "The global feature count does not match the metadata."
        )

    (
        development_indices,
        maximum_train_indices,
        validation_indices,
        test_indices,
    ) = create_fixed_partitions(
        labels=labels,
        test_ratio=args.test_ratio,
        validation_ratio=args.validation_ratio,
        seed=args.split_seed,
    )

    np.savez_compressed(
        output_root / "fixed_partitions.npz",
        development_indices=development_indices,
        maximum_train_indices=maximum_train_indices,
        validation_indices=validation_indices,
        test_indices=test_indices,
        split_seed=np.asarray(args.split_seed, dtype=np.int64),
    )

    print("Temporal shape:", temporal_matrices.shape)
    print("Global shape:", global_features.shape)
    print("Data schema:", schema_version)
    print("Temporal features:", temporal_feature_names.tolist())
    print("Global feature set:", global_feature_set)
    print("Global features:", global_feature_names.tolist())
    print("Model seed:", args.seed)
    print("Fixed split seed:", args.split_seed)
    print("DTTI channels:", list(args.dtti_channels))
    print(
        "DTTI dilations (chronological / period-aligned):",
        list(args.dtti_chronological_dilations),
        "/",
        list(args.dtti_periodic_dilations),
    )
    print(
        "Fixed partitions:",
        f"max-train={len(maximum_train_indices)}",
        f"validation={len(validation_indices)}",
        f"test={len(test_indices)}",
        f"80%-development={len(development_indices)}",
    )

    training_augmentation_suffix = (
        "_aug_" + str(int(args.target_training_instances))
        if args.target_training_instances is not None
        else ""
    )
    test_augmentation_suffix = (
        "_tta_" + str(int(args.test_time_views))
        if args.test_time_views > 1
        else ""
    )

    experiment_specs = []
    if args.protocol in ("all", "learning-curve"):
        training_subsets = create_nested_training_subsets(
            labels=labels,
        maximum_train_indices=maximum_train_indices,
        train_ratios=args.train_ratios,
        total_user_count=len(labels),
        seed=args.split_seed,
    )
        for ratio, train_indices in training_subsets.items():
            experiment_specs.append(
                {
                    "name": (
                        _ratio_directory_name("train_", ratio)
                        + test_augmentation_suffix
                    ),
                    "protocol": "learning_curve",
                    "requested_train_ratio": ratio,
                    "train_indices": train_indices,
                    "validation_indices": validation_indices,
                }
            )

    if args.protocol in ("all", "train80"):
        train80_ratio = len(development_indices) / float(len(labels))
        test_percentage = int(round(args.test_ratio * 100))
        train_percentage = int(round(train80_ratio * 100))
        experiment_specs.append(
            {
                "name": (
                    f"train_{train_percentage}_test_{test_percentage}"
                    + training_augmentation_suffix
                    + test_augmentation_suffix
                ),
                "protocol": "train_test_comparison",
                "requested_train_ratio": 1.0 - args.test_ratio,
                "train_indices": development_indices,
                "validation_indices": None,
            }
        )

    all_summaries = []
    comparison_rows = []
    best_metric_rows = []
    for specification in experiment_specs:
        result_summary, summary_row = run_experiment(
            experiment_name=specification["name"],
            protocol_name=specification["protocol"],
            requested_train_ratio=specification["requested_train_ratio"],
            train_indices=specification["train_indices"],
            validation_indices=specification["validation_indices"],
            test_indices=test_indices,
            temporal_matrices=temporal_matrices,
            global_features=global_features,
            labels=labels,
            user_ids=user_ids,
            temporal_feature_names=temporal_feature_names,
            global_feature_names=global_feature_names,
            output_root=output_root,
            args=args,
            device=device,
        )
        all_summaries.append(result_summary)
        comparison_rows.append(summary_row)
        best_metric_rows.append(
            best_metrics_summary_row(result_summary)
        )

        # Update the aggregate files after every completed run so partial
        # results remain available if a long multi-ratio job is interrupted.
        pd.DataFrame(comparison_rows).to_csv(
            output_root / "comparison_summary.csv",
            index=False,
        )
        pd.DataFrame(best_metric_rows).to_csv(
            output_root / "best_metrics_summary.csv",
            index=False,
        )
        with open(
            output_root / "comparison_summary.json",
            "w",
            encoding="utf-8",
        ) as output_file:
            json.dump(
                all_summaries,
                output_file,
                indent=2,
                ensure_ascii=True,
            )

        if device.type == "cuda":
            torch.cuda.empty_cache()
        elif device.type == "mps" and hasattr(torch, "mps"):
            torch.mps.empty_cache()

    print("\nAll requested experiments completed.")
    print("Comparison table:", output_root / "comparison_summary.csv")


if __name__ == "__main__":
    main()
