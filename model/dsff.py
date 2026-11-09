from typing import Dict

import numpy as np
import torch
import torch.nn as nn


def _robust_location_scale(values, eps=1e-6):
    """
    Calculate median and robust scale for a one-dimensional array.
    """
    values = np.asarray(values, dtype=np.float32)
    values = values[np.isfinite(values)]

    if values.size == 0:
        return 0.0, 1.0

    median = float(np.median(values))
    q25, q75 = np.percentile(values, [25.0, 75.0])

    # Convert IQR to an approximation of standard deviation.
    scale = float((q75 - q25) / 1.349)

    # Sparse features may have IQR = 0.
    if not np.isfinite(scale) or scale <= eps:
        scale = float(np.std(values))

    if not np.isfinite(scale) or scale <= eps:
        scale = 1.0

    return median, scale


def fit_dsff_statistics(
    temporal_data,
    global_data,
    train_indices,
    eps=1e-6,
) -> Dict[str, np.ndarray]:
    """
    Fit DSFF normalization statistics using training users only.

    Parameters
    ----------
    temporal_data:
        NumPy array or memory-mapped array with shape [N, K, H, D].
    global_data:
        NumPy array with shape [N, P].
    train_indices:
        Indices of training users.

    Returns
    -------
    Dictionary containing:
        temporal_center: [K, H]
        temporal_scale:  [K, H]
        global_center:    [P]
        global_scale:     [P]
    """
    train_indices = np.asarray(train_indices, dtype=np.int64)

    if temporal_data.ndim != 4:
        raise ValueError(
            "temporal_data must have shape [N, K, H, D]."
        )

    if global_data.ndim != 2:
        raise ValueError(
            "global_data must have shape [N, P]."
        )

    if temporal_data.shape[0] != global_data.shape[0]:
        raise ValueError(
            "temporal_data and global_data must contain "
            "the same users in the same order."
        )

    _, num_temporal_features, num_hours, _ = (
        temporal_data.shape
    )
    num_global_features = global_data.shape[1]

    temporal_center = np.zeros(
        (num_temporal_features, num_hours),
        dtype=np.float32,
    )
    temporal_scale = np.ones(
        (num_temporal_features, num_hours),
        dtype=np.float32,
    )

    # Estimate a separate baseline for each feature and hour.
    for feature_index in range(num_temporal_features):
        for hour_index in range(num_hours):
            values = np.asarray(
                temporal_data[
                    train_indices,
                    feature_index,
                    hour_index,
                    :
                ],
                dtype=np.float32,
            ).reshape(-1)

            values = values[np.isfinite(values)]
            values = np.log1p(
                np.clip(values, 0.0, None)
            )

            center, scale = _robust_location_scale(
                values,
                eps=eps,
            )

            temporal_center[
                feature_index, hour_index
            ] = center
            temporal_scale[
                feature_index, hour_index
            ] = scale

    global_center = np.zeros(
        num_global_features,
        dtype=np.float32,
    )
    global_scale = np.ones(
        num_global_features,
        dtype=np.float32,
    )

    train_global = np.asarray(
        global_data[train_indices],
        dtype=np.float32,
    )

    for feature_index in range(num_global_features):
        center, scale = _robust_location_scale(
            train_global[:, feature_index],
            eps=eps,
        )

        global_center[feature_index] = center
        global_scale[feature_index] = scale

    return {
        "temporal_center": temporal_center,
        "temporal_scale": temporal_scale,
        "global_center": global_center,
        "global_scale": global_scale,
    }


def _get_group_count(num_channels):
    """
    Select a valid group count for GroupNorm.
    """
    for group_count in (8, 4, 2):
        if num_channels % group_count == 0:
            return group_count

    return 1


class SharedTemporalEncoder(nn.Module):
    """
    Encode each hour-date feature map independently.

    The same encoder is shared by all behavioral feature types.
    Feature identity is represented later by a learnable embedding.

    Input:
        [B*K, 1, H, D]

    Output:
        [B*K, 2*C]

    The descriptor contains average-pooled and max-pooled
    representations.
    """

    def __init__(
        self,
        hidden_channels=16,
        output_channels=16,
        dropout=0.1,
    ):
        super().__init__()

        hidden_groups = _get_group_count(
            hidden_channels
        )
        output_groups = _get_group_count(
            output_channels
        )

        self.feature_extractor = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=hidden_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.GroupNorm(
                hidden_groups,
                hidden_channels,
            ),
            nn.ReLU(inplace=True),

            # A dilated convolution provides a wider temporal context
            # without assigning predefined anomaly categories.
            nn.Conv2d(
                in_channels=hidden_channels,
                out_channels=output_channels,
                kernel_size=3,
                padding=2,
                dilation=2,
                bias=False,
            ),
            nn.GroupNorm(
                output_groups,
                output_channels,
            ),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
        )

        self.average_pool = nn.AdaptiveAvgPool2d(
            output_size=1
        )
        self.maximum_pool = nn.AdaptiveMaxPool2d(
            output_size=1
        )

        self.output_dimension = (
            2 * output_channels
        )

    def forward(self, feature_maps):
        encoded_maps = self.feature_extractor(
            feature_maps
        )

        average_descriptor = self.average_pool(
            encoded_maps
        ).flatten(start_dim=1)

        maximum_descriptor = self.maximum_pool(
            encoded_maps
        ).flatten(start_dim=1)

        descriptor = torch.cat(
            [
                average_descriptor,
                maximum_descriptor,
            ],
            dim=1,
        )

        return descriptor


class DualScaleFeatureFusion(nn.Module):
    """
    Context-aware Dual-Scale Feature Fusion.

    Temporal scale:
        The hour-date distribution of each timestamped feature.

    Global scale:
        The overall user behavioral profile.

    Inputs
    ------
    temporal_data:
        Shape [B, K, H, D].
    global_data:
        Shape [B, P].

    Outputs
    -------
    weighted_features:
        Shape [B, K, H, D].
    feature_weights:
        Shape [B, K].
    """

    def __init__(
        self,
        num_temporal_features,
        num_global_features,
        temporal_center,
        temporal_scale,
        global_center,
        global_scale,
        temporal_hidden_channels=16,
        temporal_output_channels=16,
        global_hidden_dimension=64,
        global_context_dimension=32,
        feature_embedding_dimension=8,
        gate_hidden_dimension=32,
        temperature=1.0,
        clip_value=8.0,
        dropout=0.1,
        eps=1e-6,
    ):
        super().__init__()

        if temperature <= 0:
            raise ValueError(
                "temperature must be greater than zero."
            )

        temporal_center = np.asarray(
            temporal_center,
            dtype=np.float32,
        )
        temporal_scale = np.asarray(
            temporal_scale,
            dtype=np.float32,
        )
        global_center = np.asarray(
            global_center,
            dtype=np.float32,
        )
        global_scale = np.asarray(
            global_scale,
            dtype=np.float32,
        )

        if temporal_center.ndim != 2:
            raise ValueError(
                "temporal_center must have shape [K, H]."
            )

        if temporal_scale.shape != temporal_center.shape:
            raise ValueError(
                "temporal_scale must match temporal_center."
            )

        if temporal_center.shape[0] != num_temporal_features:
            raise ValueError(
                "The temporal statistics do not match K."
            )

        if global_center.ndim != 1:
            raise ValueError(
                "global_center must have shape [P]."
            )

        if global_scale.shape != global_center.shape:
            raise ValueError(
                "global_scale must match global_center."
            )

        if global_center.shape[0] != num_global_features:
            raise ValueError(
                "The global statistics do not match P."
            )

        self.num_temporal_features = (
            num_temporal_features
        )
        self.num_global_features = (
            num_global_features
        )
        self.num_hours = temporal_center.shape[1]
        self.temperature = temperature
        self.clip_value = clip_value
        self.eps = eps

        self.register_buffer(
            "temporal_center",
            torch.as_tensor(
                temporal_center,
                dtype=torch.float32,
            ).view(
                1,
                num_temporal_features,
                self.num_hours,
                1,
            ),
        )

        self.register_buffer(
            "temporal_scale",
            torch.as_tensor(
                temporal_scale,
                dtype=torch.float32,
            ).view(
                1,
                num_temporal_features,
                self.num_hours,
                1,
            ),
        )

        self.register_buffer(
            "global_center",
            torch.as_tensor(
                global_center,
                dtype=torch.float32,
            ).view(1, num_global_features),
        )

        self.register_buffer(
            "global_scale",
            torch.as_tensor(
                global_scale,
                dtype=torch.float32,
            ).view(1, num_global_features),
        )

        self.temporal_encoder = SharedTemporalEncoder(
            hidden_channels=temporal_hidden_channels,
            output_channels=temporal_output_channels,
            dropout=dropout,
        )

        self.global_encoder = nn.Sequential(
            nn.Linear(
                num_global_features,
                global_hidden_dimension,
            ),
            nn.LayerNorm(
                global_hidden_dimension
            ),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(
                global_hidden_dimension,
                global_context_dimension,
            ),
            nn.ReLU(inplace=True),
        )

        self.feature_embedding = nn.Parameter(
            torch.empty(
                num_temporal_features,
                feature_embedding_dimension,
            )
        )

        nn.init.normal_(
            self.feature_embedding,
            mean=0.0,
            std=0.02,
        )

        gate_input_dimension = (
            self.temporal_encoder.output_dimension
            + global_context_dimension
            + feature_embedding_dimension
        )

        self.feature_gate = nn.Sequential(
            nn.Linear(
                gate_input_dimension,
                gate_hidden_dimension,
            ),
            nn.LayerNorm(
                gate_hidden_dimension
            ),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(
                gate_hidden_dimension,
                1,
            ),
        )

    def _normalize_temporal_features(
        self,
        temporal_data,
    ):
        temporal_data = temporal_data.float()

        temporal_data = torch.where(
            torch.isfinite(temporal_data),
            temporal_data,
            torch.zeros_like(temporal_data),
        )

        temporal_data = torch.clamp(
            temporal_data,
            min=0.0,
        )

        log_temporal = torch.log1p(
            temporal_data
        )

        normalized_temporal = (
            log_temporal - self.temporal_center
        ) / (
            self.temporal_scale + self.eps
        )

        normalized_temporal = torch.clamp(
            normalized_temporal,
            min=-self.clip_value,
            max=self.clip_value,
        )

        return normalized_temporal

    def _normalize_global_features(
        self,
        global_data,
    ):
        global_data = global_data.float()

        expanded_center = self.global_center.expand_as(
            global_data
        )

        global_data = torch.where(
            torch.isfinite(global_data),
            global_data,
            expanded_center,
        )

        normalized_global = (
            global_data - self.global_center
        ) / (
            self.global_scale + self.eps
        )

        normalized_global = torch.clamp(
            normalized_global,
            min=-self.clip_value,
            max=self.clip_value,
        )

        return normalized_global

    def forward(
        self,
        temporal_data,
        global_data,
    ):
        if temporal_data.ndim != 4:
            raise ValueError(
                "temporal_data must have shape "
                "[B, K, H, D]."
            )

        if global_data.ndim != 2:
            raise ValueError(
                "global_data must have shape [B, P]."
            )

        batch_size, num_features, hours, days = (
            temporal_data.shape
        )

        if num_features != self.num_temporal_features:
            raise ValueError(
                "Unexpected number of temporal features."
            )

        if hours != self.num_hours:
            raise ValueError(
                "The hour dimension does not match "
                "the fitted statistics."
            )

        if global_data.shape[0] != batch_size:
            raise ValueError(
                "Temporal and global batches must match."
            )

        if (
            global_data.shape[1]
            != self.num_global_features
        ):
            raise ValueError(
                "Unexpected number of global features."
            )

        normalized_temporal = (
            self._normalize_temporal_features(
                temporal_data
            )
        )
        # [B, K, H, D]

        normalized_global = (
            self._normalize_global_features(
                global_data
            )
        )
        # [B, P]

        # Apply the shared temporal encoder independently to
        # every feature map.
        encoder_input = normalized_temporal.reshape(
            batch_size * num_features,
            1,
            hours,
            days,
        )

        temporal_descriptors = self.temporal_encoder(
            encoder_input
        )

        temporal_descriptors = (
            temporal_descriptors.reshape(
                batch_size,
                num_features,
                -1,
            )
        )
        # [B, K, Q]

        global_embedding = self.global_encoder(
            normalized_global
        )
        # [B, G]

        expanded_global = global_embedding.unsqueeze(
            dim=1
        ).expand(
            -1,
            num_features,
            -1,
        )

        expanded_feature_embedding = (
            self.feature_embedding.unsqueeze(
                dim=0
            ).expand(
                batch_size,
                -1,
                -1,
            )
        )

        gate_input = torch.cat(
            [
                temporal_descriptors,
                expanded_global,
                expanded_feature_embedding,
            ],
            dim=-1,
        )

        feature_logits = self.feature_gate(
            gate_input
        ).squeeze(dim=-1)
        # [B, K]

        feature_weights = torch.softmax(
            feature_logits / self.temperature,
            dim=1,
        )
        # [B, K]

        # Multiplication by K keeps the original scale when
        # all features receive uniform weights.
        scaled_weights = (
            feature_weights
            * float(num_features)
        ).view(
            batch_size,
            num_features,
            1,
            1,
        )

        weighted_features = (
            normalized_temporal
            * scaled_weights
        )
        # [B, K, H, D]

        return {
            "weighted_features": weighted_features,
            "feature_weights": feature_weights,
            "feature_logits": feature_logits,
            "temporal_descriptors":
                temporal_descriptors,
            "global_embedding": global_embedding,
            "normalized_temporal":
                normalized_temporal,
        }


if __name__ == "__main__":
    # Smoke test using artificial data.
    np.random.seed(0)
    torch.manual_seed(0)

    num_users = 10
    num_temporal_features = 8
    num_global_features = 10
    num_hours = 24
    num_days = 244

    temporal_matrix = np.random.poisson(
        lam=2.0,
        size=(
            num_users,
            num_temporal_features,
            num_hours,
            num_days,
        ),
    ).astype(np.float32)

    global_features = np.random.randn(
        num_users,
        num_global_features,
    ).astype(np.float32)

    train_indices = np.arange(8)

    statistics = fit_dsff_statistics(
        temporal_data=temporal_matrix,
        global_data=global_features,
        train_indices=train_indices,
    )

    model = DualScaleFeatureFusion(
        num_temporal_features=num_temporal_features,
        num_global_features=num_global_features,
        temporal_center=statistics[
            "temporal_center"
        ],
        temporal_scale=statistics[
            "temporal_scale"
        ],
        global_center=statistics[
            "global_center"
        ],
        global_scale=statistics[
            "global_scale"
        ],
    )

    model.eval()

    batch_temporal = torch.from_numpy(
        temporal_matrix[8:10]
    )
    batch_global = torch.from_numpy(
        global_features[8:10]
    )

    with torch.no_grad():
        outputs = model(
            batch_temporal,
            batch_global,
        )

    print(
        "feature_weights:",
        outputs["feature_weights"].shape,
    )
    print(
        outputs["feature_weights"]
    )

    print(
        "weight sums:",
        outputs["feature_weights"].sum(dim=1),
    )

    print(
        "weighted_features:",
        outputs["weighted_features"].shape,
    )

    print(
        "global_embedding:",
        outputs["global_embedding"].shape,
    )
