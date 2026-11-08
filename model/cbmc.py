import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class ContinuousBehaviorMatrixConverter(nn.Module):
    """
    Convert weighted temporal features into an anomaly-aware behavior
    matrix while preserving the hour-date coordinate system.

    Input:
        weighted_features: [B, K, H, D]

    The base branch uses point-wise feature fusion. The intensity branch
    describes level deviation, adjacent temporal change, and local-window
    residuals. A learnable nonnegative scale softly adds the intensity map
    to the base matrix logits. The final CNN input remains [B, 1, H, D].
    """

    def __init__(
        self,
        num_features,
        hidden_channels=16,
        dropout=0.1,
        initial_output_bias=-2.0,
        local_kernel_size=(3, 7),
        initial_intensity_bias=-2.0,
        initial_anomaly_scale=1.0,
    ):
        super().__init__()

        if num_features <= 0:
            raise ValueError("num_features must be greater than zero.")

        if hidden_channels <= 0:
            raise ValueError("hidden_channels must be greater than zero.")

        if len(local_kernel_size) != 2:
            raise ValueError("local_kernel_size must contain two values.")

        local_kernel_size = tuple(
            int(size) for size in local_kernel_size
        )
        if any(
            size <= 0 or size % 2 == 0
            for size in local_kernel_size
        ):
            raise ValueError(
                "local_kernel_size values must be positive odd integers."
            )

        if initial_anomaly_scale <= 0:
            raise ValueError(
                "initial_anomaly_scale must be greater than zero."
            )

        self.num_features = int(num_features)
        self.local_kernel_size = local_kernel_size

        self.hidden_projection = nn.Conv2d(
            in_channels=self.num_features,
            out_channels=hidden_channels,
            kernel_size=1,
            bias=True,
        )
        self.activation = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout2d(dropout)
        self.output_projection = nn.Conv2d(
            in_channels=hidden_channels,
            out_channels=1,
            kernel_size=1,
            bias=True,
        )

        # Three K-channel descriptors are fused at each temporal position:
        # level magnitude, adjacent change, and local-window residual.
        self.intensity_hidden_projection = nn.Conv2d(
            in_channels=3 * self.num_features,
            out_channels=hidden_channels,
            kernel_size=1,
            bias=True,
        )
        self.intensity_output_projection = nn.Conv2d(
            in_channels=hidden_channels,
            out_channels=1,
            kernel_size=1,
            bias=True,
        )

        initial_raw_scale = math.log(
            math.expm1(float(initial_anomaly_scale))
        )
        self.raw_anomaly_scale = nn.Parameter(
            torch.tensor(initial_raw_scale, dtype=torch.float32)
        )

        nn.init.kaiming_normal_(
            self.hidden_projection.weight,
            mode="fan_out",
            nonlinearity="relu",
        )
        nn.init.zeros_(self.hidden_projection.bias)
        nn.init.xavier_uniform_(self.output_projection.weight)
        nn.init.kaiming_normal_(
            self.intensity_hidden_projection.weight,
            mode="fan_out",
            nonlinearity="relu",
        )
        nn.init.zeros_(self.intensity_hidden_projection.bias)
        nn.init.xavier_uniform_(
            self.intensity_output_projection.weight
        )

        # Sigmoid(-2) is approximately 0.12. This makes weak or empty cells
        # start near zero while allowing training to adjust the scale.
        nn.init.constant_(
            self.output_projection.bias,
            float(initial_output_bias),
        )

        nn.init.constant_(
            self.intensity_output_projection.bias,
            float(initial_intensity_bias),
        )

    def _build_intensity_descriptors(self, weighted_features):
        level_magnitude = torch.log1p(
            torch.abs(weighted_features)
        )

        hour_change = torch.zeros_like(weighted_features)
        hour_change[:, :, 1:, :] = torch.abs(
            weighted_features[:, :, 1:, :]
            - weighted_features[:, :, :-1, :]
        )

        day_change = torch.zeros_like(weighted_features)
        day_change[:, :, :, 1:] = torch.abs(
            weighted_features[:, :, :, 1:]
            - weighted_features[:, :, :, :-1]
        )

        adjacent_change = torch.log1p(
            0.5 * (hour_change + day_change)
        )

        padding = tuple(
            size // 2 for size in self.local_kernel_size
        )
        local_mean = F.avg_pool2d(
            weighted_features,
            kernel_size=self.local_kernel_size,
            stride=1,
            padding=padding,
            count_include_pad=False,
        )
        local_residual = torch.log1p(
            torch.abs(weighted_features - local_mean)
        )

        return torch.cat(
            [
                level_magnitude,
                adjacent_change,
                local_residual,
            ],
            dim=1,
        )

    def forward(self, weighted_features, return_components=False):
        if weighted_features.ndim != 4:
            raise ValueError(
                "weighted_features must have shape [B, K, H, D]."
            )

        if weighted_features.shape[1] != self.num_features:
            raise ValueError(
                "The input feature count does not match num_features."
            )

        weighted_features = weighted_features.float()
        weighted_features = torch.where(
            torch.isfinite(weighted_features),
            weighted_features,
            torch.zeros_like(weighted_features),
        )

        base_hidden = self.hidden_projection(weighted_features)
        base_hidden = self.activation(base_hidden)
        base_hidden = self.dropout(base_hidden)

        base_logits = self.output_projection(base_hidden)
        base_behavior_matrix = torch.sigmoid(base_logits)

        intensity_descriptors = self._build_intensity_descriptors(
            weighted_features
        )
        intensity_hidden = self.intensity_hidden_projection(
            intensity_descriptors
        )
        intensity_hidden = self.activation(intensity_hidden)
        intensity_hidden = self.dropout(intensity_hidden)

        intensity_logits = self.intensity_output_projection(
            intensity_hidden
        )
        anomaly_intensity = torch.sigmoid(intensity_logits)

        anomaly_scale = F.softplus(self.raw_anomaly_scale)
        enhanced_logits = (
            base_logits + anomaly_scale * anomaly_intensity
        )
        behavior_matrix = torch.sigmoid(enhanced_logits)

        if return_components:
            return {
                "behavior_matrix": behavior_matrix,
                "base_behavior_matrix": base_behavior_matrix,
                "anomaly_intensity": anomaly_intensity,
                "anomaly_scale": anomaly_scale,
            }

        return behavior_matrix


if __name__ == "__main__":
    torch.manual_seed(0)

    batch_size = 2
    num_features = 8
    num_hours = 24
    num_days = 244

    converter = ContinuousBehaviorMatrixConverter(
        num_features=num_features,
        hidden_channels=16,
    )
    converter.eval()

    weighted_tensor = torch.randn(
        batch_size,
        num_features,
        num_hours,
        num_days,
    )

    with torch.no_grad():
        outputs = converter(
            weighted_tensor,
            return_components=True,
        )

    print("Input shape:", weighted_tensor.shape)
    print("Output shape:", outputs["behavior_matrix"].shape)
    print("Intensity shape:", outputs["anomaly_intensity"].shape)
    print("Anomaly scale:", outputs["anomaly_scale"].item())
    print(
        "Output range:",
        outputs["behavior_matrix"].min().item(),
        outputs["behavior_matrix"].max().item(),
    )
