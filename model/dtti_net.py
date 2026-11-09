"""Dual-Topology Temporal Interaction Network for hour-date matrices."""

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


def _group_count(num_channels):
    for group_count in (8, 4, 2):
        if num_channels % group_count == 0:
            return group_count
    return 1


def _validate_positive_sequence(values, name):
    values = tuple(int(value) for value in values)
    if not values or any(value <= 0 for value in values):
        raise ValueError(name + " must contain positive integers.")
    return values


class ChronologicalDepthwiseConv(nn.Module):
    """Convolve along the true time axis, including midnight transitions."""

    def __init__(self, channels, dilation=1):
        super().__init__()
        if channels <= 0 or dilation <= 0:
            raise ValueError("channels and dilation must be positive.")
        self.channels = int(channels)
        self.convolution = nn.Conv1d(
            self.channels,
            self.channels,
            kernel_size=3,
            padding=int(dilation),
            dilation=int(dilation),
            groups=self.channels,
            bias=False,
        )

    @staticmethod
    def flatten_chronologically(inputs):
        batch_size, channels, hours, days = inputs.shape
        return inputs.permute(0, 1, 3, 2).reshape(
            batch_size,
            channels,
            days * hours,
        )

    @staticmethod
    def restore_hour_date(inputs, hours, days):
        return inputs.reshape(
            inputs.shape[0],
            inputs.shape[1],
            days,
            hours,
        ).permute(0, 1, 3, 2)

    def forward(self, inputs):
        if inputs.ndim != 4 or inputs.shape[1] != self.channels:
            raise ValueError("inputs must have shape [B, C, H, D].")
        hours, days = inputs.shape[2], inputs.shape[3]
        flattened = self.flatten_chronologically(inputs)
        outputs = self.convolution(flattened)
        return self.restore_hour_date(outputs, hours, days)


class PeriodAlignedDepthwiseConv(nn.Module):
    """Convolve across dates while keeping the hour coordinate fixed."""

    def __init__(self, channels, dilation=1):
        super().__init__()
        if channels <= 0 or dilation <= 0:
            raise ValueError("channels and dilation must be positive.")
        self.channels = int(channels)
        self.convolution = nn.Conv2d(
            self.channels,
            self.channels,
            kernel_size=(1, 3),
            padding=(0, int(dilation)),
            dilation=(1, int(dilation)),
            groups=self.channels,
            bias=False,
        )

    def forward(self, inputs):
        if inputs.ndim != 4 or inputs.shape[1] != self.channels:
            raise ValueError("inputs must have shape [B, C, H, D].")
        return self.convolution(inputs)


class DTTIBlock(nn.Module):
    """Fuse chronological, period-aligned, and topology-difference paths."""

    def __init__(
        self,
        in_channels,
        out_channels,
        chronological_dilation,
        periodic_dilation,
        router_hidden_channels=32,
        dropout=0.1,
        use_chronological=True,
        use_periodic=True,
        use_difference=True,
        adaptive_router=True,
    ):
        super().__init__()
        active_paths = (
            bool(use_chronological),
            bool(use_periodic),
            bool(use_difference),
        )
        if not any(active_paths):
            raise ValueError("At least one DTTI path must be enabled.")

        if in_channels == out_channels:
            self.input_projection = nn.Identity()
        else:
            self.input_projection = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=1,
                bias=False,
            )

        self.chronological_depthwise = ChronologicalDepthwiseConv(
            out_channels,
            dilation=chronological_dilation,
        )
        self.chronological_pointwise = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=1,
            bias=False,
        )
        self.periodic_depthwise = PeriodAlignedDepthwiseConv(
            out_channels,
            dilation=periodic_dilation,
        )
        self.periodic_pointwise = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=1,
            bias=False,
        )

        self.router = nn.Sequential(
            nn.Conv2d(
                4 * out_channels,
                router_hidden_channels,
                kernel_size=1,
            ),
            nn.GELU(),
            nn.Dropout2d(dropout),
            nn.Conv2d(router_hidden_channels, 3, kernel_size=1),
        )
        self.output_projection = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=1,
            bias=False,
        )
        self.dropout = nn.Dropout2d(dropout)
        self.normalization = nn.GroupNorm(
            _group_count(out_channels),
            out_channels,
        )
        self.register_buffer(
            "active_path_mask",
            torch.tensor(active_paths, dtype=torch.bool),
        )
        self.adaptive_router = bool(adaptive_router)

    def _routing_weights(self, routing_logits):
        active_mask = self.active_path_mask.view(1, 3, 1, 1)
        if self.adaptive_router:
            minimum = torch.finfo(routing_logits.dtype).min
            routing_logits = routing_logits.masked_fill(
                ~active_mask,
                minimum,
            )
            return torch.softmax(routing_logits, dim=1)

        fixed = active_mask.to(dtype=routing_logits.dtype)
        fixed = fixed / fixed.sum(dim=1, keepdim=True)
        return fixed.expand_as(routing_logits)

    def forward(self, inputs):
        identity = self.input_projection(inputs)
        chronological = self.chronological_pointwise(
            self.chronological_depthwise(identity)
        )
        periodic = self.periodic_pointwise(
            self.periodic_depthwise(identity)
        )
        disagreement = torch.abs(chronological - periodic)

        responses = torch.stack(
            [chronological, periodic, disagreement],
            dim=1,
        )
        routing_logits = self.router(
            torch.cat(
                [identity, chronological, periodic, disagreement],
                dim=1,
            )
        )
        routing_weights = self._routing_weights(routing_logits)
        fused = (
            routing_weights.unsqueeze(2) * responses
        ).sum(dim=1)

        outputs = identity + self.dropout(
            self.output_projection(fused)
        )
        outputs = F.relu(
            self.normalization(outputs),
            inplace=False,
        )
        return {
            "features": outputs,
            "routing_weights": routing_weights,
        }


class DualTopologyTemporalInteractionNetwork(nn.Module):
    """DTTI feature extractor followed by adaptive pooling and classification."""

    def __init__(
        self,
        in_channels=1,
        channels=(32, 32, 64, 64, 128),
        chronological_dilations=(1, 2, 4, 8, 16),
        periodic_dilations=(1, 2, 4, 7, 14),
        router_hidden_channels=32,
        pooled_size=(3, 8),
        classifier_hidden_dimension=256,
        dropout=0.3,
        num_classes=2,
        use_chronological=True,
        use_periodic=True,
        use_difference=True,
        adaptive_router=True,
    ):
        super().__init__()
        channels = _validate_positive_sequence(channels, "channels")
        chronological_dilations = _validate_positive_sequence(
            chronological_dilations,
            "chronological_dilations",
        )
        periodic_dilations = _validate_positive_sequence(
            periodic_dilations,
            "periodic_dilations",
        )
        pooled_size = _validate_positive_sequence(
            pooled_size,
            "pooled_size",
        )
        if len(pooled_size) != 2:
            raise ValueError("pooled_size must contain two integers.")
        if not (
            len(channels)
            == len(chronological_dilations)
            == len(periodic_dilations)
        ):
            raise ValueError(
                "channels and both dilation lists must have equal length."
            )
        if classifier_hidden_dimension <= 0 or num_classes <= 1:
            raise ValueError("Invalid classifier dimensions.")

        self.stem = nn.Sequential(
            nn.Conv2d(
                int(in_channels),
                channels[0],
                kernel_size=1,
                bias=False,
            ),
            nn.GroupNorm(_group_count(channels[0]), channels[0]),
            nn.ReLU(inplace=False),
        )

        blocks: List[nn.Module] = []
        previous_channels = channels[0]
        for output_channels, chronological_dilation, periodic_dilation in zip(
            channels,
            chronological_dilations,
            periodic_dilations,
        ):
            blocks.append(
                DTTIBlock(
                    in_channels=previous_channels,
                    out_channels=output_channels,
                    chronological_dilation=chronological_dilation,
                    periodic_dilation=periodic_dilation,
                    router_hidden_channels=int(router_hidden_channels),
                    dropout=float(dropout),
                    use_chronological=use_chronological,
                    use_periodic=use_periodic,
                    use_difference=use_difference,
                    adaptive_router=adaptive_router,
                )
            )
            previous_channels = output_channels
        self.blocks = nn.ModuleList(blocks)
        self.output_channels = previous_channels
        self.pool = nn.AdaptiveAvgPool2d(tuple(pooled_size))
        classifier_input = (
            self.output_channels * pooled_size[0] * pooled_size[1]
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(classifier_input, classifier_hidden_dimension),
            nn.ReLU(inplace=False),
            nn.Dropout(dropout),
            nn.Linear(classifier_hidden_dimension, num_classes),
        )

    def forward(self, behavior_matrix, return_components=False):
        if behavior_matrix.ndim != 4 or behavior_matrix.shape[1] != 1:
            raise ValueError(
                "behavior_matrix must have shape [B, 1, H, D]."
            )
        behavior_matrix = behavior_matrix.float()
        behavior_matrix = torch.where(
            torch.isfinite(behavior_matrix),
            behavior_matrix,
            torch.zeros_like(behavior_matrix),
        )

        features = self.stem(behavior_matrix)
        routing_weights: List[torch.Tensor] = []
        for block in self.blocks:
            block_outputs = block(features)
            features = block_outputs["features"]
            routing_weights.append(block_outputs["routing_weights"])

        pooled_features = self.pool(features)
        logits = self.classifier(pooled_features)
        if not return_components:
            return logits

        stacked_routing = torch.stack(routing_weights, dim=1)
        mean_topology_weights = stacked_routing.mean(dim=(1, 3, 4))
        return {
            "logits": logits,
            "encoded_features": features,
            "last_routing_weights": routing_weights[-1],
            "mean_topology_weights": mean_topology_weights,
        }


if __name__ == "__main__":
    torch.manual_seed(0)
    model = DualTopologyTemporalInteractionNetwork()
    inputs = torch.rand(2, 1, 24, 244)
    with torch.no_grad():
        outputs = model(inputs, return_components=True)
    print("Input shape:", inputs.shape)
    print("Encoded shape:", outputs["encoded_features"].shape)
    print("Logit shape:", outputs["logits"].shape)
    print("Topology weight shape:", outputs["mean_topology_weights"].shape)
