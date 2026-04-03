from __future__ import annotations

from dataclasses import dataclass


DEFAULT_CHANNEL_ORDER = "RGBTDI"


@dataclass(frozen=True)
class ChannelSelection:
    dataset_type: str
    names: tuple[str, ...]
    indices: tuple[int, ...]

    @property
    def num_channels(self) -> int:
        return len(self.indices)


def resolve_channel_selection(dataset_type: str, channel_order: str = DEFAULT_CHANNEL_ORDER) -> ChannelSelection:
    dataset_type = dataset_type.upper().strip()
    channel_order = channel_order.upper().strip()

    if not dataset_type:
        raise ValueError("dataset_type não pode ser vazio.")
    if not channel_order:
        raise ValueError("channel_order não pode ser vazio.")
    if len(set(channel_order)) != len(channel_order):
        raise ValueError(f"channel_order contém canais repetidos: {channel_order}")

    valid_channels = set(channel_order)
    names = tuple(dataset_type)

    invalid = [channel for channel in names if channel not in valid_channels]
    if invalid:
        raise ValueError(
            f"dataset_type='{dataset_type}' contém canais inválidos {invalid}. "
            f"Canais válidos: {sorted(valid_channels)}."
        )
    if len(set(names)) != len(names):
        raise ValueError(f"dataset_type contém canais repetidos: {dataset_type}")

    channel_to_index = {channel: index for index, channel in enumerate(channel_order)}
    indices = tuple(channel_to_index[channel] for channel in names)
    return ChannelSelection(dataset_type=dataset_type, names=names, indices=indices)
