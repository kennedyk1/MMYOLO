from __future__ import annotations

import ast
import contextlib
from pathlib import Path

import torch

from ._bootstrap import ensure_local_ultralytics_repo
from .custom_modules import CBAM, ChannelAttention, ChannelSlice, MultiInputStem, SpatialAttention

ensure_local_ultralytics_repo()

from ultralytics.nn import tasks as yolo_tasks
from ultralytics.utils import LOGGER, colorstr
from ultralytics.utils.ops import make_divisible

_ORIGINAL_PARSE_MODEL = None
_PATCH_APPLIED = False


def _resolve_module(module_name_or_type):
    if not isinstance(module_name_or_type, str):
        return module_name_or_type

    custom_modules = {
        "CBAM": CBAM,
        "ChannelAttention": ChannelAttention,
        "ChannelSlice": ChannelSlice,
        "MultiInputStem": MultiInputStem,
        "SpatialAttention": SpatialAttention,
    }
    if module_name_or_type in custom_modules:
        return custom_modules[module_name_or_type]
    if "nn." in module_name_or_type:
        return getattr(torch.nn, module_name_or_type[3:])
    if "torchvision.ops." in module_name_or_type:
        return getattr(__import__("torchvision").ops, module_name_or_type[16:])
    return getattr(yolo_tasks, module_name_or_type)


def parse_attention_model(d, ch, verbose=True):
    """Parse a model YAML and extend the stock parser with attention-specific blocks."""
    legacy = True
    max_channels = float("inf")
    nc, act, scales, end2end = (d.get(x) for x in ("nc", "activation", "scales", "end2end"))
    reg_max = d.get("reg_max", 16)
    depth, width, kpt_shape = (d.get(x, 1.0) for x in ("depth_multiple", "width_multiple", "kpt_shape"))
    scale = d.get("scale")
    if scales:
        if not scale:
            scale = next(iter(scales.keys()))
            LOGGER.warning(f"no model scale passed. Assuming scale='{scale}'.")
        depth, width, max_channels = scales[scale]

    if act:
        yolo_tasks.Conv.default_act = eval(act)
        if verbose:
            LOGGER.info(f"{colorstr('activation:')} {act}")

    if verbose:
        LOGGER.info(f"\n{'':>3}{'from':>20}{'n':>3}{'params':>10}  {'module':<45}{'arguments':<30}")

    ch = [ch]
    layers, save, c2 = [], [], ch[-1]
    base_modules = frozenset(
        {
            yolo_tasks.Classify,
            yolo_tasks.Conv,
            yolo_tasks.ConvTranspose,
            yolo_tasks.GhostConv,
            yolo_tasks.Bottleneck,
            yolo_tasks.GhostBottleneck,
            yolo_tasks.SPP,
            yolo_tasks.SPPF,
            yolo_tasks.C2fPSA,
            yolo_tasks.C2PSA,
            yolo_tasks.DWConv,
            yolo_tasks.Focus,
            yolo_tasks.BottleneckCSP,
            yolo_tasks.C1,
            yolo_tasks.C2,
            yolo_tasks.C2f,
            yolo_tasks.C3k2,
            yolo_tasks.RepNCSPELAN4,
            yolo_tasks.ELAN1,
            yolo_tasks.ADown,
            yolo_tasks.AConv,
            yolo_tasks.SPPELAN,
            yolo_tasks.C2fAttn,
            yolo_tasks.C3,
            yolo_tasks.C3TR,
            yolo_tasks.C3Ghost,
            torch.nn.ConvTranspose2d,
            yolo_tasks.DWConvTranspose2d,
            yolo_tasks.C3x,
            yolo_tasks.RepC3,
            yolo_tasks.PSA,
            yolo_tasks.SCDown,
            yolo_tasks.C2fCIB,
            yolo_tasks.A2C2f,
            MultiInputStem,
        }
    )
    repeat_modules = frozenset(
        {
            yolo_tasks.BottleneckCSP,
            yolo_tasks.C1,
            yolo_tasks.C2,
            yolo_tasks.C2f,
            yolo_tasks.C3k2,
            yolo_tasks.C2fAttn,
            yolo_tasks.C3,
            yolo_tasks.C3TR,
            yolo_tasks.C3Ghost,
            yolo_tasks.C3x,
            yolo_tasks.RepC3,
            yolo_tasks.C2fPSA,
            yolo_tasks.C2fCIB,
            yolo_tasks.C2PSA,
            yolo_tasks.A2C2f,
        }
    )
    passthrough_modules = frozenset({ChannelAttention, SpatialAttention, CBAM})

    for i, (f, n, m, args) in enumerate(d["backbone"] + d["head"]):
        m = _resolve_module(m)
        for j, a in enumerate(args):
            if isinstance(a, str):
                with contextlib.suppress(ValueError):
                    args[j] = locals()[a] if a in locals() else ast.literal_eval(a)

        n = n_ = max(round(n * depth), 1) if n > 1 else n
        if m in base_modules:
            c1, c2 = ch[f], args[0]
            if c2 != nc:
                c2 = make_divisible(min(c2, max_channels) * width, 8)
            if m is yolo_tasks.C2fAttn:
                args[1] = make_divisible(min(args[1], max_channels // 2) * width, 8)
                args[2] = int(max(round(min(args[2], max_channels // 2 // 32)) * width, 1) if args[2] > 1 else args[2])
            if m is MultiInputStem:
                stem_channels = args[1] if len(args) > 1 else max(args[0] // 2, 32)
                stem_channels = make_divisible(min(stem_channels, max_channels) * width, 8)
                args = [c1, c2, stem_channels, *args[2:]]
            else:
                args = [c1, c2, *args[1:]]
            if m in repeat_modules:
                args.insert(2, n)
                n = 1
            if m is yolo_tasks.C3k2:
                legacy = False
                if scale in "mlx":
                    args[3] = True
            if m is yolo_tasks.A2C2f:
                legacy = False
                if scale in "lx":
                    args.extend((True, 1.2))
            if m is yolo_tasks.C2fCIB:
                legacy = False
        elif m in passthrough_modules:
            c2 = ch[f]
            if m is ChannelAttention:
                args = [c2]
            elif m is SpatialAttention:
                args = [*args]
            else:
                args = [c2, *args]
        elif m is ChannelSlice:
            indices = args[0] if args else []
            if isinstance(indices, int):
                indices = [indices]
            args = [indices]
            c2 = len(indices)
        elif m is yolo_tasks.AIFI:
            args = [ch[f], *args]
        elif m in frozenset({yolo_tasks.HGStem, yolo_tasks.HGBlock}):
            c1, cm, c2 = ch[f], args[0], args[1]
            args = [c1, cm, c2, *args[2:]]
            if m is yolo_tasks.HGBlock:
                args.insert(4, n)
                n = 1
        elif m is yolo_tasks.ResNetLayer:
            c2 = args[1] if args[3] else args[1] * 4
        elif m is torch.nn.BatchNorm2d:
            args = [ch[f]]
            c2 = ch[f]
        elif m is yolo_tasks.Concat:
            c2 = sum(ch[x] for x in f)
        elif m in frozenset(
            {
                yolo_tasks.Detect,
                yolo_tasks.WorldDetect,
                yolo_tasks.YOLOEDetect,
                yolo_tasks.Segment,
                yolo_tasks.Segment26,
                yolo_tasks.YOLOESegment,
                yolo_tasks.YOLOESegment26,
                yolo_tasks.Pose,
                yolo_tasks.Pose26,
                yolo_tasks.OBB,
                yolo_tasks.OBB26,
            }
        ):
            args.extend([reg_max, end2end, [ch[x] for x in f]])
            if m in {yolo_tasks.Segment, yolo_tasks.YOLOESegment, yolo_tasks.Segment26, yolo_tasks.YOLOESegment26}:
                args[2] = make_divisible(min(args[2], max_channels) * width, 8)
            if m in {
                yolo_tasks.Detect,
                yolo_tasks.YOLOEDetect,
                yolo_tasks.Segment,
                yolo_tasks.Segment26,
                yolo_tasks.YOLOESegment,
                yolo_tasks.YOLOESegment26,
                yolo_tasks.Pose,
                yolo_tasks.Pose26,
                yolo_tasks.OBB,
                yolo_tasks.OBB26,
            }:
                m.legacy = legacy
            c2 = None
        elif m is yolo_tasks.v10Detect:
            args.append([ch[x] for x in f])
            c2 = None
        elif m is yolo_tasks.ImagePoolingAttn:
            args.insert(1, [ch[x] for x in f])
            c2 = ch[f]
        elif m is yolo_tasks.RTDETRDecoder:
            args.insert(1, [ch[x] for x in f])
            c2 = None
        elif m is yolo_tasks.CBLinear:
            c2 = args[0]
            c1 = ch[f]
            args = [c1, c2, *args[1:]]
        elif m is yolo_tasks.CBFuse:
            c2 = ch[f[-1]]
        elif m in frozenset({yolo_tasks.TorchVision, yolo_tasks.Index}):
            c2 = args[0]
            c1 = ch[f]
            args = [*args[1:]]
        else:
            c2 = ch[f]

        m_ = torch.nn.Sequential(*(m(*args) for _ in range(n))) if n > 1 else m(*args)
        t = str(m)[8:-2].replace("__main__.", "")
        m_.np = sum(x.numel() for x in m_.parameters())
        m_.i, m_.f, m_.type = i, f, t
        if verbose:
            LOGGER.info(f"{i:>3}{f!s:>20}{n_:>3}{m_.np:10.0f}  {t:<45}{args!s:<30}")
        save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)
        layers.append(m_)
        if i == 0:
            ch = []
        ch.append(c2)
    return torch.nn.Sequential(*layers), sorted(save)


def register_attention_modules() -> None:
    """Patch the Ultralytics parser at runtime so custom YAML blocks are supported."""
    global _ORIGINAL_PARSE_MODEL, _PATCH_APPLIED
    if _PATCH_APPLIED:
        return

    _ORIGINAL_PARSE_MODEL = getattr(yolo_tasks, "parse_model")
    for name, module in {
        "CBAM": CBAM,
        "ChannelAttention": ChannelAttention,
        "ChannelSlice": ChannelSlice,
        "MultiInputStem": MultiInputStem,
        "SpatialAttention": SpatialAttention,
    }.items():
        setattr(yolo_tasks, name, module)
    yolo_tasks.parse_model = parse_attention_model
    _PATCH_APPLIED = True


def unregister_attention_modules() -> None:
    """Restore the original parser. Useful for tests, optional for normal runtime."""
    global _ORIGINAL_PARSE_MODEL, _PATCH_APPLIED
    if _PATCH_APPLIED and _ORIGINAL_PARSE_MODEL is not None:
        yolo_tasks.parse_model = _ORIGINAL_PARSE_MODEL
    _PATCH_APPLIED = False


def guess_scale_from_name(path_like: str | Path | None) -> str:
    if not path_like:
        return ""
    return yolo_tasks.guess_model_scale(str(path_like))


__all__ = [
    "guess_scale_from_name",
    "parse_attention_model",
    "register_attention_modules",
    "unregister_attention_modules",
]
