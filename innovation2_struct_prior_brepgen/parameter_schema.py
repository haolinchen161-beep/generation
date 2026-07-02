# -*- coding: utf-8 -*-
"""Shared class-aware parameter schema for innovation 2."""

PARAMETER_KEYS = [
    "length",
    "width",
    "thickness",
    "height",
    "flange_width",
    "rib_width",
    "rib_height",
    "rib_count",
    "fillet_radius",
]

PARAMETER_INDEX = {k: i for i, k in enumerate(PARAMETER_KEYS)}

PART_TYPE_VOCAB = {
    "l_angle": 0,
    "c_channel": 1,
    "z_beam": 2,
    "hat_stiffener": 3,
    "stiffened_panel": 4,
}

PART_TYPE_BY_ID = {v: k for k, v in PART_TYPE_VOCAB.items()}

CLASS_PARAMETER_KEYS = {
    "l_angle": {"length", "width", "height", "thickness", "fillet_radius"},
    "c_channel": {"length", "width", "height", "thickness", "flange_width", "fillet_radius"},
    "z_beam": {"length", "height", "thickness", "flange_width", "fillet_radius"},
    "hat_stiffener": {"length", "width", "height", "thickness", "flange_width", "fillet_radius"},
    "stiffened_panel": {"length", "width", "thickness", "rib_width", "rib_height", "rib_count", "fillet_radius"},
}

CONTINUOUS_PARAMETER_KEYS = [
    "length",
    "width",
    "thickness",
    "height",
    "flange_width",
    "rib_width",
    "rib_height",
    "fillet_radius",
]

CONTINUOUS_PARAMETER_INDICES = [PARAMETER_INDEX[k] for k in CONTINUOUS_PARAMETER_KEYS]

PARAMETER_BOUNDS = {
    "length": (120.0, 500.0),
    "width": (30.0, 220.0),
    "thickness": (1.8, 3.5),
    "height": (20.0, 120.0),
    "flange_width": (15.0, 80.0),
    "rib_width": (8.0, 50.0),
    "rib_height": (10.0, 100.0),
}

PARAMETER_DEFAULTS = {
    "length": 200.0,
    "width": 100.0,
    "thickness": 2.5,
    "height": 50.0,
    "flange_width": 0.0,
    "rib_width": 0.0,
    "rib_height": 0.0,
    "rib_count": 0,
    "fillet_radius": 3.75,
}

FUSION_PARAMETER_KEYS = ["length", "width", "height", "flange_width", "rib_height"]


def part_type_from_id(class_id):
    return PART_TYPE_BY_ID[int(class_id)]


def parameter_is_meaningful(part_type, key):
    return key in CLASS_PARAMETER_KEYS[part_type]


def class_parameter_mask(part_type):
    keys = CLASS_PARAMETER_KEYS[part_type]
    return [1.0 if k in keys else 0.0 for k in PARAMETER_KEYS]


def class_parameter_mask_tensor(class_ids, device=None, dtype=None):
    import torch

    if torch.is_tensor(class_ids):
        ids = class_ids.detach().cpu().view(-1).tolist()
        device = device or class_ids.device
        dtype = dtype or torch.float32
    elif isinstance(class_ids, (list, tuple)):
        ids = class_ids
        dtype = dtype or torch.float32
    else:
        ids = [class_ids]
        dtype = dtype or torch.float32

    masks = [class_parameter_mask(part_type_from_id(cid)) for cid in ids]
    return torch.tensor(masks, device=device, dtype=dtype)


def physical_tensor_to_norm(param_phys):
    import torch

    scales = torch.tensor(
        [200.0, 200.0, 3.0, 50.0, 50.0, 50.0, 50.0, 1.0, 3.0],
        device=param_phys.device,
        dtype=param_phys.dtype,
    )
    return param_phys / scales


def apply_class_mask_to_parameter_tensors(param_phys, class_ids):
    """Zero irrelevant tensor parameters and apply deterministic class defaults."""
    import torch

    mask = class_parameter_mask_tensor(class_ids, device=param_phys.device, dtype=param_phys.dtype)
    masked_phys = (param_phys * mask).clone()

    if torch.is_tensor(class_ids):
        ids = class_ids.detach().cpu().view(-1).tolist()
    elif isinstance(class_ids, (list, tuple)):
        ids = list(class_ids)
    else:
        ids = [class_ids]
    for row_idx, class_id in enumerate(ids):
        part_type = part_type_from_id(class_id)
        if part_type == "z_beam":
            masked_phys[row_idx, PARAMETER_INDEX["width"]] = masked_phys[row_idx, PARAMETER_INDEX["thickness"]]

    return masked_phys, physical_tensor_to_norm(masked_phys), mask


def apply_class_mask_to_parameter_dict(part_type, params):
    """Return a dict where parameters irrelevant to the class are neutralized."""
    masked = {k: params.get(k, PARAMETER_DEFAULTS[k]) for k in PARAMETER_KEYS}
    for key in PARAMETER_KEYS:
        if key not in CLASS_PARAMETER_KEYS[part_type]:
            masked[key] = 0 if key == "rib_count" else 0.0

    if part_type == "z_beam":
        masked["width"] = masked["thickness"]

    return masked


def meaningful_fusion_keys(part_type):
    meaningful = CLASS_PARAMETER_KEYS[part_type]
    return [k for k in FUSION_PARAMETER_KEYS if k in meaningful]


def blank_repair_flags():
    return {k: 0 for k in PARAMETER_KEYS}
