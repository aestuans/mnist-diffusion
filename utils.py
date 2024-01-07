from dataclasses import is_dataclass, fields
import json
from typing import List

import torch
import torch.nn.functional as F


def sample_change(model, starting_label: int, target_label: int, num_samples: int, w: float, seed: int = 0):
    """
    Sample num_samples images from the model, with a label input changing linearly from starting_label to target_label.

    model: model to sample from
    starting_label: label of the starting image
    target_label: label of the target image
    num_samples: number of samples to generate
    w: guidance strength
    seed: random seed

    Returns: Tensor of shape (num_samples, 1, 28, 28) containing the generated images
    """
    torch.manual_seed(seed)

    starting_one_hot = F.one_hot(torch.tensor(
        [starting_label]), num_classes=10).float()
    target_one_hot = F.one_hot(torch.tensor(
        [target_label]), num_classes=10).float()

    alphas = torch.linspace(0, 1, num_samples).unsqueeze(1)

    c_batch = (1 - alphas) * starting_one_hot + alphas * target_one_hot
    c_batch = torch.nn.functional.normalize(c_batch, p=2, dim=1)

    samples = model.sample((1, 28, 28), c_batch, w=w, same_rand=True)

    return samples[-1]


def sample_change_multi(model, labels: List[int], change_frames: int, stationary_frames: int, w: float, seed: int = 0):
    """
    Sample images from the model, with a label input changing linearly from the first element of labels to the last and back to first.
    The returned images include change_frames number of frames between each label, and stationary_frames number of frames at each label.

    model: model to sample from
    labels: list of labels to change between
    change_frames: number of frames to change between each label
    stationary_frames: number of frames to stay at each label
    w: guidance strength
    seed: random seed

    Returns: Tensor of shape (len(labels) * (change_frames + stationary_frames), 1, 28, 28) containing the generated images
    """
    torch.manual_seed(seed)
    all_samples = []

    for i in range(len(labels)):
        current_label = labels[i]
        next_label = labels[(i + 1) % len(labels)]

        change_samples = sample_change(
            model, current_label, next_label, change_frames, w, seed)

        all_samples.extend(change_samples[0, :].repeat(
            stationary_frames - 1, 1, 1, 1))

        all_samples.extend(change_samples)

    return torch.cat(all_samples, dim=0)


def load_config(json_file, config_class):
    def convert_to_dataclass(cls, data):
        if is_dataclass(cls):
            field_types = {f.name: f.type for f in fields(cls)}
            return cls(**{f: convert_to_dataclass(field_types[f], data[f]) for f in data})
        return data

    with open(json_file, 'r') as file:
        data = json.load(file)
    return convert_to_dataclass(config_class, data)
