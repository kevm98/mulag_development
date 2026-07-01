"""Scaler loading and normalization helpers."""
import torch


def normalize_input(x, mean, std):
    return (x - mean) / std


def denormalize_output(y, mean, std):
    return y * std + mean
