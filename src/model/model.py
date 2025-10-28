("""ResNet-based classifier for mammography images.

Provides a configurable ResNet encoder (pretrained) with support for single-channel
grayscale input, optional freezing of the encoder weights, and a small classification head.

Usage example:
	from src.model.model import ResNetMammo
	model = ResNetMammo(model_name='resnet18', num_classes=2, pretrained=True, in_channels=1)

""")

from typing import Optional

import torch
import torch.nn as nn
from torchvision import models


def _replace_first_conv(model: nn.Module, in_channels: int) -> None:
	"""Replace the first conv layer of a torchvision ResNet to accept different input channels.

	If converting from 3->1 channel, the new weights are initialized by averaging the
	existing weights across the input channel dimension.
	"""
	# Typical attribute name for ResNet conv1
	conv = getattr(model, "conv1", None)
	if conv is None:
		return

	if conv.in_channels == in_channels:
		return

	new_conv = nn.Conv2d(
		in_channels,
		conv.out_channels,
		kernel_size=conv.kernel_size,
		stride=conv.stride,
		padding=conv.padding,
		bias=(conv.bias is not None),
	)

	# Initialize weights: if converting from 3->1, average RGB weights
	with torch.no_grad():
		old_w = conv.weight.data
		if old_w.size(1) == 3 and in_channels == 1:
			new_w = old_w.mean(dim=1, keepdim=True)
			new_conv.weight.data.copy_(new_w)
		else:
			# fallback: replicate or truncate channels
			rep = old_w[:, :in_channels, ...]
			if rep.size(1) != in_channels:
				# If fewer channels in old, repeat first channel
				rep = rep.repeat(1, int((in_channels + rep.size(1) - 1) // rep.size(1)), 1, 1)
				rep = rep[:, :in_channels, ...]
			new_conv.weight.data.copy_(rep)

	setattr(model, "conv1", new_conv)


class ResNetMammo(nn.Module):
	"""ResNet encoder + classification head for mammography.

	Args:
		model_name: one of the ResNet constructors in torchvision.models (e.g. 'resnet18').
		num_classes: number of output classes for the classification head.
		pretrained: whether to load ImageNet pretrained weights for the encoder.
		in_channels: input channels (1 for grayscale mammograms).
		freeze_backbone: if True, encoder parameters will have requires_grad=False.
		dropout: dropout probability before the final linear layer.
	"""

	def __init__(
		self,
		model_name: str = "resnet18",
		num_classes: int = 2,
		pretrained: bool = True,
		in_channels: int = 1,
		freeze_backbone: bool = False,
		dropout: float = 0.5,
	):
		super().__init__()

		if model_name not in models.__dict__:
			raise ValueError(f"Unknown model_name={model_name}; available: resnet18,resnet34,resnet50,...")

		# Instantiate backbone
		weights = "DEFAULT" if pretrained else None
		try:
			backbone = models.__dict__[model_name](weights=weights)
		except TypeError:
			# Fallback for older torchvision versions that use `pretrained`
			backbone = models.__dict__[model_name](pretrained=pretrained)

		# Adapt first conv to requested in_channels
		_replace_first_conv(backbone, in_channels)

		# Save encoder and replace fc with identity so forward returns features
		in_features = backbone.fc.in_features if hasattr(backbone, "fc") else None
		backbone.fc = nn.Identity()
		self.backbone = backbone

		if in_features is None:
			# Try to infer by passing a dummy tensor later; for now, raise an error
			raise RuntimeError("Unable to determine backbone output features (fc.in_features missing)")

		# Optional freezing of encoder
		if freeze_backbone:
			for p in self.backbone.parameters():
				p.requires_grad = False

		# Simple classification head
		self.head = nn.Sequential(
			nn.Dropout(p=dropout) if dropout > 0 else nn.Identity(),
			nn.Linear(in_features, num_classes),
		)

	def forward(self, x: torch.Tensor, return_features: bool = False):
		"""Forward pass.

		Args:
			x: input tensor of shape (B, C, H, W).
			return_features: if True, also return the backbone features before the head.

		Returns:
			logits (and optionally features)
		"""
		features = self.backbone(x)
		logits = self.head(features)
		if return_features:
			return logits, features
		return logits


def get_model(**kwargs) -> ResNetMammo:
	"""Convenience factory returning a ResNetMammo instance.

	Example: get_model(model_name='resnet34', num_classes=4, in_channels=1)
	"""
	return ResNetMammo(**kwargs)

