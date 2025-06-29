# Adaptive Energy-Efficient CNN for Onboard Wildfire Detection on CubeSats

## Overview

The project explores a lightweight convolutional neural network with confidence-driven early exits to enable energy-efficient wildfire detection on resource-constrained 6U CubeSats. The model is based on MobileNetV2 augmented with early-exit classifiers to adaptively terminate inference when high confidence is reached, reducing computation and energy usage.

**Key outcomes include:**
- Trained on 6000 Sentinel-2 multispectral patches (3000 fire / 3000 non-fire).
- Achieved 97% accuracy and 96.9% recall on held-out test data.
- Deployed on NVIDIA Jetson Nano; reduced mean latency by 34% and energy per inference by 44% compared to full model evaluation.
