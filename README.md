# Memory-Efficient CNN-ViT for Geospatial Land Classification

This is a capstone project for the IBM AI Engineering Specialization, focusing on scalable deep learning for classifying agricultural versus non-agricultural land in gigapixel-scale satellite imagery (>100 GB datasets). It implements memory-efficient pipelines in Keras and PyTorch, progressing from CNN baselines to hybrid CNN-Vision Transformer (ViT) models that achieve high accuracy without excessive RAM or GPU demands.


## Project Summary

Satellite imagery analysis enables precision agriculture and environmental monitoring, but large-scale datasets pose memory challenges. This project addresses them by developing streaming data loaders, on-the-fly augmentations, and hybrid architectures. Using ESA-derived satellite tiles, models classify land types with >94% validation accuracy and 0.984 AUC, optimizing for real-world deployment on limited hardware. The workflow covers data ingestion, training, evaluation (accuracy, precision, recall, F1, ROC-AUC), and ViT integration, demonstrating a complete end-to-end pipeline.


## Objectives

- Stream and augment >100 GB datasets using generators to reduce RAM usage by 5x.
- Train CNNs (ResNet-50 in Keras, EfficientNet-B3 in PyTorch) as baselines.
- Build CNN-ViT hybrids: Extract local features via CNN, capture global dependencies via ViT.
- Evaluate on 50k test tiles with rigorous metrics and visualizations (ROC curves, confusion matrices).
- Optimize for efficiency: Achieve ≥95% GPU utilization via caching, prefetching, and mixed precision.


## Key Techniques

| Technique          | Keras Implementation          | PyTorch Implementation       | Impact                  |
|--------------------|-------------------------------|------------------------------|-------------------------|
| Generator Loading | `tf.keras.utils.Sequence`    | `IterableDataset`           | RAM ↓5x                |
| Caching           | `.cache(filename)`           | `persistent_workers=True`   | Epoch time ↓30%        |
| Prefetching       | `tf.data.AUTOTUNE`           | `prefetch_factor=4`         | GPU util ≥95%          |
| Mixed Precision   | `mixed_precision.set_global_policy('mixed_float16')` | `torch.cuda.amp.autocast` | VRAM ↓50%              |
| Augmentation      | `tf.image` (flip, rotate, CutMix) | `albumentations`            | No extra disk usage    |



## Model Performance

Validated on binary classification (agri/non-agri):

| Model              | Framework | Parameters | Val Accuracy | AUC   |
|--------------------|-----------|------------|--------------|-------|
| ResNet-50         | Keras    | 23M       | 92.3%       | 0.967 |
| EfficientNet-B3   | PyTorch  | 47M       | 93.1%       | 0.972 |
| CNN-ViT Hybrid    | PyTorch  | 78M       | 94.7%       | 0.984 |

Hybrids excel in recall (0.999), minimizing false negatives for agricultural detection.



## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

Built for the IBM AI Engineering Professional Certificate. Dataset from ESA Sentinel-2. Code inspired by official Keras/PyTorch docs.


---

## 📜 License
MIT © 2025 wusinyee – see [LICENSE](./LICENSE)

