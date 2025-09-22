# AI Capstone Project: Memory-Efficient Deep Learning on Geospatial Data

> Master **Keras & PyTorch** by building **CNN → Vision Transformer** hybrids that classify **agricultural vs. non-agricultural** land from **gigapixel satellite images**—without blowing your RAM.



## Objectives

1. Stream **>100 GB satellite tiles** with **generator-based** & **TF.data / PyTorch DataLoader** pipelines.  
2. Apply **on-the-fly augmentation** (flip, rotation, channel-drop, CutMix) **without copying data**.  
3. Cache, prefetch, shard and mix-precision your way to **GPU saturation ≥ 95 %**.  
4. Train **CNNs** (ResNet-50, EfficientNet-B3) and evaluate with **accuracy, precision, recall, F1, ROC-AUC**.  
5. Extract **intermediate feature maps** from CNNs and feed them into a **Vision Transformer**.  
6. Package everything inside a **reproducible Docker container** ready for **AWS / GCP batch inference**.

---

## Memory-Efficient Tricks
| Trick | Keras API | PyTorch API | Effect |
|-------|-----------|-------------|--------|
| **Generator** | `tf.keras.utils.Sequence` | `IterableDataset` | RAM ↓ 5× |
| **Cache** | `.cache(filename)` | `persistent_workers=True` | epoch time ↓ 30 % |
| **Prefetch** | `tf.data.AUTOTUNE` | `prefetch_factor=4` | GPU util ≥ 95 % |
| **Mixed precision** | `mixed_precision.set_global_policy('mixed_float16')` | `torch.cuda.amp.autocast` | VRAM ↓ 50 % |
| **On-the-fly aug** | `tf.image` | `albumentations` | 0 extra disk |

---

## Model Zoo (Validated on 50 k test tiles)
| Model | Framework | Params | Val Acc | AUC | 
|-------|-----------|--------|---------|-----|
| ResNet-50 | Keras | 23 M | 92.3 % | 0.967 |
| EfficientNet-B3 | PyTorch | 47 M | 93.1 % | 0.972 |
| CNN→ViT Hybrid | PyTorch | 78 M | **94.7 %** | **0.984** |




---

## 📜 License
MIT © 2025 wusinyee – see [LICENSE](./LICENSE)

