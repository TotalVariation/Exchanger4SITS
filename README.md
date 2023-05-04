# Exchanger4SITS: Rethinking the Encoding of Satellite Image Time Series
The official code repository for paper "Rethinking the Encoding of Satellite Image Time Series".

[![arXiv](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)](https://arxiv.org/abs/2305.02086)
[![GitHub Stars](https://img.shields.io/github/stars/TotalVariation/Exchanger4SITS?style=social)](https://github.com/TotalVariation/Exchanger4SITS)
[![Github Forks](https://img.shields.io/github/forks/TotalVariation/Exchanger4SITS?style=social)](https://github.com/TotalVariation/Exchanger4SITS)

## New SOTA Results on [PASTIS Benchmark](https://github.com/VSainteuf/pastis-benchmark) Dataset
### PASTIS - Semantic Segmentation

| Model Name         | mIoU | #Params (M) | FLOPs |
| ------------------ |----- |------------ | ------|
| U-TAE | 63.1 | 1.09 | 47G |
| TSViT | 65.4 | 2.16 | 558G |
| **Exchanger+Unet** | 66.8 | 8.08 | 300G |
| **Exchanger+Mask2Former** | **67.9** | 24.59 | 329G |

### PASTIS - Panoptic Segmentation
| Model Name         | SQ | RQ | PQ | #Params (M) | FLOPs |
| ------------------ |----|----|----|-------------|-------|
| UConvLSTM+PaPs  | 80.2 | 43.9 | 35.6 | 2.50 | 55G |
| U-TAE+PaPs | 81.5 | 53.2 | 43.8 | 1.26 | 47G |
| **Exchanger+Unet+PaPs** | 80.3 | 58.9 | 47.8 | 9.99 | 301G |
| **Exchanger+Mask2Former** | **84.6** | **61.6** | **52.6** | 24.63 | 332G |

## License

![License: MIT](https://img.shields.io/github/license/TotalVariation/Exchanger4SITS)

## Acknowledgements

The code base is based on the following great work:

- [GPViT](https://github.com/ChenhongyiYang/GPViT)
- [Mask2Former](https://github.com/facebookresearch/Mask2Former)
- [utae-paps](https://github.com/VSainteuf/utae-paps)
- [tpe](https://github.com/jnyborg/tpe)
- [DeepSatModels](https://github.com/michaeltrs/DeepSatModels)
