# VQ Layer Experiments

This directory contains experiments related to the development of a vector quantization (VQ) layer. The experiments are Jupyter notebooks.

* **Layer definition**. The VQ layer function is defined in the file `vq-layer-function.ipynb`.
* **Projection evaluation**. Evaluation of the quantization functionality, i.e. are vectors being mapped to the closest vector in the embedding space (file `projection-evaluation.ipynb`).
* **Alpha-training evaluation**. Embedding space vector-update of only those vectors that were closest to at least one of the given inputs.