# VQ Layer Experiments

This directory contains experiments related to the development of a vector quantization (VQ) layer. The experiments are Jupyter notebooks.

* **Layer definition**. The VQ layer function is defined in the file `001-vq-layer-function.ipynb`.
* **Projection evaluation**. Evaluation of the quantization functionality, i.e. are vectors being mapped to the closest vector in the embedding space (file `002-projection-evaluation.ipynb`).
* **Alpha-training evaluation**. Embedding space vector-update of only those vectors that were closest to at least one of the given inputs (file `003-embedding-space-training.ipynb`).
* **Beta-training evaluation**. Vectors that are not being used are also moving slowly towards the data samples (`004-embedding-space-training-beta.ipynb`).


Another approach, done by a different person (but basically the same thing):
* **VQ layer experiments**. Contains layer definition, a training test run and some visualizations to see what's going on (`basic-vq-layer-test.ipynb`).