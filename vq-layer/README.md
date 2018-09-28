# Vector Quantization Layer

This folder contains a Python module which exports a single function called `vector_quantization`.
It adds a couple of nodes to the TF computational graph which do the following / serve the following purpose:

* Creation of an embedding vector lookup space (referred to as _embedding space_).
* Conversion of each input vector into the closest vector in the embedding space (_quantization_).
* Counting how often a vector in the embedding space has been used for the quantization (_access count_).
* Input vector splitting: Lookup of fractions of the input vector,
  e.g. `[1, 2, 3, 4]` could be split into one (default), two, or four components. For two, `[1, 2]` and `[3, 4]` would be quantized separately.
* Addition of loss terms to the collection `tf.GraphKeys.LOSSES`
    * Alpha-loss: Penalizes the distance between inputs and the vectors from the embedding space that were chosen during the lookup process.
    * Beta-loss: Penalizes the distance between inputs and all vectors in the embedding space. The idea is to move vectors with low access counts towards the data.
    * Coulomb-loss: Aids greater distances between vectors in the embedding space (_embedding spacing_).
* Embedding spacing monitoring, in form of a vector that contains the distances between all vectors in the embedding space.
* Variable measures for distance, supported are `tf.norm`'s `ord` values `np.inf`, `1`, and `2`.
* Gradient skipping enables training of inputs, as if the layer was not present.


## Installation

Install with `pip`

```bash
pip install .
```

and import the function in other projects, e.g. with the following statement:

```python
from vq_layer import vector_quantization
```
