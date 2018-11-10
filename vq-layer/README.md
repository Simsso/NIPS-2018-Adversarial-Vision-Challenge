# Vector Quantization Layer

This folder contains a Python module which exports the following functions: 
* [`vector_quantization`](./vq_layer/vq_layer.py#L15): Standard vector quantization layer that allows quantization based on L1-norm, L2-norm or infinity-norm. Trainable, with different loss terms added to the `tf.GraphKeys.LOSSES` collection.
* [`cosine_vector_quantization`](./vq_layer/vq_layer.py#L119): Vector quantization layer performing the lookup based on the largest cosine similarity (argmax of dot product). Not trainable, since it does not add loss terms.
* [`cosine_knn_vector_quantization`](./vq_layer/vq_layer.py#L160): Vector quantization layer performing the lookup based on an `emb_label` majority vote (_k_-nearest-neighbors) of the _k_ embedding vectors with largest cosine similarity.  Not trainable, since it does not add loss terms.

And uses the following named tuples as return types:
* `VQEndpoints`:
    * `layer_out`: Layer output tensor
    * `emb_space`: Embedding space tensor
    * `access_count`: Access counter with integral values indicating how often each embedding vector was used
    * `distance`: Distance of inputs from the embedding space vectors
    * `emb_spacing`: Embedding spacing vector where each entry indicates the distance between embedding vectors
    * `emb_closest_spacing`: Distance of embedding vectors to the closest other embedding vector
    * `replace_embeds`: Op that replaces the least used embedding vectors with the most distant input vectors
    * `emb_space_batch_init`: Embedding space batch init op
* `CosineVQEndpoints`:
    * `layer_out`: Layer output tensor
    * `emb_space`: Embedding space tensor
    * `percentage_identity_mapped`: A float scalar tensor describing the percentage of inputs identity-mapped
    * `similarity_values`: A rank-1 tensor containing all maximum cosine similarity values for a given batch (used to calculate a similarity-histogram)


## Description and Features
These layer functions add a couple of nodes to the TF computational graph which do the following / serve the following purpose:

* Creation of an embedding vector lookup space (referred to as _embedding space_).
* Conversion of each input vector into the closest vector in the embedding space (_quantization_).
* Counting how often a vector in the embedding space has been used for the quantization (_access count_) (only `vector_quantization`).
* Input vector splitting: Lookup of fractions of the input vector,
  e.g. `[1, 2, 3, 4]` could be split into one (default), two, or four components. For two, `[1, 2]` and `[3, 4]` would be quantized separately.
* Addition of loss terms to the collection `tf.GraphKeys.LOSSES` (only `vector_quantization`)
    * Alpha-loss: Penalizes the distance between inputs and the vectors from the embedding space that were chosen during the lookup process.
    * Beta-loss: Penalizes the distance between inputs and all vectors in the embedding space. The idea is to move vectors with low access counts towards the data.
    * Coulomb-loss: Aids greater distances between vectors in the embedding space (_embedding spacing_).
* Embedding spacing monitoring, in form of a vector that contains the distances between all vectors in the embedding space (only `vector_quantization`).
* Variable measures for distance, supported are `tf.norm`'s `ord` values `np.inf`, `1`, and `2`. The functions with a `cosine` prefix use cosine similarity (dot product) instead.
* Gradient skipping enables training of inputs, as if the layer was not present (only `vector_quantization`).
* Dynamic replacement substitutes the least frequently used vectors in the embedding space with the inputs that were furthest away from any vector in the embedding space (only `vector_quantization`).
* The embedding space batch initialization feature sets the embedding space to the input vectors of the current batch. That way the embeddings are guaranteed to have reasonable magnitudes and directions.
* Dimensionality reduction can be used to project both inputs and embedding space to a lower dimension before comparing them. Available modes are:
    * `pca-batch` calculates the PCA based on the embedding space and the whole input batch
    * `pca-emb-space` calculates the PCA only based on the embedding space


## Installation

Install with `pip`

```bash
pip install .
```

and import the function in other projects, e.g. with the following statement:

```python
from vq_layer import vector_quantization as vq
from vq_layer import cosine_knn_vector_quantization as cosine_knn_vq
```
