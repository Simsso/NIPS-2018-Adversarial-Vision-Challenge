import util.cache_inception_transfer_values as cache
import os

LIMIT = None  # for local testing
BATCH_SIZE = 100

if __name__ == '__main__':
    if not os.path.exists(cache.CACHE_DIR):
        os.makedirs(cache.CACHE_DIR)

    for mode in ['train', 'val']:
        path = cache.get_cache_path(mode)
        labels_only = cache.activations_are_cached(path)
        all_images, _ = cache.load_tiny_image_net(mode=mode, labels_only=labels_only, limit=LIMIT)

        print("Starting calculation for %d %s-images in batches of %d..." % (len(all_images), mode, BATCH_SIZE))
        activations = cache.read_cache_or_generate_activations(cache_path=path,
                                                               all_images=all_images,
                                                               mode=mode,
                                                               batch_size=BATCH_SIZE)
        print("Done. Result shape: ", activations.shape, "\n")