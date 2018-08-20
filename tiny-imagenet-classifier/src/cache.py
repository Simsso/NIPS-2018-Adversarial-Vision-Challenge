import util.cache_inception_transfer_values as cache
import os

if __name__ == '__main__':
    if not os.path.exists(cache.CACHE_DIR):
        os.makedirs(cache.CACHE_DIR)

    for mode in ['val', 'train']:
        limit = None

        path = cache.get_cache_path(mode)
        labels_only = cache.activations_are_cached(path)
        all_images, _ = cache.load_tiny_image_net(mode=mode, labels_only=labels_only, limit=limit)

        print("Starting calculation for %d %s-images..." % (len(all_images), mode))
        activations = cache.read_cache_or_generate_activations(cache_path=path, all_images=all_images, batch_size=200)
        print("Done. Result shape: ", activations.shape)

        print(activations[0])
        break