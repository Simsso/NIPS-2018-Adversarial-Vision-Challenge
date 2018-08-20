import util.cache_inception_transfer_values as cache
import os

if __name__ == '__main__':
    if not os.path.exists(cache.CACHE_DIR):
        os.makedirs(cache.CACHE_DIR)

    for mode in ['train', 'val']:
        limit = None
        all_images, _ = cache.load_tiny_image_net(mode=mode, limit=limit)

        path = cache.get_cache_path(mode)
        print("Starting calculation for %d %s-images, caching at %s" % (len(all_images), mode, path))
        activations = cache.read_cache_or_generate_activations(cache_path=path, all_images=all_images)
        print("Done. Result shape: ", activations.shape)