config = {
    'bucket_id': 'nips-2018-adversarial-vision-challenge-data',
    'region': 'europe-west4',
    'gs_uri': 'gs://nips-2018-adversarial-vision-challenge-data',
    'wnids_path': 'tiny-imagenet-200/',
    'wnids_filename' : 'wnids.txt',
    'train_folder': 'tiny-imagenet-200/train',
    'class_count': 200,
    'train_count_per_class': 500,
    'image_size': 64,
    'learning_rate': 0.005,
    'batch_size': 20,
    'checkpoint_save_path': 'gs://nips-2018-adversarial-vision-challenge-data/evaluation/checkpoints',

    'model_name': 'reference_cnn'
}
