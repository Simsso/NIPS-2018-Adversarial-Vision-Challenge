from google.cloud import storage
from google.cloud.storage import Blob
from config import config


# TODO Write Unit Tests
class Data:
    def __init__(self):
        self.gcs_client = storage.Client()
        self.bucket = self.gcs_client.get_bucket(config['bucket_id'])

    def get_file(self, filepath, filename):
        if not filepath.endswith('/'):
            filepath += '/'

        return self.bucket.get_blob(filepath + filename)

    def delete_file(self, filepath, filename):
        if not filepath.endswith('/'):
            filepath += '/'

        return self.bucket.delete_blob(filepath + filename, None)

    def upload_file(self, filepath, filename):
        if not filepath.endswith('/'):
            filepath += '/'

        blob_path = filepath + filename
        blob = Blob(filename, self.bucket)

        with open(blob_path, 'rb') as file:
            blob.upload_from_file(file)

    def list_files(self, path):
        iterator = self.bucket.list_blobs(
            versions=True,
            prefix=path,
            delimiter='/'
        )
        return list(iterator)
