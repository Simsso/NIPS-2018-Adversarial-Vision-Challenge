from google.cloud import storage
from google.cloud.storage import Blob
from .. import config

class data:
    def __init__(self):
        self.gcs_client = storage.Client()
        self.bucket = self.gcs_client.get_bucket(config.CONFIG['bucket_id'])

    def getBlob(self, filepath, filename):
        if not filepath.endswith('/'):
            filepath += '/'

        return self.bucket.get_blob(filepath+filename)

    def deleteBlob(self, filepath, filename):
        if not filepath.endswith('/'):
            filepath += '/'

        return self.bucket.delete_blob(filepath+filename, None)

    def uploadBlob(self,filepath, filename):
        if not filepath.endswith('/'):
            filepath += '/'

        blob_path = filepath+filename
        blob = Blob(filename, self.bucket)

        with open(blob_path, 'rb') as file:
            blob.upload_from_file(file)


