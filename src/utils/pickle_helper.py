import pickle
from typing import Any


# Sources to compare
# https://stackoverflow.com/questions/30329726/fastest-save-and-load-options-for-a-numpy-array
class PickleHelper:
    triplet_comics_seq_firebase_face_body_batch = 'triplet_comics_seq_firebase_face_body_batch.obj'
    comics_seq_firebase_face_body_batch = 'comics_seq_firebase_face_body_batch.obj'

    @staticmethod
    def save_object(filename: str, object: Any):
        filehandler = open(filename, 'wb')
        pickle.dump(object, filehandler)
        filehandler.close()

    @staticmethod
    def load_object(filename: str):
        filehandler = open(filename, 'rb')
        object = pickle.load(filehandler)
        filehandler.close()
        return object
