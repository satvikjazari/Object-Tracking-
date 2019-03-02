import pickle
import os

import cv2
import numpy as np
import tensorflow as tf
from scipy import misc

import align.detect_face
import facenet, time
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# import sys
# sys.path.append('/darkflow/net/build.py')
# import build
from darkflow.net.build import TFNet

# Was 0.3
# gpu_memory_fraction = 0.3
gpu_memory_fraction = 0.5
facenet_model_checkpoint = os.path.dirname(__file__) + "/20180402-114759"
classifier_model = os.path.dirname(__file__) + "/autopickle/auto_final.pkl"
debug = False
option = {
    'model': 'cfg/yolo-widerface.cfg',
    'load': 'bin/yolo-widerface_final.weights',
    'threshold': 0.03,
    'gpu': 0.5
}


# tfnet = TFNet(option)

class Face:
    def __init__(self):
        self.name = None
        self.bounding_box = None
        self.image = None
        self.container_image = None
        self.embedding = None
        self.conf = 0.0


class Recognition:
    def __init__(self):
        self.detect = Detection()
        self.encoder = Encoder()
        self.identifier = Identifier()

    def add_identity(self, image, person_name):
        faces = self.detect.find_faces(image)

        if len(faces) == 1:
            face = faces[0]
            face.name = person_name
            face.embedding = self.encoder.generate_embedding(face)
            return faces


    # here the image is the frame that we are sending
    def identify(self, image):
        faces = self.detect.find_faces(image)

        for i, face in enumerate(faces):
            if debug:
                cv2.imshow("Face: " + str(i), face.image)
            face.embedding = self.encoder.generate_embedding(face)
            # print(face.conf)
            try:
                face.name, face.conf = self.identifier.identify(face)
            except:
                print('')
            # face.name = None, face.conf = None

        return faces


class Identifier:

    def __init__(self):
        with open(classifier_model, 'rb') as infile:
            self.model, self.class_names = pickle.load(infile)
            self.best_class_probabilities = 0

    def identify(self, face):
        ident_time = time.time()
        # CHANGER THE THRESHOLD HERE__________________________________________________
        threshold = 0.47
        threshold_unkowwn = 0.456
        if face.embedding is not None:
            predictions = self.model.predict_proba([face.embedding])
            best_class_indices = np.argmax(predictions, axis=1)
            best_class_probabilities = predictions[np.arange(len(best_class_indices)), best_class_indices]
            if best_class_probabilities > threshold:
                # print(best_class_probabilities)
                # print('Identify Time > thresh ', time.time()-encoder_time)
                return (self.class_names[best_class_indices[0]], float(best_class_probabilities))
            elif best_class_probabilities > threshold_unkowwn:
                # print('Identify Time < thresh ', time.time()-ident_time)
                return ('Unknown', float(best_class_probabilities))


class Encoder:

    def __init__(self):
        self.sess = tf.Session()
        with self.sess.as_default():
            facenet.load_model(facenet_model_checkpoint)

    def generate_embedding(self, face):
        encoder_time = time.time()
        # Get input and output tensors
        images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
        embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
        phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")

        prewhiten_face = facenet.prewhiten(face.image)

        # Run forward pass to calculate embeddings
        feed_dict = {images_placeholder: [prewhiten_face], phase_train_placeholder: False}
        #   print('Encoding Time ', time.time()-encoder_time)
        return self.sess.run(embeddings, feed_dict=feed_dict)[0]


class Detection:
    # face detection parameters
    minsize = 20  # minimum size of face
    threshold = [0.6, 0.7, 0.7]  # three steps's threshold
    factor = 0.709  # scale factor

    def __init__(self, face_crop_size=160, face_crop_margin=32):
        self.pnet, self.rnet, self.onet = self._setup_mtcnn()
        self.face_crop_size = face_crop_size
        self.face_crop_margin = face_crop_margin
        self.tfnet = TFNet(option)

    def _setup_mtcnn(self):
        with tf.Graph().as_default():
            gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_memory_fraction)
            sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
            with sess.as_default():
                return align.detect_face.create_mtcnn(sess, None)

    def find_faces(self, image):
        find_faces_st = time.time()

        faces = []

        # bounding_boxes, _ = align.detect_face.detect_face(image, self.minsize,
        #                                                   self.pnet, self.rnet, self.onet,
        #                                                   self.threshold, self.factor)
        # for bb in bounding_boxes:
        #     face = Face()
        #     face.container_image = image
        #     face.bounding_box = np.zeros(4, dtype=np.int32)

        #     img_size = np.asarray(image.shape)[0:2]
        #     face.bounding_box[0] = np.maximum(bb[0] - self.face_crop_margin / 2, 0)
        #     face.bounding_box[1] = np.maximum(bb[1] - self.face_crop_margin / 2, 0)
        #     face.bounding_box[2] = np.minimum(bb[2] + self.face_crop_margin / 2, img_size[1])
        #     face.bounding_box[3] = np.minimum(bb[3] + self.face_crop_margin / 2, img_size[0])
        #     cropped = image[face.bounding_box[1]:face.bounding_box[3], face.bounding_box[0]:face.bounding_box[2], :]
        #     face.image = misc.imresize(cropped, (self.face_crop_size, self.face_crop_size), interp='bilinear')

        #     faces.append(face)

        # here faces are returned by using tfnet with frame
        # what is the nature of face which is returned here?
        # here in the faces we are getting all the bounding box dimensions and the confidence on processing a frame
        faces = self.tfnet.return_predict(image, self)
        #  print('Find_faces_time_TF ', time.time()-find_faces_st)

        return faces
