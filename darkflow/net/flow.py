import os
import time
import numpy as np
import tensorflow as tf
import pickle
from multiprocessing.pool import ThreadPool
from scipy import misc
from PIL import Image
import cv2


import sys, os
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "C:\\Users\\jazari1\\Desktop\\Sayyam_GUI\\facenet_attendance\\Re-organize")
import face

print('DARKFLOW')




train_stats = (
    'Training statistics: \n'
    '\tLearning rate : {}\n'
    '\tBatch size    : {}\n'
    '\tEpoch number  : {}\n'
    '\tBackup every  : {}'
)
pool = ThreadPool()

def _save_ckpt(self, step, loss_profile):
    file = '{}-{}{}'
    model = self.meta['name']

    profile = file.format(model, step, '.profile')
    profile = os.path.join(self.FLAGS.backup, profile)
    with open(profile, 'wb') as profile_ckpt: 
        pickle.dump(loss_profile, profile_ckpt)

    ckpt = file.format(model, step, '')
    ckpt = os.path.join(self.FLAGS.backup, ckpt)
    self.say('Checkpoint at step {}'.format(step))
    self.saver.save(self.sess, ckpt)


def train(self):
    loss_ph = self.framework.placeholders
    loss_mva = None; profile = list()

    batches = self.framework.shuffle()
    loss_op = self.framework.loss

    for i, (x_batch, datum) in enumerate(batches):
        if not i: self.say(train_stats.format(
            self.FLAGS.lr, self.FLAGS.batch,
            self.FLAGS.epoch, self.FLAGS.save
        ))

        feed_dict = {
            loss_ph[key]: datum[key] 
                for key in loss_ph }
        feed_dict[self.inp] = x_batch
        feed_dict.update(self.feed)

        fetches = [self.train_op, loss_op]

        if self.FLAGS.summary:
            fetches.append(self.summary_op)

        fetched = self.sess.run(fetches, feed_dict)
        loss = fetched[1]

        if loss_mva is None: loss_mva = loss
        loss_mva = .9 * loss_mva + .1 * loss
        step_now = self.FLAGS.load + i + 1

        if self.FLAGS.summary:
            self.writer.add_summary(fetched[2], step_now)

        form = 'step {} - loss {} - moving ave loss {}'
        self.say(form.format(step_now, loss, loss_mva))
        profile += [(loss, loss_mva)]

        ckpt = (i+1) % (self.FLAGS.save // self.FLAGS.batch)
        args = [step_now, profile]
        if not ckpt: _save_ckpt(self, *args)

    if ckpt: _save_ckpt(self, *args)

def return_predict(self, im, a):
    assert isinstance(im, np.ndarray), \
				'Image is not a np.ndarray'
    h, w, _ = im.shape
    im_orig = im
    im = self.framework.resize_input(im)
    this_inp = np.expand_dims(im, 0)
    feed_dict = {self.inp : this_inp}

    out = self.sess.run(self.out, feed_dict)[0]
    boxes = self.framework.findboxes(out)
    #threshold = self.FLAGS.threshold
    ##### threshold to be changed via config needs to be designed #####
    threshold = 0.5
    boxesInfo = list()


    # faces = face.Face()
    # face = face1
    faces = []



    for box in boxes:
        face1 = face.Face()
        tmpBox = self.framework.process_box(box, h, w, threshold)
            
        #     ##CODE CHANGE##
            
            
        if tmpBox is None:
            continue
        #     boxesInfo.append({
        #         "label": tmpBox[4],
        #         "confidence": tmpBox[6],
        # 
        #      CONFIDENCE BLOCK IS HERE ###
        # 
        #         "topleft": {
        #             "x": tmpBox[0],
        #             "y": tmpBox[2]},
        #         "bottomright": {
        #             "x": tmpBox[1],
        #             "y": tmpBox[3]}
        #     })
        # return boxesInfo



        #### YOLO INTEGRATION CODE ###


        face1.container_image = im
        face1.bounding_box = np.zeros(4, dtype=np.int32)
        img_size = np.asarray(im.shape)[0:2]
        face1.bounding_box[0] = tmpBox[0]
        face1.bounding_box[1] = tmpBox[2]
        face1.bounding_box[2] = tmpBox[1]
        face1.bounding_box[3] = tmpBox[3]
        # face1.bounding_box[0] = np.maximum(tmpBox[0] - a.face_crop_margin / 2, 0)
        # face1.bounding_box[1] = np.maximum(tmpBox[1] - a.face_crop_margin / 2, 0)
        # face1.bounding_box[2] = np.minimum(tmpBox[2] + a.face_crop_margin / 2, img_size[1])
        # face1.bounding_box[3] = np.minimum(tmpBox[3] + a.face_crop_margin / 2, img_size[0])
        cropped = im_orig[face1.bounding_box[1]:face1.bounding_box[3], face1.bounding_box[0]:face1.bounding_box[2], :]
        # face1.image = misc.imresize(cropped, (a.face_crop_size, a.face_crop_size), interp='bilinear')
        # face1.image = cropped.resize((a.face_crop_size, a.face_crop_size), Image.ANTIALIAS)
        face1.image = cv2.resize(cropped, (a.face_crop_size, a.face_crop_size), interpolation = cv2.INTER_CUBIC)
        face1.confidence = tmpBox[6]
        # try:
        #     # face1.image = misc.imresize(cropped, (a.face_crop_size, a.face_crop_size), interp='bilinear')
        #     # face1.img = cropped.resize((a.face_crop_size, a.face_crop_size), Image.ANTIALIAS)
        #     # print("in try block")
        #     face1.image = cv2.resize(cropped, (a.face_crop_size, a.face_crop_size), interpolation = cv2.INTER_CUBIC)
        # except:
        #     try:
        #         face1.image = cv2.resize(cropped, (30, 30), interpolation = cv2.INTER_CUBIC)
                
        #     except:
        #         print("in catch block")
        #         continue    
            
        faces.append(face1)
        # print('Printing faces of individual block')
        # print(faces)








    # print("Printing face list")
    # print(faces)
    return faces







import math

def predict(self):
    inp_path = self.FLAGS.imgdir
    all_inps = os.listdir(inp_path)
    all_inps = [i for i in all_inps if self.framework.is_inp(i)]
    if not all_inps:
        msg = 'Failed to find any images in {} .'
        exit('Error: {}'.format(msg.format(inp_path)))

    batch = min(self.FLAGS.batch, len(all_inps))

    # predict in batches
    n_batch = int(math.ceil(len(all_inps) / batch))
    for j in range(n_batch):
        from_idx = j * batch
        to_idx = min(from_idx + batch, len(all_inps))

        # collect images input in the batch
        this_batch = all_inps[from_idx:to_idx]
        inp_feed = pool.map(lambda inp: (
            np.expand_dims(self.framework.preprocess(
                os.path.join(inp_path, inp)), 0)), this_batch)

        # Feed to the net
        feed_dict = {self.inp : np.concatenate(inp_feed, 0)}    
        self.say('Forwarding {} inputs ...'.format(len(inp_feed)))
        start = time.time()
        out = self.sess.run(self.out, feed_dict)
        stop = time.time(); last = stop - start
        self.say('Total time = {}s / {} inps = {} ips'.format(
            last, len(inp_feed), len(inp_feed) / last))

        # Post processing
        self.say('Post processing {} inputs ...'.format(len(inp_feed)))
        start = time.time()
        pool.map(lambda p: (lambda i, prediction:
            self.framework.postprocess(
               prediction, os.path.join(inp_path, this_batch[i])))(*p),
            enumerate(out))
        stop = time.time(); last = stop - start

        # Timing
        self.say('Total time = {}s / {} inps = {} ips'.format(
            last, len(inp_feed), len(inp_feed) / last))
