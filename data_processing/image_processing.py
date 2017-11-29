import os

import numpy as np
from PIL import Image
from PIL import ImageOps

current_directory = os.path.dirname(os.path.abspath(__file__))
omniglot_directory = 'omniglot'
files_path = os.path.join(current_directory, omniglot_directory)

data = []
image_size = (20, 20)

for dirname, subdirname, filelist in os.walk(files_path):
    if filelist:
        data.append([Image.open(dirname + '/' + filename).copy() for filename in filelist])

train_data = data[:800]
test_data = data[-164:]

print('data size', len(data))
print('train_data size', len(train_data))
print('test_data size', len(test_data))

n_classes = 5
batch_size = 16
seq_length = 50

classes = []

data = train_data

classes = [np.random.choice(range(len(data)), replace=False, size=n_classes) for _ in range(batch_size)]
print('classes[0]', classes[0])
seq = np.random.randint(0, n_classes, [batch_size, seq_length])
print('seq[0]', seq[0])

rand_rotate_map = np.random.randint(0, 4, [batch_size, n_classes])

print('rand_rotate_map[0]', rand_rotate_map[0])



def augment(image, batch, c, only_resize=False):
    if only_resize:
        image = ImageOps.invert(image.convert('L')).resize(image_size)
    else:
        rand_rotate = rand_rotate_map[batch, c] * 90                       # rotate by 0, pi/2, pi, 3pi/2
        image = ImageOps.invert(image.convert('L')) \
            .rotate(rand_rotate + np.random.rand() * 22.5 - 11.25, translate=np.random.randint(-10, 11, size=2).tolist()) \
            .resize(image_size)   # rotate between -pi/16 to pi/16, translate bewteen -10 and 10
    np_image = np.reshape(np.array(image, dtype=np.float32),
                        newshape=(image_size[0] * image_size[1]))
    max_value = np.max(np_image)    # normalization is important
    if max_value > 0.:
        np_image = np_image / max_value
    return np_image

seq_pic = [[augment(data[classes[i][j]][np.random.randint(0, len(data[classes[i][j]]))], batch=i, c=j, only_resize=not augment) for j in seq[i, :]] for i in range(batch_size)]
# for i in range(batch_size):
#     for j in seq[i, :]:
        # batch = i, c = j
        # print('class i:', i, 'j:', j, classes[i][j])
        # print(len(data[classes[i][j]]))

print(len(seq_pic[0][0]))




# print(len(classes))


# x_image, x_label, y = data_loader.fetch_batch(iv.n_classes, iv.batch_size, iv.seq_length)

#
#     def fetch_batch(self, n_classes, batch_size, seq_length,
#                     type='train',
#                     sample_strategy='random',
#                     augment=True,
#                     label_type='one_hot'):

#         self.rand_rotate_init(n_classes, batch_size)
#         seq_pic = [[self.augment(data[classes[i][j]][np.random.randint(0, len(data[classes[i][j]]))],
#                                  batch=i, c=j,
#                                  only_resize=not augment)
#                    for j in seq[i, :]]
#                    for i in range(batch_size)]
#
#         if label_type == 'one_hot':
#             seq_encoded = one_hot_encode(seq, n_classes)
#             seq_encoded_shifted = np.concatenate(
#                 [np.zeros(shape=[batch_size, 1, n_classes]), seq_encoded[:, :-1, :]], axis=1
#             )
#         return seq_pic, seq_encoded_shifted, seq_encoded
#
#     def rand_rotate_init(self, n_classes, batch_size):
#         self.rand_rotate_map = np.random.randint(0, 4, [batch_size, n_classes])
#
#     def augment(self, image, batch, c, only_resize=False):
#         if only_resize:
#             image = ImageOps.invert(image.convert('L')).resize(self.image_size)
#         else:
#             rand_rotate = self.rand_rotate_map[batch, c] * 90                       # rotate by 0, pi/2, pi, 3pi/2
#             image = ImageOps.invert(image.convert('L')) \
#                 .rotate(rand_rotate + np.random.rand() * 22.5 - 11.25,
#                         translate=np.random.randint(-10, 11, size=2).tolist()) \
#                 .resize(self.image_size)   # rotate between -pi/16 to pi/16, translate bewteen -10 and 10
#         np_image = np.reshape(np.array(image, dtype=np.float32),
#                           newshape=(self.image_size[0] * self.image_size[1]))
#         max_value = np.max(np_image)    # normalization is important
#         if max_value > 0.:
#             np_image = np_image / max_value
#         return np_image