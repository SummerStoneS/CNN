import glob
import tensorflow as tf
from itertools import groupby
from collections import defaultdict

sess = tf.Session()

image_filenames = glob.glob(".\\imagenet-dogs\\images\\n02*\\*.jpg")

# key:狗的品种，value:[狗的jpg]
training_dataset = defaultdict(list)
testing_dataset = defaultdict(list)

# 先生成一个[(狗名，狗jpg)，(狗名，狗jpg)]，从刚才的路径里拆分出来，对该list做迭代
image_filename_with_breed = [(x.split('\\')[3], x) for x in image_filenames]

for dog_breed, breed_images in groupby(image_filename_with_breed, lambda x: x[0]):
    # 按照每个tuple的第一个元素也就是品种分组，dog_breed是group key, breed_images是group value，group value是原来的tuple构成的list，不是只有第二个元素
    for i, breed_image in enumerate(breed_images):
        if i % 5 == 0:
            testing_dataset[dog_breed].append(breed_image[1])          # 注意是tuples,0元素是dog_breed
        else:
            training_dataset[dog_breed].append(breed_image[1])

    breed_training_count = len(training_dataset[dog_breed])
    breed_testing_count = len(testing_dataset[dog_breed])

    assert round(breed_testing_count/(breed_training_count+breed_testing_count), 2) > 0.18, 'not enough testing images'

"""
    转换成TFRecord,让标签和图在一起
"""


def write_records_file(dataset, record_location):
    """
    :param dataset: 刚才生成的字典
    :param record_location: TFRecord存储路径
    :return:
    """
    writer = None
    current_index = 0
    for breed, images_filenames in dataset.items():
        for image_filename in images_filenames:
            if current_index % 100 == 0:
                if writer:
                    writer.close()
                record_filename = "{record_location}-{current_index}.tfrecords".format(record_location=record_location,
                                                                                       current_index=current_index)
                writer = tf.python_io.TFRecordWriter(record_filename)
                current_index += 1

                image_file = tf.read_file(image_filename)
                try:
                    image = tf.image.decode_jpeg(image_file)
                except:
                    print(image_filename)
                    continue
                grayscale_image = tf.image.rgb_to_grayscale(image)
                resized_image = tf.image.resize_images(grayscale_image, size=[250, 151])
                image_bytes = sess.run(tf.cast(resized_image, tf.uint8)).tobytes()

                image_label = breed.encode("utf-8")
                example = tf.train.Example(features=tf.train.Features(feature={
                    "label": tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_label])),
                    "image": tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_bytes]))
                }))
                writer.write(example.SerializeToString())
                writer.close()

write_records_file(testing_dataset, './output/testing-images/testing-image')
write_records_file(training_dataset, './output/training-images/training-image')