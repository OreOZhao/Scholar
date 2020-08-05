from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.models import KeyedVectors, Word2Vec
import sys, os
import numpy as np
from tensorboard.plugins import projector
import tensorflow.compat.v1 as tf


# glove_file = 'GloVe/nat_abstract_vectors.txt'
# word2vec_file = 'data/nat_abstract_glove2word2vec.txt'
#
# (count, dimensions) = glove2word2vec(glove_file, word2vec_file)
# glove_model = KeyedVectors.load_word2vec_format(word2vec_file, binary=False)

def visualize(model, output_path):
    meta_file = "test/w2x_metadata.tsv"
    placeholder = np.zeros((len(model.wv.index2word), 50))

    with open(os.path.join(output_path, meta_file), 'wb') as file_metadata:
        for i, word in enumerate(model.wv.index2word):
            placeholder[i] = model[word]
            # temporary solution for https://github.com/tensorflow/tensorflow/issues/9094
            if word == '':
                print("Emply Line, should replecaed by any thing else, or will cause a bug of tensorboard")
                file_metadata.write("{0}".format('<Empty Line>').encode('utf-8') + b'\n')
            else:
                file_metadata.write("{0}".format(word).encode('utf-8') + b'\n')

    # define the model without training
    sess = tf.InteractiveSession()

    embedding = tf.Variable(placeholder, trainable=False, name='w2x_metadata')
    tf.global_variables_initializer().run()
    tf.disable_eager_execution()
    saver = tf.train.Saver()
    writer = tf.summary.FileWriter(output_path, sess.graph)

    # adding into projector
    config = projector.ProjectorConfig()
    embed = config.embeddings.add()
    embed.tensor_name = 'w2x_metadata'
    embed.metadata_path = meta_file

    # Specify the width and height of a single thumbnail.
    projector.visualize_embeddings(writer, config)
    saver.save(sess=sess, save_path=os.path.join(output_path, 'w2x_metadata.ckpt'))
    print('Run `tensorboard --logdir={0}` to run visualize result on tensorboard'.format(output_path))


if __name__ == "__main__":
    model = KeyedVectors.load("data/glove2w2v_model")
    visualize(model, "/Users/limingxia/desktop/study/AI/scholar-svm/test")
    tf.disable_eager_execution()