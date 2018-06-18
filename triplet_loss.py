import keras.backend as K

alpha = 0.2  # used in FaceNet https://arxiv.org/pdf/1503.03832.pdf


def batch_cosine_similarity(x1, x2):

    dot = K.squeeze(K.batch_dot(x1, x2, axes=1), axis=1)
    logging.info('dot: {}'.format(dot))
    return dot


def deep_speaker_loss(y_true, y_pred):

    elements = int(K.int_shape(y_pred)[0] / 3)
    logging.info('elements={}'.format(elements))

    anchor = y_pred[0:elements]
    positive_ex = y_pred[elements:2 * elements]
    negative_ex = y_pred[2 * elements:]

    sap = batch_cosine_similarity(anchor, positive_ex)
    san = batch_cosine_similarity(anchor, negative_ex)
    loss = K.maximum(san - sap + alpha, 0.0)
    total_loss = K.sum(loss)
    return total_loss
