import io
import numpy as np
import tensorflow as tf
from hparams import hparams
from librosa import effects
from models import create_model
from text import text_to_sequence
from util import audio
from g2pk import G2p


class Synthesizer:
    def load(self, checkpoint_path, model_name='tacotron'):
        print('Constructing model: %s' % model_name)
        c_inputs = tf.placeholder(tf.int32, [1, None], 'c_inputs')
        p_inputs = tf.placeholder(tf.int32, [1, None], 'p_inputs')
        c_input_lengths = tf.placeholder(tf.int32, [1], 'c_input_lengths')
        p_input_lengths = tf.placeholder(tf.int32, [1], 'p_input_lengths')
        with tf.variable_scope('model') as scope:
            self.model = create_model(model_name, hparams)
            self.model.initialize(c_inputs, p_inputs, c_input_lengths, p_input_lengths)
            self.wav_output = audio.inv_spectrogram_tensorflow(self.model.linear_outputs[0])

        print('Loading checkpoint: %s' % checkpoint_path)
        self.session = tf.Session()
        self.session.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        saver.restore(self.session, checkpoint_path)

    def synthesize(self, text):
        cleaner_names = [x.strip() for x in hparams.cleaners.split(',')]
        g2p = G2p()
        c_seq = text_to_sequence('정성스럽게 적었던 거야.', cleaner_names)
        p_seq = text_to_sequence('정성스럽께 저걷떤 거야.', cleaner_names)
        feed_dict = {
            self.model.c_inputs: [np.asarray(c_seq, dtype=np.int32)],
            self.model.p_inputs: [np.asarray(p_seq, dtype=np.int32)],
            self.model.c_input_lengths: np.asarray([len(c_seq)], dtype=np.int32),
            self.model.p_input_lengths: np.asarray([len(p_seq)], dtype=np.int32)
        }
        wav = self.session.run(self.wav_output, feed_dict=feed_dict)
        wav = audio.inv_preemphasis(wav)
        wav = wav[:audio.find_endpoint(wav)]
        out = io.BytesIO()
        audio.save_wav(wav, out)
        return out.getvalue()
