import os

import numpy as np
import tensorflow as tf

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


class Transformer(tf.keras.Model):
    def __init__(
        self,
        inputs_vocab_size,
        target_vocab_size,
        encoder_count,
        decoder_count,
        attention_head_count,
        d_model,
        d_point_wise_ff,
        dropout_prob,
    ):
        super(Transformer, self).__init__()

        # model hyper parameter variables
        self.encoder_count = encoder_count
        self.decoder_count = decoder_count
        self.attention_head_count = attention_head_count
        self.d_model = d_model
        self.d_point_wise_ff = d_point_wise_ff
        self.dropout_prob = dropout_prob

        self.encoder_embedding_layer = Embeddinglayer(inputs_vocab_size, d_model)
        self.encoder_embedding_dropout = tf.keras.layers.Dropout(dropout_prob)
        self.decoder_embedding_layer = Embeddinglayer(target_vocab_size, d_model)
        self.decoder_embedding_dropout = tf.keras.layers.Dropout(dropout_prob)

        self.encoder_layers = [
            EncoderLayer(attention_head_count, d_model, d_point_wise_ff, dropout_prob)
            for _ in range(encoder_count)
        ]

        self.decoder_layers = [
            DecoderLayer(attention_head_count, d_model, d_point_wise_ff, dropout_prob)
            for _ in range(decoder_count)
        ]

        self.linear = tf.keras.layers.Dense(target_vocab_size)

    def call(
        self,
        inputs,
        target=None,
        training=True,
        inputs_padding_mask=None,
        look_ahead_mask=None,
        target_padding_mask=None
    ):
        encoder_tensor = self.encoder_embedding_layer(inputs)
        encoder_tensor = self.encoder_embedding_dropout(
            encoder_tensor, training=training
        )

        for i in range(self.encoder_count):
            encoder_tensor, _ = self.encoder_layers[i](
                encoder_tensor, training=training, mask=inputs_padding_mask
            )
        return self.linear(encoder_tensor)
        # target = self.decoder_embedding_layer(target)
        # decoder_tensor = self.decoder_embedding_dropout(target, training=training)
        # for i in range(self.decoder_count):
        #     decoder_tensor, _, _ = self.decoder_layers[i](
        #         decoder_tensor,
        #         encoder_tensor,
        #         training,
        #         look_ahead_mask,
        #         target_padding_mask,
        #     )
        # return self.linear(decoder_tensor)


class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, attention_head_count, d_model, d_point_wise_ff, dropout_prob):
        super(EncoderLayer, self).__init__()

        # model hyper parameter variables
        self.attention_head_count = attention_head_count
        self.d_model = d_model
        self.d_point_wise_ff = d_point_wise_ff
        self.dropout_prob = dropout_prob

        self.multi_head_attention = MultiHeadAttention(attention_head_count, d_model)
        self.dropout_1 = tf.keras.layers.Dropout(dropout_prob)
        self.layer_norm_1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.position_wise_feed_forward_layer = PositionWiseFeedForwardLayer(
            d_point_wise_ff, d_model
        )
        self.dropout_2 = tf.keras.layers.Dropout(dropout_prob)
        self.layer_norm_2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

    def call(self, inputs, training, mask=None):
        output, attention = self.multi_head_attention(inputs, inputs, inputs, mask)
        output = self.dropout_1(output, training=training)
        output = self.layer_norm_1(tf.add(inputs, output))  # residual network
        output_temp = output

        output = self.position_wise_feed_forward_layer(output)
        output = self.dropout_2(output, training=training)
        output = self.layer_norm_2(tf.add(output_temp, output))  # correct

        return output, attention


class DecoderLayer(tf.keras.layers.Layer):
    def __init__(self, attention_head_count, d_model, d_point_wise_ff, dropout_prob):
        super(DecoderLayer, self).__init__()

        # model hyper parameter variables
        self.attention_head_count = attention_head_count
        self.d_model = d_model
        self.d_point_wise_ff = d_point_wise_ff
        self.dropout_prob = dropout_prob

        self.masked_multi_head_attention = MultiHeadAttention(
            attention_head_count, d_model
        )
        self.dropout_1 = tf.keras.layers.Dropout(dropout_prob)
        self.layer_norm_1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.encoder_decoder_attention = MultiHeadAttention(
            attention_head_count, d_model
        )
        self.dropout_2 = tf.keras.layers.Dropout(dropout_prob)
        self.layer_norm_2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.position_wise_feed_forward_layer = PositionWiseFeedForwardLayer(
            d_point_wise_ff, d_model
        )
        self.dropout_3 = tf.keras.layers.Dropout(dropout_prob)
        self.layer_norm_3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

    def call(
        self, decoder_inputs, encoder_output, training, look_ahead_mask=None, padding_mask=None
    ):
        output, attention_1 = self.masked_multi_head_attention(
            decoder_inputs, decoder_inputs, decoder_inputs, look_ahead_mask
        )
        output = self.dropout_1(output, training=training)
        query = self.layer_norm_1(tf.add(decoder_inputs, output))  # residual network
        output, attention_2 = self.encoder_decoder_attention(
            query, encoder_output, encoder_output, padding_mask
        )
        output = self.dropout_2(output, training=training)
        encoder_decoder_attention_output = self.layer_norm_2(tf.add(output, query))

        output = self.position_wise_feed_forward_layer(encoder_decoder_attention_output)
        output = self.dropout_3(output, training=training)
        output = self.layer_norm_3(
            tf.add(encoder_decoder_attention_output, output)
        )  # residual network

        return output, attention_1, attention_2


class PositionWiseFeedForwardLayer(tf.keras.layers.Layer):
    def __init__(self, d_point_wise_ff, d_model):
        super(PositionWiseFeedForwardLayer, self).__init__()
        self.w_1 = tf.keras.layers.Dense(d_point_wise_ff)
        self.w_2 = tf.keras.layers.Dense(d_model)

    def call(self, inputs):
        inputs = self.w_1(inputs)
        inputs = tf.nn.relu(inputs)
        return self.w_2(inputs)


class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, attention_head_count, d_model):
        super(MultiHeadAttention, self).__init__()

        # model hyper parameter variables
        self.attention_head_count = attention_head_count
        self.d_model = d_model

        if d_model % attention_head_count != 0:
            raise ValueError(
                "d_model({}) % attention_head_count({}) is not zero.d_model must be multiple of attention_head_count.".format(
                    d_model, attention_head_count
                )
            )

        self.d_h = d_model // attention_head_count

        self.w_query = tf.keras.layers.Dense(d_model)
        self.w_key = tf.keras.layers.Dense(d_model)
        self.w_value = tf.keras.layers.Dense(d_model)

        self.scaled_dot_product = ScaledDotProductAttention(self.d_h)

        self.ff = tf.keras.layers.Dense(d_model)

    def call(self, query, key, value, mask=None):
        batch_size = tf.shape(query)[0]

        query = self.w_query(query)
        key = self.w_key(key)
        value = self.w_value(value)

        query = self.split_head(query, batch_size)
        key = self.split_head(key, batch_size)
        value = self.split_head(value, batch_size)

        output, attention = self.scaled_dot_product(query, key, value, mask)
        output = self.concat_head(output, batch_size)

        return self.ff(output), attention

    def split_head(self, tensor, batch_size):
        # inputs tensor: (batch_size, seq_len, d_model)
        return tf.transpose(
            tf.reshape(
                tensor,
                (batch_size, -1, self.attention_head_count, self.d_h)
                # tensor: (batch_size, seq_len_splited, attention_head_count, d_h)
            ),
            [0, 2, 1, 3]
            # tensor: (batch_size, attention_head_count, seq_len_splited, d_h)
        )

    def concat_head(self, tensor, batch_size):
        return tf.reshape(
            tf.transpose(tensor, [0, 2, 1, 3]),
            (batch_size, -1, self.attention_head_count * self.d_h),
        )


class ScaledDotProductAttention(tf.keras.layers.Layer):
    def __init__(self, d_h):
        super(ScaledDotProductAttention, self).__init__()
        self.d_h = d_h

    def call(self, query, key, value, mask=None):
        matmul_q_and_transposed_k = tf.matmul(query, key, transpose_b=True)
        scale = tf.sqrt(tf.cast(self.d_h, dtype=tf.float32))
        scaled_attention_score = matmul_q_and_transposed_k / scale
        if mask is not None:
            scaled_attention_score += mask * -1e9

        attention_weight = tf.nn.softmax(scaled_attention_score, axis=-1)

        return tf.matmul(attention_weight, value), attention_weight


class Embeddinglayer(tf.keras.layers.Layer):
    def __init__(self, vocab_size, d_model):
        # model hyper parameter variables
        super(Embeddinglayer, self).__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model

        self.embedding = tf.keras.layers.Embedding(vocab_size, d_model)

    def call(self, sequences):
        max_sequence_len = sequences.shape[1]
        # output = self.embedding(sequences) * tf.sqrt(
        #     tf.cast(self.d_model, dtype=tf.float32)
        # )
        output = sequences
        output += self.positional_encoding(max_sequence_len)

        return output

    def positional_encoding(self, max_len):
        pos = np.expand_dims(np.arange(0, max_len), axis=1)
        index = np.expand_dims(np.arange(0, self.d_model), axis=0)

        pe = self.angle(pos, index)

        pe[:, 0::2] = np.sin(pe[:, 0::2])
        pe[:, 1::2] = np.cos(pe[:, 1::2])

        pe = np.expand_dims(pe, axis=0)
        return tf.cast(pe, dtype=tf.float32)

    def angle(self, pos, index):
        return pos / np.power(10000, (index - index % 2) / np.float32(self.d_model))


def build_transformer_model(input_shape, class_num):
    # Input Time-series
    inputs = tf.keras.layers.Input(shape=input_shape)

    # ResNet
    x = tf.keras.layers.Conv1D(64, 10, strides=5, padding="same", activation=tf.nn.gelu)(
        inputs
    )
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.1)(x)
    # x = tf.keras.layers.Flatten()(x)
    # x = tf.keras.layers.Conv1D(64, 3, strides=2, padding='same', activation=tf.nn.gelu, kernel_initializer='he_uniform')(x)
    # x = tf.keras.layers.BatchNormalization()(x)
    # x = tf.keras.layers.MaxPooling1D(3, strides=1, padding='same')(x)

    # x = tf.keras.layers.LSTM(64, return_sequences=True)(x)
    # x = tf.keras.layers.Dropout(0.3)(x)
    # x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=False))(inputs)
    # x = tf.keras.layers.LSTM(64, return_sequences=True)(x) # LSTM代替了positional encoding
    # x = tf.keras.layers.Dropout(0.3)(x)
    # x = tf.keras.layers.GRU(64, return_sequences=True)(x)
    # x = tf.keras.layers.LSTM(32, return_sequences=True, recurrent_activation='hard_sigmoid', activation=None)(x) # LSTM代替了positional encoding
    # x = tf.keras.layers.Dropout(0.3)(x)

    # Default params
    D_POINT_WISE_FF = 48
    D_MODEL = 64
    ENCODER_COUNT = DECODER_COUNT = 1
    ATTENTION_HEAD_COUNT = 2
    DROPOUT_PROB = 0.1
    SEQ_MAX_LEN_SOURCE = 100
    SEQ_MAX_LEN_TARGET = 100
    BPE_VOCAB_SIZE = 20

    x = Transformer(
        inputs_vocab_size=BPE_VOCAB_SIZE,
        target_vocab_size=BPE_VOCAB_SIZE,
        encoder_count=ENCODER_COUNT,
        decoder_count=DECODER_COUNT,
        attention_head_count=ATTENTION_HEAD_COUNT,
        d_model=D_MODEL,
        d_point_wise_ff=D_POINT_WISE_FF,
        dropout_prob=DROPOUT_PROB
    )(x)
    # 内部使用
    # x = tf.keras.layers.Add()([x, x2])
    x = x[:, 0, :]

    x = tf.keras.layers.Dense(32, activation="tanh")(x)
    x = tf.keras.layers.Dense(16, activation="tanh")(x)
    # Classify outputs.
    logits = tf.keras.layers.Dense(class_num, activation="softmax")(x)
    # Create the Keras model.
    model = tf.keras.Model(inputs=inputs, outputs=logits)

    optimizer = tf.keras.optimizers.Adam(
        learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, name="Adam"
    )
    # optimizer = tf.keras.optimizers.SGD(learning_rate=0.001, name='GradientDescent')
    # model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    model.compile(
        optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"]
    )
    # SparseCategoricalFocalLoss(gamma=2)
    model.summary()
    return model


# build_transformer_model((100, 1), class_num=2)
