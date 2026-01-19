import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

# -------------------------
# Time2Vec layer (multi-dim input support)
# -------------------------
class Time2Vec(layers.Layer):
    def __init__(self, k, **kwargs):
        """
        k: output dimension for Time2Vec
        Input: shape (..., time_feat_dim)
        Output: (..., k)
        """
        super().__init__(**kwargs)
        self.k = k

    def build(self, input_shape):
        input_dim = int(input_shape[-1])
        # linear projection weights for linear term (map input_dim -> 1)
        self.wb = self.add_weight(shape=(input_dim, 1), initializer="glorot_uniform", name="wb")
        self.bb = self.add_weight(shape=(1,), initializer="zeros", name="bb")
        # periodic term weights (map input_dim -> k-1)
        self.wa = self.add_weight(shape=(input_dim, self.k - 1), initializer="glorot_uniform", name="wa")
        self.ba = self.add_weight(shape=(1, self.k - 1), initializer="zeros", name="ba")
        super().build(input_shape)

    def call(self, x):
        # x shape: (..., time_feat_dim)
        # compute linear term: (..., 1)
        lin = tf.matmul(x, self.wb) + self.bb  # broadcasts over leading dims
        # periodic: (..., k-1)
        per = tf.math.sin(tf.matmul(x, self.wa) + self.ba)
        return tf.concat([lin, per], axis=-1)  # (..., k)

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"k": self.k})
        return cfg


# -------------------------
# Transformer building blocks
# -------------------------
def transformer_encoder_block(d_model, num_heads, d_ff, dropout=0.1, name=None):
    mha = layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model // num_heads, name=(name and name + "_mha"))
    ln1 = layers.LayerNormalization(epsilon=1e-6, name=(name and name + "_ln1"))
    ff = keras.Sequential([
        layers.Dense(d_ff, activation="leaky_relu"),
        layers.Dense(d_model),
    ], name=(name and name + "_ff"))
    ln2 = layers.LayerNormalization(epsilon=1e-6, name=(name and name + "_ln2"))
    drop1 = layers.Dropout(dropout)
    drop2 = layers.Dropout(dropout)

    def call(x, training=None):
        # x: (batch, seq_len, d_model)
        attn_out = mha(x, x, x, training=training)  # self-attention
        attn_out = drop1(attn_out, training=training)
        x = ln1(x + attn_out)
        ff_out = ff(x, training=training)
        ff_out = drop2(ff_out, training=training)
        x = ln2(x + ff_out)
        return x
    return call


def transformer_decoder_block(d_model, num_heads, d_ff, dropout=0.1, name=None):
    # We'll return a function with internal layers so we can reuse it easily
    self_attn = layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model // num_heads, name=(name and name + "_self_attn"))
    cross_attn = layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model // num_heads, name=(name and name + "_cross_attn"))
    ln1 = layers.LayerNormalization(epsilon=1e-6, name=(name and name + "_ln1"))
    ln2 = layers.LayerNormalization(epsilon=1e-6, name=(name and name + "_ln2"))
    ln3 = layers.LayerNormalization(epsilon=1e-6, name=(name and name + "_ln3"))
    ff = keras.Sequential([layers.Dense(d_ff, activation="leaky_relu"), layers.Dense(d_model)], name=(name and name + "_ff"))
    drop = layers.Dropout(dropout)

    def call(x, enc_out, look_ahead_mask=None, training=None):
        # x: (batch, target_len, d_model)
        # enc_out: (batch, enc_len, d_model)
        # look_ahead_mask: (batch, target_len, target_len) boolean or float mask
        attn1 = self_attn(x, x, x, attention_mask=look_ahead_mask, training=training)  # causal self-attention
        x = ln1(x + drop(attn1, training=training))
        attn2 = cross_attn(x, enc_out, enc_out, training=training)  # cross-attention to encoder output
        x = ln2(x + drop(attn2, training=training))
        ff_out = ff(x, training=training)
        x = ln3(x + drop(ff_out, training=training))
        return x

    return call


# -------------------------
# Simple utilities
# -------------------------
def create_look_ahead_mask(seq_len):
    # Returns a mask shape (1, seq_len, seq_len) so it can broadcast over batch
    # allowed positions are 1, future masked are 0 for attention_mask usage.
    mask = tf.linalg.band_part(tf.ones((seq_len, seq_len)), -1, 0)  # lower triangular
    # Keras MultiHeadAttention expects a mask where 1 indicates keep and 0 indicates mask,
    # but many versions accept boolean True for keep. We'll convert to float.
    # Shape needed: (1, seq_len, seq_len)
    return tf.expand_dims(mask, axis=0)


# -------------------------
# Build encoder-decoder model factory
# -------------------------
def build_transformer_forecaster(
    enc_len,
    target_len,
    in_dim=1,
    out_dim=1,
    time_dim=3,
    time2vec_dim=32,
    d_model=256,
    num_heads=8,
    d_ff=512,
    enc_layers=3,
    dec_layers=2,
    dropout=0.25,
):
    """
    Returns a keras Model for training (teacher forcing).
    Inputs:
      - encoder_values: (batch, enc_len, in_dim)
      - encoder_time:   (batch, enc_len, time_dim)
      - decoder_values: (batch, target_len, in_dim)   # shifted targets during training
      - decoder_time:   (batch, target_len, time_dim)
    Output:
      - preds: (batch, target_len, out_dim)
    """
    # Inputs
    encoder_values = layers.Input(shape=(enc_len, in_dim), name="encoder_values")       # (B, enc_len, in_dim)
    encoder_time = layers.Input(shape=(enc_len, time_dim), name="encoder_time")         # (B, enc_len, time_dim)
    decoder_values = layers.Input(shape=(target_len, 1), name="decoder_values")    # (B, target_len, in_dim)
    decoder_time = layers.Input(shape=(target_len, time_dim), name="decoder_time")      # (B, target_len, time_dim)

    # Time2Vec embeddings (applied to time features)
    t2v = Time2Vec(k=time2vec_dim)
    enc_time_emb = t2v(encoder_time)  # (B, enc_len, time2vec_dim)
    dec_time_emb = t2v(decoder_time)  # (B, target_len, time2vec_dim)

    # Value projection (project numeric features into d_model)
    enc_val_proj = layers.Dense(d_model, name="enc_val_proj")(encoder_values)  # (B, enc_len, d_model)
    dec_val_proj = layers.Dense(d_model, name="dec_val_proj")(decoder_values)  # (B, target_len, d_model)

    # Time projection & sum/concat choice
    # Here we **concatenate** and project to d_model to preserve both signal types.
    enc_comb = layers.Concatenate(axis=-1)([enc_val_proj, enc_time_emb])  # (B, enc_len, d_model + time2vec_dim)
    enc_input = layers.Dense(d_model, name="enc_input_proj")(enc_comb)    # (B, enc_len, d_model)

    dec_comb = layers.Concatenate(axis=-1)([dec_val_proj, dec_time_emb])  # (B, target_len, d_model + time2vec_dim)
    dec_input = layers.Dense(d_model, name="dec_input_proj")(dec_comb)     # (B, target_len, d_model)

    # Encoder stack
    x = enc_input
    for i in range(enc_layers):
        block = transformer_encoder_block(d_model, num_heads, d_ff, dropout=dropout, name=f"encblock{i}")
        x = block(x)
        # optional downsampling/distillation can be inserted here if desired
    enc_out = x  # (B, enc_len, d_model)

    # Decoder stack
    y = dec_input
    look_mask = create_look_ahead_mask(target_len)  # shape (1, target_len, target_len)
    for i in range(dec_layers):
        block = transformer_decoder_block(d_model, num_heads, d_ff, dropout=dropout, name=f"decblock{i}")
        y = block(y, enc_out, look_ahead_mask=look_mask)
    dec_out = y  # (B, target_len, d_model)

    # Final projection to prediction dimension
    preds = layers.Dense(out_dim, name="predictions")(dec_out)  # (B, target_len, out_dim)

    model = keras.Model(
        inputs=[encoder_values, encoder_time, decoder_values, decoder_time],
        outputs=preds,
        name="transformer_forecaster"
    )
    return model


# -------------------------
# Helper: build inference wrapper
# -------------------------
class InferenceWrapper(keras.Model):
    """
    Wraps the trained encoder-decoder model to run inference.
    Supports:
      - 'learned' start tokens: parallel one-shot prediction
      - 'autoregressive': step-by-step generation (uses previous predicted value as next input)
    """
    def __init__(self, trained_model, enc_len, target_len, in_dim=1, d_model=64, strategy="learned"):
        """
        trained_model: Keras model built by build_transformer_forecaster
        strategy: 'learned' or 'autoregressive'
        """
        super().__init__()
        self.trained_model = trained_model
        self.enc_len = enc_len
        self.target_len = target_len
        self.in_dim = in_dim
        self.d_model = d_model
        self.strategy = strategy

        if strategy == "learned":
            # learned decoder start tokens in value-space (projected later inside model)
            self.decoder_start = self.add_weight(shape=(1, target_len, in_dim),
                                                 initializer="zeros",
                                                 trainable=True,
                                                 name="decoder_start")

    def call(self, encoder_values, encoder_time, decoder_time, training=False):
        """
        encoder_values: (B, enc_len, in_dim)
        encoder_time:   (B, enc_len, time_dim)
        decoder_time:   (B, target_len, time_dim)
        returns: preds (B, target_len, out_dim)
        """
        batch = tf.shape(encoder_values)[0]
        if self.strategy == "learned":
            # tile learned start tokens to batch
            decoder_values = tf.tile(self.decoder_start, [batch, 1, 1])  # (B, target_len, in_dim)
            # Call the underlying model (parallel)
            preds = self.trained_model([encoder_values, encoder_time, decoder_values, decoder_time], training=training)
            return preds

        elif self.strategy == "autoregressive":
            # step-by-step generation using model as a function that expects full decoder sequence
            # We'll append one prediction at a time (inefficient but straightforward)
            preds = []
            # start with zeros as first decoder input (batch, 0, in_dim) then append
            cur_dec_inputs = tf.zeros((batch, 0, self.in_dim), dtype=encoder_values.dtype)
            for t in range(self.target_len):
                # prepare decoder values: pad to length t+1 with previous preds (or zeros)
                if t == 0:
                    # first input zeros
                    cur = tf.zeros((batch, 1, self.in_dim), dtype=encoder_values.dtype)
                else:
                    cur = preds[-1]  # last predicted step shape (batch, 1, out_dim) ; ensure same dim as in_dim
                # build full decoder_values by concatenation
                # convert preds list to tensor each loop (inefficient). For production optimize with tensors.
                if t == 0:
                    dec_vals = cur
                else:
                    dec_vals = tf.concat(preds + [cur], axis=1)  # (batch, t+1, in_dim or out_dim)
                # pad dec_vals to target_len by zeros so shape matches expected input (we'll send full target_len width)
                pad_len = self.target_len - tf.shape(dec_vals)[1]
                if pad_len > 0:
                    pad_tensor = tf.zeros((batch, pad_len, self.in_dim), dtype=encoder_values.dtype)
                    dec_padded = tf.concat([dec_vals, pad_tensor], axis=1)
                else:
                    dec_padded = dec_vals[:, :self.target_len, :]
                # call model and take t-th output
                out_all = self.trained_model([encoder_values, encoder_time, dec_padded, decoder_time], training=False)
                step_pred = tf.expand_dims(out_all[:, t, :], axis=1)  # (batch,1,out_dim)
                preds.append(step_pred)
            preds = tf.concat(preds, axis=1)  # (batch, target_len, out_dim)
            return preds
        else:
            raise ValueError("Unknown strategy")


# -------------------------
# Example usage
# -------------------------
if __name__ == "__main__":
    # Example sizes
    B = 4
    ENC_LEN = 96
    TARGET_LEN = 24
    IN_DIM = 1
    TIME_DIM = 3
    OUT_DIM = 1

    # Build training model
    model = build_transformer_forecaster(
        enc_len=ENC_LEN, target_len=TARGET_LEN,
        in_dim=IN_DIM, out_dim=OUT_DIM, time_dim=TIME_DIM,
        time2vec_dim=32, d_model=256, num_heads=8, d_ff=512,
        enc_layers=2, dec_layers=2, dropout=0.25
    )
    model.summary()

    # Fake data for quick sanity check
    enc_vals = np.random.randn(B, ENC_LEN, IN_DIM).astype(np.float32)
    enc_time = np.random.rand(B, ENC_LEN, TIME_DIM).astype(np.float32)   # e.g., normalized day/month/etc.
    dec_vals = np.random.randn(B, TARGET_LEN, IN_DIM).astype(np.float32) # shifted targets during training
    dec_time = np.random.rand(B, TARGET_LEN, TIME_DIM).astype(np.float32)

    preds = model([enc_vals, enc_time, dec_vals, dec_time])
    print("preds shape (training):", preds.shape)  # (B, TARGET_LEN, OUT_DIM)

    # Wrap for inference (learned parallel)
    wrapper = InferenceWrapper(model, enc_len=ENC_LEN, target_len=TARGET_LEN, in_dim=IN_DIM, strategy="learned")
    preds_inf = wrapper(enc_vals, enc_time, dec_time)
    print("preds shape (inference learned):", preds_inf.shape)

    # Wrap for inference (autoregressive)
    wrapper_ar = InferenceWrapper(model, enc_len=ENC_LEN, target_len=TARGET_LEN, in_dim=IN_DIM, strategy="autoregressive")
    preds_ar = wrapper_ar(enc_vals, enc_time, dec_time)
    print("preds shape (inference autoregressive):", preds_ar.shape)
