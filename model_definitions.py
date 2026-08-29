import keras
from keras import layers
import numpy as np
import random
import tensorflow as tf
import datetime

import encode_attributes
from conquestdb_data import vocab, planet_vocab, attachment_vocab, hand_vocab, in_play_vocab, discard_vocab, \
    headquarters_vocab
from GameObject import Game

import moveFilter

from maxValues import OUTPUT_DIM, MAX_HAND_SIZE, MAX_CARDS_IN_ONE_PLAY_ZONE, CARD_FEATURE_DIM, MAX_ACTIONS, \
    MAX_ACTION_ARGS, VOCAB_SIZE, NUM_SCALARS, NUM_ACTION_TYPES, NUM_PLANETS, PLANET_VOCAB_SIZE, \
    MAX_ATTACHMENTS_PER_CARD, ATTACHMENT_VOCAB_SIZE, MAX_DISCARD_SIZE, NUM_SPECIAL_CHOICES, \
    UNIQUE_CHOICES, NUM_HEADS, FF_DIM, DROPOUT_RATE, HEADQUARTERS_VOCAB_SIZE, HAND_VOCAB_SIZE, DISCARD_VOCAB_SIZE, IN_PLAY_VOCAB_SIZE, MAX_SEARCHED_CARDS, MAX_PREVIOUS_ACTIONS, HEAD_SIZE


def transformer_block(inputs, head_size, num_heads, ff_dim, dropout_rate=DROPOUT_RATE):
    attn_output = layers.MultiHeadAttention(
        num_heads=num_heads,
        key_dim=head_size
    )(inputs, inputs)
    attn_output = layers.Dropout(dropout_rate)(attn_output)
    output_1 = layers.Normalization()(inputs + attn_output)

    output_2 = layers.Dense(ff_dim, activation="relu")(output_1)
    output_2 = layers.Dense(OUTPUT_DIM)(output_2)
    output_2 = layers.Dropout(dropout_rate)(output_2)
    final_output = layers.LayerNormalization()(output_1 + output_2)
    return final_output


def create_simplified_model():
    hand_input_1 = keras.Input(shape=(MAX_HAND_SIZE, CARD_FEATURE_DIM), name="hand_1")
    selected_card_hand_data_input = keras.Input(shape=(1, CARD_FEATURE_DIM), name="hand_selected_card")
    # discard_input_1 = keras.Input(shape=(MAX_DISCARD_SIZE,), name="discard_1")
    hq_input_1 = keras.Input(shape=(MAX_CARDS_IN_ONE_PLAY_ZONE, CARD_FEATURE_DIM), name="hq_1")
    in_play_input_1_0 = keras.Input(shape=(MAX_CARDS_IN_ONE_PLAY_ZONE, CARD_FEATURE_DIM), name="in_play_1_0")
    in_play_input_1_1 = keras.Input(shape=(MAX_CARDS_IN_ONE_PLAY_ZONE, CARD_FEATURE_DIM), name="in_play_1_1")
    in_play_input_1_2 = keras.Input(shape=(MAX_CARDS_IN_ONE_PLAY_ZONE, CARD_FEATURE_DIM), name="in_play_1_2")
    in_play_input_1_3 = keras.Input(shape=(MAX_CARDS_IN_ONE_PLAY_ZONE, CARD_FEATURE_DIM), name="in_play_1_3")
    in_play_input_1_4 = keras.Input(shape=(MAX_CARDS_IN_ONE_PLAY_ZONE, CARD_FEATURE_DIM), name="in_play_1_4")

    hq_input_2 = keras.Input(shape=(MAX_CARDS_IN_ONE_PLAY_ZONE, CARD_FEATURE_DIM), name="hq_2")
    in_play_input_2_0 = keras.Input(shape=(MAX_CARDS_IN_ONE_PLAY_ZONE, CARD_FEATURE_DIM), name="in_play_2_0")
    in_play_input_2_1 = keras.Input(shape=(MAX_CARDS_IN_ONE_PLAY_ZONE, CARD_FEATURE_DIM), name="in_play_2_1")
    in_play_input_2_2 = keras.Input(shape=(MAX_CARDS_IN_ONE_PLAY_ZONE, CARD_FEATURE_DIM), name="in_play_2_2")
    in_play_input_2_3 = keras.Input(shape=(MAX_CARDS_IN_ONE_PLAY_ZONE, CARD_FEATURE_DIM), name="in_play_2_3")
    in_play_input_2_4 = keras.Input(shape=(MAX_CARDS_IN_ONE_PLAY_ZONE, CARD_FEATURE_DIM), name="in_play_2_4")

    # search_input = keras.Input(shape=(MAX_SEARCHED_CARDS,), name="search")
    planets_input = keras.Input(shape=(NUM_PLANETS,), name="planets")

    scalar_inputs = keras.Input(shape=(NUM_SCALARS,), name="scalars")
    action_type_input = keras.Input(shape=(MAX_ACTIONS,), name="action_types")
    action_args_input = keras.Input(shape=(MAX_ACTIONS, MAX_ACTION_ARGS), name="action_args")
    action_relevant_card_input = keras.Input(shape=(MAX_ACTIONS, CARD_FEATURE_DIM), name="action_card_input")
    mask_input = keras.Input(shape=(MAX_ACTIONS,), name="action_mask")
    # previous_actions_type_input = keras.Input(shape=(MAX_PREVIOUS_ACTIONS,), name="previous_action_types")
    # previous_actions_args_input = keras.Input(shape=(MAX_PREVIOUS_ACTIONS, MAX_ACTION_ARGS), name="previous_action_args")

    # discard_emb = layers.Embedding(DISCARD_VOCAB_SIZE, OUTPUT_DIM, name="discard_name_embedding")
    planet_emb = layers.Embedding(PLANET_VOCAB_SIZE, OUTPUT_DIM, name="planet_name_embedding")

    hand_1_enc = layers.GlobalAveragePooling1D()(hand_input_1)
    hand_1_enc = layers.Dense(FF_DIM, activation="relu")(hand_1_enc)
    hand_1_enc = layers.Dropout(DROPOUT_RATE)(hand_1_enc)
    selected_hand_card_enc = layers.GlobalAveragePooling1D()(selected_card_hand_data_input)
    selected_hand_card_enc = layers.Dense(FF_DIM, activation="relu")(selected_hand_card_enc)
    selected_hand_card_enc = layers.Dropout(DROPOUT_RATE)(selected_hand_card_enc)

    planet_enc = layers.GlobalAveragePooling1D()(planet_emb(planets_input))

    hq_1_enc = layers.GlobalAveragePooling1D()(hq_input_1)
    hq_1_enc = layers.Dense(FF_DIM, activation="relu")(hq_1_enc)
    hq_1_enc = layers.Dropout(DROPOUT_RATE)(hq_1_enc)
    in_play_1_0_enc = layers.GlobalAveragePooling1D()(in_play_input_1_0)
    in_play_1_0_enc = layers.Dense(FF_DIM, activation="relu")(in_play_1_0_enc)
    in_play_1_0_enc = layers.Dropout(DROPOUT_RATE)(in_play_1_0_enc)
    in_play_1_1_enc = layers.GlobalAveragePooling1D()(in_play_input_1_1)
    in_play_1_1_enc = layers.Dense(FF_DIM, activation="relu")(in_play_1_1_enc)
    in_play_1_1_enc = layers.Dropout(DROPOUT_RATE)(in_play_1_1_enc)
    in_play_1_2_enc = layers.GlobalAveragePooling1D()(in_play_input_1_2)
    in_play_1_2_enc = layers.Dense(FF_DIM, activation="relu")(in_play_1_2_enc)
    in_play_1_2_enc = layers.Dropout(DROPOUT_RATE)(in_play_1_2_enc)
    in_play_1_3_enc = layers.GlobalAveragePooling1D()(in_play_input_1_3)
    in_play_1_3_enc = layers.Dense(FF_DIM, activation="relu")(in_play_1_3_enc)
    in_play_1_3_enc = layers.Dropout(DROPOUT_RATE)(in_play_1_3_enc)
    in_play_1_4_enc = layers.GlobalAveragePooling1D()(in_play_input_1_4)
    in_play_1_4_enc = layers.Dense(FF_DIM, activation="relu")(in_play_1_4_enc)
    in_play_1_4_enc = layers.Dropout(DROPOUT_RATE)(in_play_1_4_enc)

    hq_2_enc = layers.GlobalAveragePooling1D()(hq_input_2)
    hq_2_enc = layers.Dense(FF_DIM, activation="relu")(hq_2_enc)
    hq_2_enc = layers.Dropout(DROPOUT_RATE)(hq_2_enc)
    in_play_2_0_enc = layers.GlobalAveragePooling1D()(in_play_input_2_0)
    in_play_2_0_enc = layers.Dense(FF_DIM, activation="relu")(in_play_2_0_enc)
    in_play_2_0_enc = layers.Dropout(DROPOUT_RATE)(in_play_2_0_enc)
    in_play_2_1_enc = layers.GlobalAveragePooling1D()(in_play_input_2_1)
    in_play_2_1_enc = layers.Dense(FF_DIM, activation="relu")(in_play_2_1_enc)
    in_play_2_1_enc = layers.Dropout(DROPOUT_RATE)(in_play_2_1_enc)
    in_play_2_2_enc = layers.GlobalAveragePooling1D()(in_play_input_2_2)
    in_play_2_2_enc = layers.Dense(FF_DIM, activation="relu")(in_play_2_2_enc)
    in_play_2_2_enc = layers.Dropout(DROPOUT_RATE)(in_play_2_2_enc)
    in_play_2_3_enc = layers.GlobalAveragePooling1D()(in_play_input_2_3)
    in_play_2_3_enc = layers.Dense(FF_DIM, activation="relu")(in_play_2_3_enc)
    in_play_2_3_enc = layers.Dropout(DROPOUT_RATE)(in_play_2_3_enc)
    in_play_2_4_enc = layers.GlobalAveragePooling1D()(in_play_input_2_4)
    in_play_2_4_enc = layers.Dense(FF_DIM, activation="relu")(in_play_2_4_enc)
    in_play_2_4_enc = layers.Dropout(DROPOUT_RATE)(in_play_2_4_enc)

    scalar_enc = layers.Dense(8, activation="relu")(scalar_inputs)
    scalar_enc = layers.Dropout(DROPOUT_RATE)(scalar_enc)

    game_state = layers.Concatenate(axis=-1)(
        [
            hand_1_enc, hq_1_enc, hq_2_enc,
            in_play_1_0_enc, in_play_1_1_enc, in_play_1_2_enc, in_play_1_3_enc, in_play_1_4_enc,
            in_play_2_0_enc, in_play_2_1_enc, in_play_2_2_enc, in_play_2_3_enc, in_play_2_4_enc,
            planet_enc, scalar_enc, selected_hand_card_enc
        ]
    )
    game_state = layers.Dense(64, activation="relu")(game_state)
    # game_state = layers.Dense(64, activation="relu")(game_state)

    type_emb = layers.Embedding(NUM_ACTION_TYPES, 8)(action_type_input)
    action_enc = layers.Concatenate()([type_emb, action_args_input, action_relevant_card_input])

    game_context = layers.RepeatVector(MAX_ACTIONS)(game_state)
    action_scored = layers.Concatenate()([action_enc, game_context])
    action_scored = layers.TimeDistributed(layers.Dense(FF_DIM, activation="relu"))(action_scored)

    logits = layers.TimeDistributed(layers.Dense(1))(action_scored)
    logits = layers.Flatten()(logits)

    masked_logits = logits + (1.0 - mask_input) * (tf.float32.min)
    policy_output = layers.Softmax(name="policy")(masked_logits)

    value_output = layers.Dense(1, name="value")(game_state)

    model = keras.Model(
        inputs=[scalar_inputs, hand_input_1, hq_input_1, hq_input_2, 
                in_play_input_1_0, in_play_input_1_1, in_play_input_1_2, in_play_input_1_3, in_play_input_1_4,
                in_play_input_2_0, in_play_input_2_1, in_play_input_2_2, in_play_input_2_3, in_play_input_2_4,
                planets_input, action_type_input, action_args_input, mask_input, action_relevant_card_input, selected_card_hand_data_input],
        outputs={"policy": policy_output, "value": value_output}
    )
    return model


def create_simplified_transformer_model():
    hand_input_1 = keras.Input(shape=(MAX_HAND_SIZE, CARD_FEATURE_DIM), name="hand_1")
    selected_card_hand_data_input = keras.Input(shape=(1, CARD_FEATURE_DIM), name="hand_selected_card")
    hq_input_1 = keras.Input(shape=(MAX_CARDS_IN_ONE_PLAY_ZONE, CARD_FEATURE_DIM), name="hq_1")
    in_play_input_1_0 = keras.Input(shape=(MAX_CARDS_IN_ONE_PLAY_ZONE, CARD_FEATURE_DIM), name="in_play_1_0")
    in_play_input_1_1 = keras.Input(shape=(MAX_CARDS_IN_ONE_PLAY_ZONE, CARD_FEATURE_DIM), name="in_play_1_1")
    in_play_input_1_2 = keras.Input(shape=(MAX_CARDS_IN_ONE_PLAY_ZONE, CARD_FEATURE_DIM), name="in_play_1_2")
    in_play_input_1_3 = keras.Input(shape=(MAX_CARDS_IN_ONE_PLAY_ZONE, CARD_FEATURE_DIM), name="in_play_1_3")
    in_play_input_1_4 = keras.Input(shape=(MAX_CARDS_IN_ONE_PLAY_ZONE, CARD_FEATURE_DIM), name="in_play_1_4")

    hq_input_2 = keras.Input(shape=(MAX_CARDS_IN_ONE_PLAY_ZONE, CARD_FEATURE_DIM), name="hq_2")
    in_play_input_2_0 = keras.Input(shape=(MAX_CARDS_IN_ONE_PLAY_ZONE, CARD_FEATURE_DIM), name="in_play_2_0")
    in_play_input_2_1 = keras.Input(shape=(MAX_CARDS_IN_ONE_PLAY_ZONE, CARD_FEATURE_DIM), name="in_play_2_1")
    in_play_input_2_2 = keras.Input(shape=(MAX_CARDS_IN_ONE_PLAY_ZONE, CARD_FEATURE_DIM), name="in_play_2_2")
    in_play_input_2_3 = keras.Input(shape=(MAX_CARDS_IN_ONE_PLAY_ZONE, CARD_FEATURE_DIM), name="in_play_2_3")
    in_play_input_2_4 = keras.Input(shape=(MAX_CARDS_IN_ONE_PLAY_ZONE, CARD_FEATURE_DIM), name="in_play_2_4")

    planets_input = keras.Input(shape=(NUM_PLANETS,), name="planets")

    scalar_inputs = keras.Input(shape=(NUM_SCALARS,), name="scalars")
    action_type_input = keras.Input(shape=(MAX_ACTIONS,), name="action_types")
    action_args_input = keras.Input(shape=(MAX_ACTIONS, MAX_ACTION_ARGS), name="action_args")
    action_relevant_card_input = keras.Input(shape=(MAX_ACTIONS, CARD_FEATURE_DIM), name="action_card_input")
    mask_input = keras.Input(shape=(MAX_ACTIONS,), name="action_mask")

    planet_emb = layers.Embedding(PLANET_VOCAB_SIZE, OUTPUT_DIM, name="planet_name_embedding")

    hand_1_projected = layers.Dense(OUTPUT_DIM)(hand_input_1)
    hand_1_enc = transformer_block(hand_1_projected, HEAD_SIZE, NUM_HEADS, FF_DIM)
    hand_1_enc = layers.GlobalAveragePooling1D()(hand_1_enc)
    selected_hand_card_enc = layers.GlobalAveragePooling1D()(selected_card_hand_data_input)
    selected_hand_card_enc = layers.Dense(FF_DIM, activation="relu")(selected_hand_card_enc)
    selected_hand_card_enc = layers.Dropout(DROPOUT_RATE)(selected_hand_card_enc)

    planet_enc = layers.GlobalAveragePooling1D()(planet_emb(planets_input))

    hq_1_enc = layers.GlobalAveragePooling1D()(hq_input_1)
    hq_1_enc = layers.Dense(FF_DIM, activation="relu")(hq_1_enc)
    hq_1_enc = layers.Dropout(DROPOUT_RATE)(hq_1_enc)
    in_play_1_0_projected = layers.Dense(OUTPUT_DIM)(in_play_input_1_0)
    in_play_1_0_enc = transformer_block(in_play_1_0_projected, HEAD_SIZE, NUM_HEADS, FF_DIM)
    in_play_1_0_enc = layers.GlobalAveragePooling1D()(in_play_1_0_enc)
    in_play_1_1_projected = layers.Dense(OUTPUT_DIM)(in_play_input_1_1)
    in_play_1_1_enc = transformer_block(in_play_1_1_projected, HEAD_SIZE, NUM_HEADS, FF_DIM)
    in_play_1_1_enc = layers.GlobalAveragePooling1D()(in_play_1_1_enc)
    in_play_1_2_projected = layers.Dense(OUTPUT_DIM)(in_play_input_1_2)
    in_play_1_2_enc = transformer_block(in_play_1_2_projected, HEAD_SIZE, NUM_HEADS, FF_DIM)
    in_play_1_2_enc = layers.GlobalAveragePooling1D()(in_play_1_2_enc)
    in_play_1_3_projected = layers.Dense(OUTPUT_DIM)(in_play_input_1_3)
    in_play_1_3_enc = transformer_block(in_play_1_3_projected, HEAD_SIZE, NUM_HEADS, FF_DIM)
    in_play_1_3_enc = layers.GlobalAveragePooling1D()(in_play_1_3_enc)
    in_play_1_4_projected = layers.Dense(OUTPUT_DIM)(in_play_input_1_4)
    in_play_1_4_enc = transformer_block(in_play_1_4_projected, HEAD_SIZE, NUM_HEADS, FF_DIM)
    in_play_1_4_enc = layers.GlobalAveragePooling1D()(in_play_1_4_enc)

    hq_2_enc = layers.GlobalAveragePooling1D()(hq_input_2)
    hq_2_enc = layers.Dense(FF_DIM, activation="relu")(hq_2_enc)
    hq_2_enc = layers.Dropout(DROPOUT_RATE)(hq_2_enc)
    in_play_2_0_projected = layers.Dense(OUTPUT_DIM)(in_play_input_2_0)
    in_play_2_0_enc = transformer_block(in_play_2_0_projected, HEAD_SIZE, NUM_HEADS, FF_DIM)
    in_play_2_0_enc = layers.GlobalAveragePooling1D()(in_play_2_0_enc)
    in_play_2_1_projected = layers.Dense(OUTPUT_DIM)(in_play_input_2_1)
    in_play_2_1_enc = transformer_block(in_play_2_1_projected, HEAD_SIZE, NUM_HEADS, FF_DIM)
    in_play_2_1_enc = layers.GlobalAveragePooling1D()(in_play_2_1_enc)
    in_play_2_2_projected = layers.Dense(OUTPUT_DIM)(in_play_input_2_2)
    in_play_2_2_enc = transformer_block(in_play_2_2_projected, HEAD_SIZE, NUM_HEADS, FF_DIM)
    in_play_2_2_enc = layers.GlobalAveragePooling1D()(in_play_2_2_enc)
    in_play_2_3_projected = layers.Dense(OUTPUT_DIM)(in_play_input_2_3)
    in_play_2_3_enc = transformer_block(in_play_2_3_projected, HEAD_SIZE, NUM_HEADS, FF_DIM)
    in_play_2_3_enc = layers.GlobalAveragePooling1D()(in_play_2_3_enc)
    in_play_2_4_projected = layers.Dense(OUTPUT_DIM)(in_play_input_2_4)
    in_play_2_4_enc = transformer_block(in_play_2_4_projected, HEAD_SIZE, NUM_HEADS, FF_DIM)
    in_play_2_4_enc = layers.GlobalAveragePooling1D()(in_play_2_4_enc)

    scalar_enc = layers.Dense(8, activation="relu")(scalar_inputs)
    scalar_enc = layers.Dropout(DROPOUT_RATE)(scalar_enc)

    game_state = layers.Concatenate(axis=-1)(
        [
            hand_1_enc, hq_1_enc, hq_2_enc,
            in_play_1_0_enc, in_play_1_1_enc, in_play_1_2_enc, in_play_1_3_enc, in_play_1_4_enc,
            in_play_2_0_enc, in_play_2_1_enc, in_play_2_2_enc, in_play_2_3_enc, in_play_2_4_enc,
            planet_enc, scalar_enc, selected_hand_card_enc
        ]
    )
    game_state = layers.Dense(64, activation="relu")(game_state)
    # game_state = layers.Dense(64, activation="relu")(game_state)

    type_emb = layers.Embedding(NUM_ACTION_TYPES, 8)(action_type_input)
    action_enc = layers.Concatenate()([type_emb, action_args_input, action_relevant_card_input])
    action_enc = layers.Dense(OUTPUT_DIM)(action_enc)
    action_enc = transformer_block(action_enc, HEAD_SIZE, NUM_HEADS, FF_DIM)

    game_context = layers.RepeatVector(MAX_ACTIONS)(game_state)
    action_scored = layers.Concatenate()([action_enc, game_context])
    action_scored = layers.TimeDistributed(layers.Dense(FF_DIM, activation="relu"))(action_scored)

    logits = layers.TimeDistributed(layers.Dense(1))(action_scored)
    logits = layers.Flatten()(logits)

    masked_logits = logits + (1.0 - mask_input) * (tf.float32.min)
    policy_output = layers.Softmax(name="policy")(masked_logits)

    value_output = layers.Dense(1, name="value")(game_state)

    model = keras.Model(
        inputs=[scalar_inputs, hand_input_1, hq_input_1, hq_input_2, 
                in_play_input_1_0, in_play_input_1_1, in_play_input_1_2, in_play_input_1_3, in_play_input_1_4,
                in_play_input_2_0, in_play_input_2_1, in_play_input_2_2, in_play_input_2_3, in_play_input_2_4,
                planets_input, action_type_input, action_args_input, mask_input, action_relevant_card_input, selected_card_hand_data_input],
        outputs={"policy": policy_output, "value": value_output}
    )
    return model


def create_model(simplified=True, transformer=False):
    if simplified:
        if transformer:
            return create_simplified_transformer_model()
        return create_simplified_model()
    # Code below does not work; CARD_FEATURE_DIM is different when embedding layers are used.
    # It has been left in as I would like to return to a model like this at some point.
    raise ValueError("The more complex model type is not supported. Please set simplified to True.")
    """
    hand_input_1 = keras.Input(shape=(MAX_HAND_SIZE,), name="hand_1")
    discard_input_1 = keras.Input(shape=(MAX_DISCARD_SIZE,), name="discard_1")
    hq_input_1 = keras.Input(shape=(MAX_CARDS_IN_ONE_PLAY_ZONE, CARD_FEATURE_DIM), name="hq_1")
    hq_input_ids_1 = keras.Input(shape=(MAX_CARDS_IN_ONE_PLAY_ZONE,), name="hq_ids_1")
    att_ids_hq_input_1 = keras.Input(shape=(MAX_CARDS_IN_ONE_PLAY_ZONE, MAX_ATTACHMENTS_PER_CARD), name="att_ids_hq_1")
    in_play_input_1_0 = keras.Input(shape=(MAX_CARDS_IN_ONE_PLAY_ZONE, CARD_FEATURE_DIM), name="in_play_1_0")
    in_play_input_ids_1_0 = keras.Input(shape=(MAX_CARDS_IN_ONE_PLAY_ZONE,), name="in_play_ids_1_0")
    att_ids_in_play_input_1_0 = keras.Input(shape=(MAX_CARDS_IN_ONE_PLAY_ZONE, MAX_ATTACHMENTS_PER_CARD), name="att_ids_in_play_1_0")
    in_play_input_1_1 = keras.Input(shape=(MAX_CARDS_IN_ONE_PLAY_ZONE, CARD_FEATURE_DIM), name="in_play_1_1")
    in_play_input_ids_1_1 = keras.Input(shape=(MAX_CARDS_IN_ONE_PLAY_ZONE,), name="in_play_ids_1_1")
    att_ids_in_play_input_1_1 = keras.Input(shape=(MAX_CARDS_IN_ONE_PLAY_ZONE, MAX_ATTACHMENTS_PER_CARD), name="att_ids_in_play_1_1")
    in_play_input_1_2 = keras.Input(shape=(MAX_CARDS_IN_ONE_PLAY_ZONE, CARD_FEATURE_DIM), name="in_play_1_2")
    in_play_input_ids_1_2 = keras.Input(shape=(MAX_CARDS_IN_ONE_PLAY_ZONE,), name="in_play_ids_1_2")
    att_ids_in_play_input_1_2 = keras.Input(shape=(MAX_CARDS_IN_ONE_PLAY_ZONE, MAX_ATTACHMENTS_PER_CARD), name="att_ids_in_play_1_2")
    in_play_input_1_3 = keras.Input(shape=(MAX_CARDS_IN_ONE_PLAY_ZONE, CARD_FEATURE_DIM), name="in_play_1_3")
    in_play_input_ids_1_3 = keras.Input(shape=(MAX_CARDS_IN_ONE_PLAY_ZONE,), name="in_play_ids_1_3")
    att_ids_in_play_input_1_3 = keras.Input(shape=(MAX_CARDS_IN_ONE_PLAY_ZONE, MAX_ATTACHMENTS_PER_CARD), name="att_ids_in_play_1_3")
    in_play_input_1_4 = keras.Input(shape=(MAX_CARDS_IN_ONE_PLAY_ZONE, CARD_FEATURE_DIM), name="in_play_1_4")
    in_play_input_ids_1_4 = keras.Input(shape=(MAX_CARDS_IN_ONE_PLAY_ZONE,), name="in_play_ids_1_4")
    att_ids_in_play_input_1_4 = keras.Input(shape=(MAX_CARDS_IN_ONE_PLAY_ZONE, MAX_ATTACHMENTS_PER_CARD), name="att_ids_in_play_1_4")

    discard_input_2 = keras.Input(shape=(MAX_DISCARD_SIZE,), name="discard_2")
    hq_input_2 = keras.Input(shape=(MAX_CARDS_IN_ONE_PLAY_ZONE, CARD_FEATURE_DIM), name="hq_2")
    hq_input_ids_2 = keras.Input(shape=(MAX_CARDS_IN_ONE_PLAY_ZONE,), name="hq_ids_2")
    att_ids_hq_input_2 = keras.Input(shape=(MAX_CARDS_IN_ONE_PLAY_ZONE, MAX_ATTACHMENTS_PER_CARD), name="att_ids_hq_2")
    in_play_input_2_0 = keras.Input(shape=(MAX_CARDS_IN_ONE_PLAY_ZONE, CARD_FEATURE_DIM), name="in_play_2_0")
    in_play_input_ids_2_0 = keras.Input(shape=(MAX_CARDS_IN_ONE_PLAY_ZONE,), name="in_play_ids_2_0")
    att_ids_in_play_input_2_0 = keras.Input(shape=(MAX_CARDS_IN_ONE_PLAY_ZONE, MAX_ATTACHMENTS_PER_CARD), name="att_ids_in_play_2_0")
    in_play_input_2_1 = keras.Input(shape=(MAX_CARDS_IN_ONE_PLAY_ZONE, CARD_FEATURE_DIM), name="in_play_2_1")
    in_play_input_ids_2_1 = keras.Input(shape=(MAX_CARDS_IN_ONE_PLAY_ZONE,), name="in_play_ids_2_1")
    att_ids_in_play_input_2_1 = keras.Input(shape=(MAX_CARDS_IN_ONE_PLAY_ZONE, MAX_ATTACHMENTS_PER_CARD), name="att_ids_in_play_2_1")
    in_play_input_2_2 = keras.Input(shape=(MAX_CARDS_IN_ONE_PLAY_ZONE, CARD_FEATURE_DIM), name="in_play_2_2")
    in_play_input_ids_2_2 = keras.Input(shape=(MAX_CARDS_IN_ONE_PLAY_ZONE,), name="in_play_ids_2_2")
    att_ids_in_play_input_2_2 = keras.Input(shape=(MAX_CARDS_IN_ONE_PLAY_ZONE, MAX_ATTACHMENTS_PER_CARD), name="att_ids_in_play_2_2")
    in_play_input_2_3 = keras.Input(shape=(MAX_CARDS_IN_ONE_PLAY_ZONE, CARD_FEATURE_DIM), name="in_play_2_3")
    in_play_input_ids_2_3 = keras.Input(shape=(MAX_CARDS_IN_ONE_PLAY_ZONE,), name="in_play_ids_2_3")
    att_ids_in_play_input_2_3 = keras.Input(shape=(MAX_CARDS_IN_ONE_PLAY_ZONE, MAX_ATTACHMENTS_PER_CARD), name="att_ids_in_play_2_3")
    in_play_input_2_4 = keras.Input(shape=(MAX_CARDS_IN_ONE_PLAY_ZONE, CARD_FEATURE_DIM), name="in_play_2_4")
    in_play_input_ids_2_4 = keras.Input(shape=(MAX_CARDS_IN_ONE_PLAY_ZONE,), name="in_play_ids_2_4")
    att_ids_in_play_input_2_4 = keras.Input(shape=(MAX_CARDS_IN_ONE_PLAY_ZONE, MAX_ATTACHMENTS_PER_CARD), name="att_ids_in_play_2_4")

    search_input = keras.Input(shape=(MAX_SEARCHED_CARDS,), name="search")
    planets_input = keras.Input(shape=(NUM_PLANETS,), name="planets")

    scalar_inputs = keras.Input(shape=(NUM_SCALARS,), name="scalars")
    action_type_input = keras.Input(shape=(MAX_ACTIONS,), name="action_types")
    action_args_input = keras.Input(shape=(MAX_ACTIONS, MAX_ACTION_ARGS), name="action_args")
    mask_input = keras.Input(shape=(MAX_ACTIONS,), name="action_mask")
    previous_actions_type_input = keras.Input(shape=(MAX_PREVIOUS_ACTIONS,), name="previous_action_types")
    previous_actions_args_input = keras.Input(shape=(MAX_PREVIOUS_ACTIONS, MAX_ACTION_ARGS), name="previous_action_args")

    hand_emb = layers.Embedding(HAND_VOCAB_SIZE, OUTPUT_DIM, name="hand_name_embedding")
    discard_emb = layers.Embedding(DISCARD_VOCAB_SIZE, OUTPUT_DIM, name="discard_name_embedding")
    in_play_emb = layers.Embedding(IN_PLAY_VOCAB_SIZE, OUTPUT_DIM, name="in_play_name_embedding")
    headquarters_emb = layers.Embedding(HEADQUARTERS_VOCAB_SIZE, OUTPUT_DIM, name="headquarters_name_embedding")
    planet_emb = layers.Embedding(PLANET_VOCAB_SIZE, OUTPUT_DIM, name="planet_name_embedding")
    att_emb = layers.Embedding(ATTACHMENT_VOCAB_SIZE, OUTPUT_DIM, name="attachment_name_embedding")

    def create_attachment_embedding_layer(attachment_input_layer):
        new_att_emb = att_emb(attachment_input_layer)
        return layers.Reshape((MAX_CARDS_IN_ONE_PLAY_ZONE, MAX_ATTACHMENTS_PER_CARD * OUTPUT_DIM))(new_att_emb)

    hand_1_emb = hand_emb(hand_input_1)
    hand_1_enc = transformer_block(hand_1_emb, HEAD_SIZE, NUM_HEADS, OUTPUT_DIM)
    hand_1_enc = layers.GlobalAveragePooling1D()(hand_1_enc)

    search_emb = hand_emb(search_input)
    search_enc = transformer_block(search_emb, HEAD_SIZE, NUM_HEADS, OUTPUT_DIM)
    search_enc = layers.GlobalAveragePooling1D()(search_enc)

    discard_1_emb = discard_emb(discard_input_1)
    discard_1_enc = transformer_block(discard_1_emb, HEAD_SIZE, NUM_HEADS, OUTPUT_DIM)
    discard_1_enc = layers.GlobalAveragePooling1D()(discard_1_enc)

    discard_2_emb = discard_emb(discard_input_2)
    discard_2_enc = transformer_block(discard_2_emb, HEAD_SIZE, NUM_HEADS, OUTPUT_DIM)
    discard_2_enc = layers.GlobalAveragePooling1D()(discard_2_enc)

    planet_enc = layers.GlobalAveragePooling1D()(planet_emb(planets_input))

    hq_1_projected = layers.Dense(OUTPUT_DIM)(layers.Concatenate()([hq_input_1, headquarters_emb(hq_input_ids_1), create_attachment_embedding_layer(att_ids_hq_input_1)]))
    hq_1_enc = transformer_block(hq_1_projected, HEAD_SIZE, NUM_HEADS, FF_DIM)
    hq_1_enc = layers.GlobalAveragePooling1D()(hq_1_enc)
    in_play_1_0_projected = layers.Dense(OUTPUT_DIM)(layers.Concatenate()([in_play_input_1_0, in_play_emb(in_play_input_ids_1_0), create_attachment_embedding_layer(att_ids_in_play_input_1_0)]))
    in_play_1_0_enc = transformer_block(in_play_1_0_projected, HEAD_SIZE, NUM_HEADS, FF_DIM)
    in_play_1_0_enc = layers.GlobalAveragePooling1D()(in_play_1_0_enc)
    in_play_1_1_projected = layers.Dense(OUTPUT_DIM)(layers.Concatenate()([in_play_input_1_1, in_play_emb(in_play_input_ids_1_1), create_attachment_embedding_layer(att_ids_in_play_input_1_1)]))
    in_play_1_1_enc = transformer_block(in_play_1_1_projected, HEAD_SIZE, NUM_HEADS, FF_DIM)
    in_play_1_1_enc = layers.GlobalAveragePooling1D()(in_play_1_1_enc)
    in_play_1_2_projected = layers.Dense(OUTPUT_DIM)(layers.Concatenate()([in_play_input_1_2, in_play_emb(in_play_input_ids_1_2), create_attachment_embedding_layer(att_ids_in_play_input_1_2)]))
    in_play_1_2_enc = transformer_block(in_play_1_2_projected, HEAD_SIZE, NUM_HEADS, FF_DIM)
    in_play_1_2_enc = layers.GlobalAveragePooling1D()(in_play_1_2_enc)
    in_play_1_3_projected = layers.Dense(OUTPUT_DIM)(layers.Concatenate()([in_play_input_1_3, in_play_emb(in_play_input_ids_1_3), create_attachment_embedding_layer(att_ids_in_play_input_1_3)]))
    in_play_1_3_enc = transformer_block(in_play_1_3_projected, HEAD_SIZE, NUM_HEADS, FF_DIM)
    in_play_1_3_enc = layers.GlobalAveragePooling1D()(in_play_1_3_enc)
    in_play_1_4_projected = layers.Dense(OUTPUT_DIM)(layers.Concatenate()([in_play_input_1_4, in_play_emb(in_play_input_ids_1_4), create_attachment_embedding_layer(att_ids_in_play_input_1_4)]))
    in_play_1_4_enc = transformer_block(in_play_1_4_projected, HEAD_SIZE, NUM_HEADS, FF_DIM)
    in_play_1_4_enc = layers.GlobalAveragePooling1D()(in_play_1_4_enc)

    hq_2_projected = layers.Dense(OUTPUT_DIM)(layers.Concatenate()([hq_input_2, headquarters_emb(hq_input_ids_2), create_attachment_embedding_layer(att_ids_hq_input_2)]))
    hq_2_enc = transformer_block(hq_2_projected, HEAD_SIZE, NUM_HEADS, FF_DIM)
    hq_2_enc = layers.GlobalAveragePooling1D()(hq_2_enc)
    in_play_2_0_projected = layers.Dense(OUTPUT_DIM)(layers.Concatenate()([in_play_input_2_0, in_play_emb(in_play_input_ids_2_0), create_attachment_embedding_layer(att_ids_in_play_input_2_0)]))
    in_play_2_0_enc = transformer_block(in_play_2_0_projected, HEAD_SIZE, NUM_HEADS, FF_DIM)
    in_play_2_0_enc = layers.GlobalAveragePooling1D()(in_play_2_0_enc)
    in_play_2_1_projected = layers.Dense(OUTPUT_DIM)(layers.Concatenate()([in_play_input_2_1, in_play_emb(in_play_input_ids_2_1), create_attachment_embedding_layer(att_ids_in_play_input_2_1)]))
    in_play_2_1_enc = transformer_block(in_play_2_1_projected, HEAD_SIZE, NUM_HEADS, FF_DIM)
    in_play_2_1_enc = layers.GlobalAveragePooling1D()(in_play_2_1_enc)
    in_play_2_2_projected = layers.Dense(OUTPUT_DIM)(layers.Concatenate()([in_play_input_2_2, in_play_emb(in_play_input_ids_2_2), create_attachment_embedding_layer(att_ids_in_play_input_2_2)]))
    in_play_2_2_enc = transformer_block(in_play_2_2_projected, HEAD_SIZE, NUM_HEADS, FF_DIM)
    in_play_2_2_enc = layers.GlobalAveragePooling1D()(in_play_2_2_enc)
    in_play_2_3_projected = layers.Dense(OUTPUT_DIM)(layers.Concatenate()([in_play_input_2_3, in_play_emb(in_play_input_ids_2_3), create_attachment_embedding_layer(att_ids_in_play_input_2_3)]))
    in_play_2_3_enc = transformer_block(in_play_2_3_projected, HEAD_SIZE, NUM_HEADS, FF_DIM)
    in_play_2_3_enc = layers.GlobalAveragePooling1D()(in_play_2_3_enc)
    in_play_2_4_projected = layers.Dense(OUTPUT_DIM)(layers.Concatenate()([in_play_input_2_4, in_play_emb(in_play_input_ids_2_4), create_attachment_embedding_layer(att_ids_in_play_input_2_4)]))
    in_play_2_4_enc = transformer_block(in_play_2_4_projected, HEAD_SIZE, NUM_HEADS, FF_DIM)
    in_play_2_4_enc = layers.GlobalAveragePooling1D()(in_play_2_4_enc)

    scalar_enc = layers.Dense(8, activation="relu")(scalar_inputs)

    previous_action_type_emb = layers.Embedding(NUM_ACTION_TYPES, 32, name="previous_action_type_embedding")(previous_actions_type_input)
    previous_action_enc = layers.Concatenate()([previous_action_type_emb, previous_actions_args_input])

    previous_action_enc = layers.Dense(OUTPUT_DIM)(previous_action_enc)
    previous_action_enc = transformer_block(previous_action_enc, HEAD_SIZE, NUM_HEADS, FF_DIM)
    previous_action_enc = layers.GlobalAveragePooling1D()(previous_action_enc)

    game_state = layers.Concatenate(axis=-1)(
        [
            hand_1_enc, discard_1_enc, discard_2_enc, hq_1_enc, hq_2_enc,
            in_play_1_0_enc, in_play_1_1_enc, in_play_1_2_enc, in_play_1_3_enc, in_play_1_4_enc,
            in_play_2_0_enc, in_play_2_1_enc, in_play_2_2_enc, in_play_2_3_enc, in_play_2_4_enc,
            planet_enc, search_enc, scalar_enc, previous_action_enc
        ]
    )
    game_state = layers.Dense(64, activation="relu")(game_state)
    game_state = layers.Dense(64, activation="relu")(game_state)

    type_emb = layers.Embedding(NUM_ACTION_TYPES, 32)(action_type_input)
    action_enc = layers.Concatenate()([type_emb, action_args_input])
    action_enc = layers.Dense(OUTPUT_DIM)(action_enc)
    action_enc = transformer_block(action_enc, HEAD_SIZE, NUM_HEADS, FF_DIM)

    game_context = layers.RepeatVector(MAX_ACTIONS)(game_state)
    game_context = layers.Dense(OUTPUT_DIM)(game_context)

    logits = layers.TimeDistributed(layers.Dense(1))(action_enc)
    logits = layers.Flatten()(logits)

    masked_logits = logits + (1.0 - mask_input) * (tf.float32.min)
    policy_output = layers.Softmax(name="policy")(masked_logits)

    value_output = layers.Dense(64, activation="relu")(game_state)
    value_output = layers.Dense(1, name="value")(value_output)

    model = keras.Model(
        inputs=[scalar_inputs, hand_input_1, hq_input_1, discard_input_1, discard_input_2, hq_input_2,
                hq_input_ids_1, hq_input_ids_2,
                in_play_input_1_0, in_play_input_1_1, in_play_input_1_2, in_play_input_1_3, in_play_input_1_4,
                in_play_input_2_0, in_play_input_2_1, in_play_input_2_2, in_play_input_2_3, in_play_input_2_4,
                in_play_input_ids_1_0, in_play_input_ids_1_1, in_play_input_ids_1_2,
                in_play_input_ids_1_3, in_play_input_ids_1_4,
                in_play_input_ids_2_0, in_play_input_ids_2_1, in_play_input_ids_2_2,
                in_play_input_ids_2_3, in_play_input_ids_2_4,
                planets_input, search_input, action_type_input, action_args_input, mask_input, 
                previous_actions_type_input, previous_actions_args_input,
                att_ids_hq_input_1, att_ids_hq_input_2,
                att_ids_in_play_input_1_0, att_ids_in_play_input_1_1, att_ids_in_play_input_1_2,
                att_ids_in_play_input_1_3, att_ids_in_play_input_1_4,
                att_ids_in_play_input_2_0, att_ids_in_play_input_2_1, att_ids_in_play_input_2_2,
                att_ids_in_play_input_2_3, att_ids_in_play_input_2_4],
        outputs={"policy": policy_output, "value": value_output}
    )
    return model
    """


def extract_data_from_game(game):
    active_player_name = game.active_player
    possible_actions = game.active_options
    production = game.production
    active_player_number = game.get_active_player_number() - 1
    if active_player_number < 0:
        active_player_number = 0
    player = game.determine_player(active_player_name)
    enemy_player = game.determine_enemy_player(active_player_name)
    previous_actions = player.get_previous_actions()

    p1_resources, p2_resources, p1_target_hand, p1_target_discard, p2_target_discard, initiative, \
        planet_aiming, special_combat_action_window, damage_that_can_be_shielded, choice_str, \
        round_number, phase, mode, p1_red, p1_blue, p1_green, p2_red, p2_blue, p2_green, \
        p1_bloodied, p2_bloodied, p2_num_cards = game.get_scalars_perspective_player(player)

    p1_hand = player.get_hand()

    p1_discard = player.get_discard()
    p2_discard = enemy_player.get_discard()

    searched_cards = game.get_searched_cards()

    hq_1 = player.get_headquarters()
    in_play_1_0 = player.get_cards_at_planet(0, round_number=round_number-1)
    in_play_1_1 = player.get_cards_at_planet(1, round_number=round_number-1)
    in_play_1_2 = player.get_cards_at_planet(2, round_number=round_number-1)
    in_play_1_3 = player.get_cards_at_planet(3, round_number=round_number-1)
    in_play_1_4 = player.get_cards_at_planet(4, round_number=round_number-1)
    hq_2 = enemy_player.get_headquarters()
    in_play_2_0 = enemy_player.get_cards_at_planet(0, round_number=round_number-1)
    in_play_2_1 = enemy_player.get_cards_at_planet(1, round_number=round_number-1)
    in_play_2_2 = enemy_player.get_cards_at_planet(2, round_number=round_number-1)
    in_play_2_3 = enemy_player.get_cards_at_planet(3, round_number=round_number-1)
    in_play_2_4 = enemy_player.get_cards_at_planet(4, round_number=round_number-1)

    planet_names = game.get_planets()

    hand_enc_1, _, _ = encode_attributes.encode_card_list_and_attachment_ids(p1_hand, hand_vocab, attachment_vocab, production=production, MAX_SIZE=MAX_HAND_SIZE)
    selected_card_hand = player.get_selected_card_hand()
    selected_card_hand_enc = np.stack([encode_attributes.encode_card(selected_card_hand)])
    discard_enc_1 = encode_attributes.encode_list_of_card_names(p1_discard, discard_vocab, MAX_DISCARD_SIZE, production=production)
    discard_enc_2 = encode_attributes.encode_list_of_card_names(p2_discard, discard_vocab, MAX_DISCARD_SIZE, production=production)
    search_enc = encode_attributes.encode_list_of_card_names(searched_cards, hand_vocab, MAX_SEARCHED_CARDS, production=production)
    hq_1_enc, hq_ids_1_enc, att_hq_1_enc = encode_attributes.encode_card_list_and_attachment_ids(hq_1, headquarters_vocab, attachment_vocab, production=production)
    in_play_1_0_enc, in_play_ids_1_0_enc, att_in_play_1_0_enc = encode_attributes.encode_card_list_and_attachment_ids(in_play_1_0, in_play_vocab, attachment_vocab, production=production)
    in_play_1_1_enc, in_play_ids_1_1_enc, att_in_play_1_1_enc = encode_attributes.encode_card_list_and_attachment_ids(in_play_1_1, in_play_vocab, attachment_vocab, production=production)
    in_play_1_2_enc, in_play_ids_1_2_enc, att_in_play_1_2_enc = encode_attributes.encode_card_list_and_attachment_ids(in_play_1_2, in_play_vocab, attachment_vocab, production=production)
    in_play_1_3_enc, in_play_ids_1_3_enc, att_in_play_1_3_enc = encode_attributes.encode_card_list_and_attachment_ids(in_play_1_3, in_play_vocab, attachment_vocab, production=production)
    in_play_1_4_enc, in_play_ids_1_4_enc, att_in_play_1_4_enc = encode_attributes.encode_card_list_and_attachment_ids(in_play_1_4, in_play_vocab, attachment_vocab, production=production)
    hq_2_enc, hq_ids_2_enc, att_hq_2_enc = encode_attributes.encode_card_list_and_attachment_ids(hq_2, headquarters_vocab, attachment_vocab, production=production)
    in_play_2_0_enc, in_play_ids_2_0_enc, att_in_play_2_0_enc = encode_attributes.encode_card_list_and_attachment_ids(in_play_2_0, in_play_vocab, attachment_vocab, production=production)
    in_play_2_1_enc, in_play_ids_2_1_enc, att_in_play_2_1_enc = encode_attributes.encode_card_list_and_attachment_ids(in_play_2_1, in_play_vocab, attachment_vocab, production=production)
    in_play_2_2_enc, in_play_ids_2_2_enc, att_in_play_2_2_enc = encode_attributes.encode_card_list_and_attachment_ids(in_play_2_2, in_play_vocab, attachment_vocab, production=production)
    in_play_2_3_enc, in_play_ids_2_3_enc, att_in_play_2_3_enc = encode_attributes.encode_card_list_and_attachment_ids(in_play_2_3, in_play_vocab, attachment_vocab, production=production)
    in_play_2_4_enc, in_play_ids_2_4_enc, att_in_play_2_4_enc = encode_attributes.encode_card_list_and_attachment_ids(in_play_2_4, in_play_vocab, attachment_vocab, production=production)

    planets_enc = encode_attributes.encode_list_of_card_names(planet_names, planet_vocab, NUM_PLANETS, production=production)

    scalar_enc = encode_attributes.encode_scalars(
        initiative, p1_resources, p2_resources,
        planet_aiming, special_combat_action_window, damage_that_can_be_shielded, choice_str,
        round_number, phase, mode, p1_red, p1_blue, p1_green, p2_red, p2_blue, p2_green, p1_bloodied, p2_bloodied,
        p2_num_cards
    )
    action_type_ids, action_args, action_mask, rel_card_data = encode_attributes.encode_action_list(game, possible_actions, player)
    # previous_actions_types, previous_actions_args = encode_attributes.encode_previous_actions(previous_actions)

    inputs = {
        "hand_1": hand_enc_1[np.newaxis],
        # "discard_1": discard_enc_1[np.newaxis],
        # "discard_2": discard_enc_2[np.newaxis],
        # "search": search_enc[np.newaxis],
        "hand_selected_card": selected_card_hand_enc[np.newaxis],
        "hq_1": hq_1_enc[np.newaxis],
        # "hq_ids_1": hq_ids_1_enc[np.newaxis],
        # "att_ids_hq_1": att_hq_1_enc[np.newaxis],
        "hq_2": hq_2_enc[np.newaxis],
        # "hq_ids_2": hq_ids_2_enc[np.newaxis],
        # "att_ids_hq_2": att_hq_2_enc[np.newaxis],
        "in_play_1_0": in_play_1_0_enc[np.newaxis],
        "in_play_1_1": in_play_1_1_enc[np.newaxis],
        "in_play_1_2": in_play_1_2_enc[np.newaxis],
        "in_play_1_3": in_play_1_3_enc[np.newaxis],
        "in_play_1_4": in_play_1_4_enc[np.newaxis],
        # "in_play_ids_1_0": in_play_ids_1_0_enc[np.newaxis],
        # "in_play_ids_1_1": in_play_ids_1_1_enc[np.newaxis],
        # "in_play_ids_1_2": in_play_ids_1_2_enc[np.newaxis],
        # "in_play_ids_1_3": in_play_ids_1_3_enc[np.newaxis],
        # "in_play_ids_1_4": in_play_ids_1_4_enc[np.newaxis],
        # "att_ids_in_play_1_0": att_in_play_1_0_enc[np.newaxis],
        # "att_ids_in_play_1_1": att_in_play_1_1_enc[np.newaxis],
        # "att_ids_in_play_1_2": att_in_play_1_2_enc[np.newaxis],
        # "att_ids_in_play_1_3": att_in_play_1_3_enc[np.newaxis],
        # "att_ids_in_play_1_4": att_in_play_1_4_enc[np.newaxis],
        "in_play_2_0": in_play_2_0_enc[np.newaxis],
        "in_play_2_1": in_play_2_1_enc[np.newaxis],
        "in_play_2_2": in_play_2_2_enc[np.newaxis],
        "in_play_2_3": in_play_2_3_enc[np.newaxis],
        "in_play_2_4": in_play_2_4_enc[np.newaxis],
        # "in_play_ids_2_0": in_play_ids_2_0_enc[np.newaxis],
        # "in_play_ids_2_1": in_play_ids_2_1_enc[np.newaxis],
        # "in_play_ids_2_2": in_play_ids_2_2_enc[np.newaxis],
        # "in_play_ids_2_3": in_play_ids_2_3_enc[np.newaxis],
        # "in_play_ids_2_4": in_play_ids_2_4_enc[np.newaxis],
        # "att_ids_in_play_2_0": att_in_play_2_0_enc[np.newaxis],
        # "att_ids_in_play_2_1": att_in_play_2_1_enc[np.newaxis],
        # "att_ids_in_play_2_2": att_in_play_2_2_enc[np.newaxis],
        # "att_ids_in_play_2_3": att_in_play_2_3_enc[np.newaxis],
        # "att_ids_in_play_2_4": att_in_play_2_4_enc[np.newaxis],
        "planets": planets_enc[np.newaxis],
        "scalars": scalar_enc,
        "action_types": action_type_ids[np.newaxis],
        "action_args": action_args[np.newaxis],
        "action_mask": action_mask[np.newaxis],
        "action_card_input": rel_card_data[np.newaxis]
    }
    return inputs


def determine_reward_of_move_just_made(game: Game, move_just_made_string):
    player = game.get_active_player()
    enemy_player = game.get_inactive_player()
    split_string = move_just_made_string.split(sep="/")
    if game.active_context == "Search":
        if len(split_string) == 2:
            return 1
        if len(split_string) == 1:
            return -1
    if game.active_context == "Deploy Turn":
        card_just_deployed = player.get_selected_card_hand()
        if card_just_deployed is not None:
            if len(split_string) == 2:
                if split_string[0] == "PLANETS":
                    planet_pos = int(split_string[1])
                    command_card = card_just_deployed.get_command()
                    attack_card = card_just_deployed.get_attack()
                    cost_card = card_just_deployed.get_cost()
                    is_combat_unit = card_just_deployed.check_if_combat_unit()
                    is_command_unit = card_just_deployed.check_if_command_unit()
                    first_planet = game.check_if_first_planet(planet_pos)
                    if is_command_unit:
                        if first_planet and attack_card == 0:
                            return -5  # TODO: Reward based on enemy unit presence to encourage pressuring potential fp victories with void pirates and rogue traders
                        own_command = player.count_command_at_planet(planet_pos)
                        enemy_command = enemy_player.count_command_at_planet(planet_pos)
                        if own_command > enemy_command:
                            if not first_planet:
                                return -1
                        elif own_command == enemy_command:
                            return 1
                        elif command_card > enemy_command - own_command:
                            return 5  # very heavy reward for deploying multi command units to swing command
                    elif is_combat_unit:
                        if first_planet:
                            return 5
            elif len(split_string) == 3:
                if split_string[0] == "HQ":
                    if card_just_deployed.get_card_name() == "Promotion":
                        return -1
                    if player.get_number() == int(split_string[1]):
                        return 1
                    return -1
            elif len(split_string) == 4:
                if split_string[0] == "IN_PLAY":
                    if int(split_string[1]) != player.get_number():
                        return -1
                    card_just_deployed = player.get_selected_card_hand()
                    if card_just_deployed is not None:
                        if card_just_deployed.get_card_name() == "Promotion":
                            planet_pos = int(split_string[2])
                            command_card = 2
                            own_command = player.count_command_at_planet(planet_pos)
                            enemy_command = enemy_player.count_command_at_planet(planet_pos)
                            if own_command > enemy_command:
                                return -1
                            elif own_command == enemy_command:
                                return 1
                            elif command_card > enemy_command - own_command:
                                return 5  # very heavy reward for deploying multi command units to swing command
        else:
            if len(split_string) == 1:
                if split_string[0] == "pass-P1":
                    if player.search_hand_low_cost_command_unit():
                        return -1
            if len(split_string) == 3:
                if split_string[0] == "HAND":
                    card = player.get_card_in_hand(int(split_string[2]))
                    if card is not None:
                        if card.check_if_command_unit():
                            return 1
    elif game.active_context == "Combat Turn":
        if len(split_string) == 4:
            if split_string[0] == "IN_PLAY":
                if int(split_string[1]) == player.get_number():
                    reward_calc = 0
                    planet_pos = int(split_string[2])
                    unit_pos = int(split_string[3])
                    card = player.get_card_given_pos(planet_pos, unit_pos)
                    if card is not None:
                        if card.get_attack() == 0:
                            return -1
                        reward_calc = card.get_attack() - 1
                    return reward_calc
        elif len(split_string) == 2:
            return 5
    elif game.active_context == "Retreat Turn":
        icons_red_enemy, icons_blue_enemy, icons_green_enemy = enemy_player.count_icons_victory_display()
        icon_red_first, icon_blue_first, icon_green_first = game.get_colors_first_planet()
        icons_red_pot_enemy = icon_red_first + icons_red_enemy
        icons_blue_pot_enemy = icon_blue_first + icons_blue_enemy
        icons_green_pot_enemy = icon_green_first + icons_green_enemy
        if game.targeted_planet == game.round_number - 1:
            if icons_red_pot_enemy == 3 or icons_blue_pot_enemy == 3 or icons_green_pot_enemy == 3:
                return -5
            if player.count_attack_at_planet(game.targeted_planet) > enemy_player.count_attack_at_planet(game.targeted_planet):
                return -1
        elif game.targeted_planet == game.round_number:
            if player.count_attack_at_planet(game.targeted_planet) > enemy_player.count_attack_at_planet(game.targeted_planet):
                return -1
        else:
            if len(split_string) == 4:
                planet_pos = int(split_string[2])
                unit_pos = int(split_string[3])
                if player.get_card_given_pos(planet_pos, unit_pos).get_card_type() == "Warlord":
                    return -5
    elif game.active_context == "Damage":
        if len(split_string) > 0:
            if split_string[0] == "IN_PLAY" or split_string[0] == "HQ":
                return 1
            if split_string[0] == "ATTACHMENT":
                return game.get_damage_that_can_be_shielded() - 1
            if split_string[0] == "HAND":
                card = player.get_card_in_hand(int(split_string[2]))
                if card is not None:
                    if game.get_damage_that_can_be_shielded() == card.get_shields():
                        return 1
    return 0


def model_make_choice(model, game, training=True, record_action=True, choose_random=False, filter_moves=False):
    if filter_moves:
        moveFilter.filter_obvious_bad_moves(game)
    possible_actions = game.get_active_options()
    p_num = game.get_active_player_number()
    if not possible_actions:
        game.active_options = ["pass-P1"]
        possible_actions = ["pass-P1"]
    if len(possible_actions) == 1 or choose_random:
        chosen_action = possible_actions[0]
        if choose_random:
            chosen_action = random.choice(possible_actions)
        if record_action:
            if p_num == 1:
                if len(game.player_one.previous_actions) > MAX_PREVIOUS_ACTIONS:
                    del game.player_one.previous_actions[0]
                game.player_one.previous_actions.append(chosen_action)
            else:
                if len(game.player_two.previous_actions) > MAX_PREVIOUS_ACTIONS:
                    del game.player_two.previous_actions[0]
                game.player_two.previous_actions.append(chosen_action)
        return {"choice": chosen_action, "action_idx": 0, "player_choosing": p_num, "flag": "Single"}
    inputs = extract_data_from_game(game)
    returned_values = model(inputs)
    policy_probs = returned_values["policy"][0].numpy()
    value = returned_values["value"][0][0].numpy()
    if not training:
        print(policy_probs)
    if training:
        chosen_action_idx = np.random.choice(len(policy_probs), p=policy_probs)
    else:
        chosen_action_idx = np.argmax(policy_probs)
    # print(chosen_action_idx)
    chosen_action = possible_actions[chosen_action_idx]
    if record_action:
        if p_num == 1:
            if len(game.player_one.previous_actions) > MAX_PREVIOUS_ACTIONS:
                del game.player_one.previous_actions[0]
            game.player_one.previous_actions.append(chosen_action)
        else:
            if len(game.player_two.previous_actions) > MAX_PREVIOUS_ACTIONS:
                del game.player_two.previous_actions[0]
            game.player_two.previous_actions.append(chosen_action)
    return {"choice": chosen_action, "action_idx": chosen_action_idx, "policy_probs": policy_probs,
            "player_choosing": p_num, "critic_value": value, "state": inputs, "flag": "Multi"}


def transform_rewards(rewards):
    discounted = np.array(rewards, dtype=np.float32)
    return discounted


def compute_loss(model, inputs, act_idxs, rewards):
    batched_inputs = {
        key: tf.concat([input[key] for input in inputs], axis=0)
        for key in inputs[0]  # inputs is used to get the key names
    }

    outputs = model(batched_inputs, training=True)
    policy_probs = outputs["policy"]
    value_estimates = tf.squeeze(outputs["value"], axis=1)

    print("REWARDS:")
    print(rewards)

    Gt = tf.cast(rewards, dtype=tf.float32)

    Gt_normalised = (Gt - tf.reduce_mean(Gt)) / (tf.math.reduce_std(Gt) + 1e-8)

    advantages = Gt_normalised - value_estimates

    action_indices = tf.constant([action_idx for action_idx in act_idxs])
    action_indices_2d = tf.stack(
        [tf.range(tf.shape(action_indices)[0]), action_indices], axis=1
    )
    taken_probs = tf.gather_nd(policy_probs, action_indices_2d)

    actor_loss = -tf.reduce_mean(tf.math.log(taken_probs + 1e-8) * advantages)
    critic_loss = tf.reduce_mean(tf.square(advantages))
    print("ACTOR_LOSS:", actor_loss)
    print("CRITIC_LOSS:", critic_loss)
    total_loss = actor_loss + 0.5 * critic_loss
    print("TOTAL_LOSS:", total_loss)
    return total_loss


def resolve_training(model, game, optimizer, inputs_per_player, actions_idx_per_player, 
                     rewards_per_player, rewards_stay_fixed, game_completed=True):
    if game_completed:
        for player in [1, 2]:
            winner = game.get_winner_number()
            if winner == -1:
                final_reward = 0
            elif winner == player:
                final_reward = 1
            else:
                final_reward = -1
            for i in range(len(rewards_per_player[player])):
                if not rewards_stay_fixed[player][i]:
                    rewards_per_player[player][i] = final_reward
    else:
        estimated_reward_of_round = game.estimate_reward_end_of_round(1)
        for i in range(len(rewards_per_player[1])):
            if not rewards_stay_fixed[1][i]:
                rewards_per_player[1][i] = estimated_reward_of_round
        estimated_reward_of_round = -estimated_reward_of_round
        for i in range(len(rewards_per_player[2])):
            if not rewards_stay_fixed[2][i]:
                rewards_per_player[2][i] = estimated_reward_of_round

    with tf.GradientTape() as tape:
        total_loss = 0
        print("Computing loss")
        start_time = datetime.datetime.now()
        for player in [1, 2]:
            if not inputs_per_player[player]:
                continue
            reward_slice_position = game.last_round_id_nums[player - 1]
            if game_completed:
                reward_slice_position = 0
            rewards = transform_rewards(rewards_per_player[player][reward_slice_position:])
            total_loss += compute_loss(
                model, inputs_per_player[player][reward_slice_position:], 
                actions_idx_per_player[player][reward_slice_position:], rewards
            )
            game.last_round_id_nums[player - 1] = len(rewards_per_player[player])
        finished_loss = datetime.datetime.now()
        print(finished_loss - start_time)
        print("Applying gradient")
        grads = tape.gradient(total_loss, model.trainable_variables)
        grads, _ = tf.clip_by_global_norm(grads, clip_norm=1)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        print("finished gradient")
        finished_grad = datetime.datetime.now()
        print(finished_grad - finished_loss)
