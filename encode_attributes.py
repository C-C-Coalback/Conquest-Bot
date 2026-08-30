import numpy as np
from maxValues import MAX_DAMAGE, MAX_CARDS_IN_ONE_PLAY_ZONE, MAX_RESOURCES, CARD_FEATURE_DIM, \
    MAX_ACTIONS, MAX_ACTION_ARGS, ACTION_TYPES, MAX_ATTACHMENTS_PER_CARD, TARGET_TYPES, \
    MAX_HAND_SIZE, UNIQUE_CHOICES, NUM_SPECIAL_CHOICES, PHASES, NUM_PHASES, MODES, \
    NUM_MODES, CARD_TYPES, NUM_CARD_TYPES, MAX_COMMAND, MAX_ATTACK, MAX_HEALTH, MAX_ONE_ICON, NUM_ROUNDS, \
    MAX_ARG_VALUE, FACTIONS, NUM_FACTIONS, MAX_COST, SCALARS_PER_CARD
from unknown_card_name import UNKNOWN_CARD_NAME


def filter_card_not_in_vocab(card_name, vocab):
    if card_name not in vocab:
        return UNKNOWN_CARD_NAME
    return card_name


def encode_integer(value, max_value):
    if value > max_value:
        value = max_value
    return np.array([value / max_value])


def encode_target(target_id):
    return encode_integer(target_id, TARGET_TYPES)


def encode_card_type(card_type):
    return encode_integer(CARD_TYPES[card_type], NUM_CARD_TYPES)


def encode_faction(faction):
    return encode_integer(FACTIONS[faction], NUM_FACTIONS)


def encode_resources(resources):
    return encode_integer(resources, MAX_RESOURCES)


def encode_scalars(initiative, resources_1, resources_2, planet_aiming, is_special_combat_action_window,
                   damage_that_can_be_blocked, choice_str, round_number, phase, mode, p1_red, p1_blue, p1_green,
                   p2_red, p2_blue, p2_green, p1_bloodied, p2_bloodied, p2_num_cards):
    return np.array(
        [[resources_1 / MAX_RESOURCES, resources_2 / MAX_RESOURCES,
          planet_aiming / NUM_ROUNDS, float(initiative), float(is_special_combat_action_window),
          damage_that_can_be_blocked / MAX_DAMAGE, encode_choice(choice_str),
          round_number / NUM_ROUNDS, encode_phase(phase), encode_mode(mode),
          p1_red / MAX_ONE_ICON, p1_blue / MAX_ONE_ICON, p1_green / MAX_ONE_ICON,
          p2_red / MAX_ONE_ICON, p2_blue / MAX_ONE_ICON, p2_green / MAX_ONE_ICON,
          float(p1_bloodied), float(p2_bloodied), p2_num_cards / MAX_HAND_SIZE]])


def encode_mode(mode):
    if mode not in MODES:
        return MODES["UNKNOWN"] / NUM_MODES
    return MODES[mode] / NUM_MODES


def encode_phase(phase):
    if phase not in PHASES:
        return 0
    return PHASES[phase] / NUM_PHASES


def encode_choice(choice_str):
    if choice_str not in UNIQUE_CHOICES:
        choice_str = "Unknown Choice"
    return UNIQUE_CHOICES[choice_str] / NUM_SPECIAL_CHOICES


def encode_boolean(value):
    return np.array([float(value)])


def encode_list_of_card_names(list_strings, vocab, max_len, production=False):
    if production:
        list_strings = [filter_card_not_in_vocab(s, vocab) for s in list_strings]
    ids = [vocab[s] for s in list_strings]
    padded_ids = ids + [0] * (max_len - len(ids))
    return np.array(padded_ids[:max_len])


def encode_attachment_id(attachment, attachment_vocab, production=False):
    card_name = attachment.get_card_name()
    if production:
        card_name = filter_card_not_in_vocab(card_name, attachment_vocab)
    return attachment_vocab[card_name]


def encode_card_id(card, card_name_vocab, production=False):
    card_name = card.get_card_name()
    if production:
        card_name = filter_card_not_in_vocab(card_name, card_name_vocab)
    return card_name_vocab[card_name]


def encode_dummy_card():
    simple_data = np.zeros(SCALARS_PER_CARD)
    padding = [0] * (MAX_ATTACHMENTS_PER_CARD)
    final_attachment_enc = np.array(padding)
    return np.concatenate(
        [
            simple_data, final_attachment_enc
        ]
    )


def encode_card(card):
    if card is None:
        return encode_dummy_card()
    ready_enc = encode_boolean(card.get_ready())
    damage_enc = encode_integer(card.get_damage(), MAX_DAMAGE)
    target_enc = encode_target(card.get_target())
    attack_enc = encode_integer(card.get_attack(), MAX_ATTACK)
    health_enc = encode_integer(card.get_health(), MAX_HEALTH)
    command_enc = encode_integer(card.get_command(), MAX_COMMAND)
    cost_enc = encode_integer(card.get_cost(), MAX_COST)
    type_enc = encode_card_type(card.get_card_type())
    faction_enc = encode_faction(card.get_faction())
    attachment_ready_list = [float(a.get_ready()) for a in card.get_attachments()[:MAX_ATTACHMENTS_PER_CARD]]
    padding = [0] * (MAX_ATTACHMENTS_PER_CARD - len(attachment_ready_list))
    final_attachment_enc = np.array(attachment_ready_list + padding)
    return np.concatenate(
        [
            ready_enc, damage_enc, target_enc, attack_enc, health_enc, command_enc, cost_enc, type_enc, faction_enc, final_attachment_enc
        ]
    )


def encode_card_list_and_attachment_ids(card_list, vocab, attachment_vocab, production=False, MAX_SIZE=MAX_CARDS_IN_ONE_PLAY_ZONE):
    card_encs = []
    card_ids = []
    attachment_ids = []

    for card in card_list[:MAX_SIZE]:
        card_ids.append(encode_card_id(card, vocab, production=production))
        card_encs.append(encode_card(card))

        ids = [encode_attachment_id(a, attachment_vocab, production=production) for a in card.get_attachments()[:MAX_ATTACHMENTS_PER_CARD]]
        padding = [0] * (MAX_ATTACHMENTS_PER_CARD - len(ids))
        attachment_ids.append(ids + padding)

    pad_count = MAX_SIZE - len(card_encs)
    padding = [0] * pad_count
    card_ids = card_ids + padding
    card_encs += [np.zeros(CARD_FEATURE_DIM)] * pad_count
    attachment_ids += [[0] * MAX_ATTACHMENTS_PER_CARD] * pad_count

    return (
        np.stack(card_encs),
        np.array(card_ids, dtype=np.int32),
        np.array(attachment_ids, dtype=np.int32)
    )


def encode_action(game, action_str, player):
    split_string = action_str.split("/")
    action_type = split_string[0]
    action_type_replaced = action_type.replace("SPECIAL_ACTION_", "")
    relevant_card_data = encode_dummy_card()
    enemy_card = False
    if action_type_replaced in ["IN_PLAY", "HQ", "HAND"]:
        if len(split_string) > 1:
            if player.number != int(split_string[1]):
                enemy_card = True
            if action_type_replaced == "IN_PLAY" and len(split_string) > 3:
                relevant_card_data = encode_card(player.get_card_given_pos(int(split_string[2]), int(split_string[3])))
            elif action_type_replaced == "HQ" and len(split_string) > 2:
                relevant_card_data = encode_card(player.get_card_given_pos(-2, int(split_string[2])))
            elif action_type_replaced == "HAND" and len(split_string) > 2:
                relevant_card_data = encode_card(player.get_card_in_hand(int(split_string[2])))
    elif action_type_replaced == "ATTACHMENT":
        if len(split_string) > 2:
            if split_string[2] == str(player.number):
                split_string[2] = str(MAX_ARG_VALUE)
            else:
                split_string[2] = "0"
    args = [0]
    if action_type in ["PLANETS", "CHOICE"] and len(split_string) > 1:
        args = [int(split_string[1]) / MAX_ARG_VALUE]
        if game.active_context == "Deploy Turn" or game.active_context == "Commitment":
            active_player = game.get_active_player()
            enemy_player = game.get_inactive_player()
            selected_card = active_player.get_selected_card_hand()
            if game.active_context == "Commitment":
                planet_pos = int(split_string[1])
                own_command = active_player.count_command_at_planet(planet_pos)
                enemy_command = enemy_player.count_command_at_planet(planet_pos)
                total_command = own_command - enemy_command
                total_command = total_command / MAX_COMMAND
                if total_command > 1:
                    total_command = 1
                if total_command < -1:
                    total_command = -1
                args = args + [total_command]
            elif selected_card is not None:
                if selected_card.check_if_command_unit():
                    planet_pos = int(split_string[1])
                    own_command = active_player.count_command_at_planet(planet_pos)
                    enemy_command = enemy_player.count_command_at_planet(planet_pos)
                    total_command = own_command - enemy_command
                    total_command = total_command / MAX_COMMAND
                    if total_command > 1:
                        total_command = 1
                    if total_command < -1:
                        total_command = -1
                    args = args + [total_command]
                else:
                    planet_pos = int(split_string[1])
                    if planet_pos == game.round_number - 1:
                        args = args + [0.5]
                    elif planet_pos == game.round_number:
                        args = args + [0]
                    else:
                        args = args + [1]
    while len(args) < 2:
        args = args + [0]
    if enemy_card:
        args = args + [1]
    if action_type_replaced == "ATTACHMENT":
        action_type = action_type + "_" + split_string[1]
    type_id = ACTION_TYPES[action_type]
    padded_args = args + [0] * (MAX_ACTION_ARGS - len(args))
    return type_id, np.array(padded_args, dtype=np.float32), relevant_card_data


def encode_action_list(game, action_list, player):
    action_list = action_list[:MAX_ACTIONS]
    type_ids, args_list, rel_card_data = zip(*[encode_action(game, a, player) for a in action_list])
    pad_count = MAX_ACTIONS - len(action_list)
    type_ids = list(type_ids) + [0] * pad_count
    args_list = list(args_list) + [np.zeros(MAX_ACTION_ARGS)] * pad_count
    rel_card_data = list(rel_card_data) + [encode_dummy_card()] * pad_count
    mask = [1] * len(action_list) + [0] * pad_count
    return (
        np.array(type_ids, dtype=np.int32),
        np.stack(args_list).astype(np.float32),
        np.array(mask, dtype=np.float32),
        np.stack(rel_card_data).astype(np.float32)
    )
"""
def encode_previous_actions(previous_actions):
    previous_actions = previous_actions[-MAX_PREVIOUS_ACTIONS:]
    type_ids, args_list = zip(*[encode_action(a, -1) for a in previous_actions]) if previous_actions else ([], [])
    pad_count = MAX_PREVIOUS_ACTIONS - len(previous_actions)
    type_ids = list(type_ids) + [0] * pad_count
    args_list = list(args_list) + [np.zeros(MAX_ACTION_ARGS)] * pad_count

    return (
        np.array(type_ids, dtype=np.int32),
        np.stack(args_list).astype(np.float32)
    )
"""
