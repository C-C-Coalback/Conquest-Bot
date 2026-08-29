from conquestdb_data import vocab, attachment_vocab, planet_vocab, \
    headquarters_vocab, hand_vocab, in_play_vocab, discard_vocab


MAX_RESOURCES = 100
MAX_DAMAGE = 30
MAX_HAND_SIZE = 30
MAX_DISCARD_SIZE = 5
MAX_REMOVED_SIZE = 10
MAX_CARDS_IN_ONE_PLAY_ZONE = 30
MAX_ATTACHMENTS_PER_CARD = 3
MAX_SEARCHED_CARDS = 6
MAX_ACTIONS = 20
VOCAB_SIZE = len(vocab)
IN_PLAY_VOCAB_SIZE = len(in_play_vocab)
HEADQUARTERS_VOCAB_SIZE = len(headquarters_vocab)
HAND_VOCAB_SIZE = len(hand_vocab)
DISCARD_VOCAB_SIZE = len(discard_vocab)
PLANET_VOCAB_SIZE = len(planet_vocab)
ATTACHMENT_VOCAB_SIZE = len(attachment_vocab)
OUTPUT_DIM = 16
SCALARS_PER_ATTACHMENT = 1
ATTACHMENT_CARD_FEATURE_DIM = ATTACHMENT_VOCAB_SIZE + SCALARS_PER_ATTACHMENT  # embedding(Card Size) + ready(1)
SCALARS_PER_CARD = 9 # ready(1) + damage(1) + target(1) + attack(1) + health(1) + command(1) + card type(1) + faction(1)
CARD_FEATURE_DIM = 1*SCALARS_PER_CARD + MAX_ATTACHMENTS_PER_CARD*SCALARS_PER_ATTACHMENT  # embedding(Card Size) + Scalars per card + (Attachment Size * Max Attachments)
PHASES = {"DEPLOY": 1, "COMMAND": 2, "COMBAT": 3, "HEADQUARTERS": 4}
NUM_PHASES = len(PHASES)
UNIQUE_CHOICES = {"Gains from Tarrus": 1, "Shadowsun plays attachment from hand or discard?": 2, "Use Nullify?": 3, "Interrupt Effect?": 4, "Use alternative shield effect?": 5, 
                  "Eldorath Starbane": 6, "Foresight": 7, "Alaitoc Shrine": 8, "Resolve Battle Ability?": 9, "Amount to spend for Tzeentch's Firestorm:": 10, 
                  "Mulligan Opening Hand?": 11, "Use No Mercy?": 12, "Use Guardian Mesh Armor?": 13, "Use an extra source of damage?": 14, "Use The Fury of Sicarius?": 15, 
                  "Cato's Stronghold": 16, "Which deck to use Biel-Tan Warp Spiders:": 17, "Retreat Warlord?": 18, "Promethium Mine": 19}
UNIQUE_CHOICES["Unknown Choice"] = 0
NUM_SPECIAL_CHOICES = max(len(UNIQUE_CHOICES), 100)
CARD_TYPES = {"Warlord": 0, "Army": 1, "Token": 2, "Synapse": 3, "Support": 4, "Attachment": 5, "Event": 6}
NUM_CARD_TYPES = len(CARD_TYPES)
FACTIONS = {"Neutral": 0, "Space Marines": 1, "Tau": 2, "Eldar": 3, "Dark Eldar": 4, "Chaos": 5, "Orks": 6, "Astra Militarum": 7, "Tyranids": 8, "Necrons": 9}
NUM_FACTIONS = len(FACTIONS)
ACTION_TYPES = {"pass-P1": 0, "HAND": 1, "HQ": 2, "IN_PLAY": 3, "ATTACHMENT_HQ": 4, "ATTACHMENT_IN_PLAY": 5,
                "RESERVE": 6, "IN_DISCARD": 7, "REMOVED": 8, "PLANETS": 9, "CHOICE": 10, "SEARCH": 11,
                "SPECIAL_ACTION_HAND": 12, "SPECIAL_ACTION_HQ": 13, "SPECIAL_ACTION_IN_PLAY": 14,
                "SPECIAL_ACTION_ATTACHMENT_HQ": 15, "SPECIAL_ACTION_ATTACHMENT_IN_PLAY": 16, "SPECIAL_ACTION_RESERVE": 17,
                "SPECIAL_ACTION_IN_DISCARD": 18, "SPECIAL_ACTION_REMOVED": 19, "SPECIAL_ACTION_PLANET_ATTACHMENT": 20}
NUM_ACTION_TYPES = len(ACTION_TYPES)
MAX_ACTION_ARGS = 3
MAX_PREVIOUS_ACTIONS = 5
MODES = {"Normal": 0, "ACTION": 1, "RETREAT": 2, "DISCOUNT": 3, "UNKNOWN": 4}
NUM_MODES = len(MODES)
TARGET_TYPES = 4
NUM_SCALARS = 19
MAX_ARG_VALUE = 30
NUM_PLANETS = 5
NUM_ROUNDS = 7
MAX_ONE_ICON = 3
NUM_HEADS = 4
FF_DIM = 32
HEAD_SIZE = 256
DROPOUT_RATE = 0.1

# Card attributes
MAX_COST = 10
MAX_ATTACK = 9
MAX_HEALTH = 11
MAX_COMMAND = 6
