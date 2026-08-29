import pandas as pd
from unknown_card_name import UNKNOWN_CARD_NAME

df = pd.read_csv("conquestdb_card_data.csv")
# df = df[df["cycle"] == "Core Set"]
df = df[df["faction"].isin(["Tau", "Space Marines", "Neutral"])]
card_names = df["name"].to_list()
special_attachments = ["Gun Drones", "Shadowsun's Stealth Cadre"]
special_attachments.append("Escort Drone")
special_attachments.append("Cult of Khorne")
special_units = []
special_units.append("The Glovodan Eagle")
hand_df = df[(df["card type"].isin(["Army", "Event", "Attachment", "Support"]))]
hand_names = hand_df["name"].to_list()
discard_df = df[(df["card type"].isin(["Army", "Event", "Attachment", "Support", "Warlord", "Synapse"]))]
discard_names = discard_df["name"].to_list()
in_play_df = df[(df["card type"].isin(["Army", "Warlord", "Token", "Synapse"])) | df["name"].isin(special_units)]
in_play_names = in_play_df["name"].to_list()
headquarters_df = df[(df["card type"].isin(["Army", "Warlord", "Token", "Support", "Synapse"])) | df["name"].isin(special_units)]
headquarters_names = headquarters_df["name"].to_list()
attachment_df = df[(df["card type"] == "Attachment") | (df["name"].isin(special_attachments))]
attachment_names = attachment_df["name"].to_list()
df.set_index("name", inplace=True)


hand_vocab = {card_name: idx + 1 for idx, card_name in enumerate(hand_names)}
hand_vocab[UNKNOWN_CARD_NAME] = 0
discard_vocab = {card_name: idx + 1 for idx, card_name in enumerate(discard_names)}
discard_vocab[UNKNOWN_CARD_NAME] = 0
in_play_vocab = {card_name: idx + 1 for idx, card_name in enumerate(in_play_names)}
in_play_vocab[UNKNOWN_CARD_NAME] = 0
headquarters_vocab = {card_name: idx + 1 for idx, card_name in enumerate(headquarters_names)}
headquarters_vocab[UNKNOWN_CARD_NAME] = 0
vocab = {card_name: idx + 1 for idx, card_name in enumerate(card_names)}
vocab[UNKNOWN_CARD_NAME] = 0
attachment_vocab = {card_name: idx + 1 for idx, card_name in enumerate(attachment_names)}
attachment_vocab[UNKNOWN_CARD_NAME] = 0

planet_df = pd.read_csv("conquest_planet_data.csv")
planet_df = planet_df[planet_df["cycle"] == "Core Set"]
planet_names = planet_df["name"].to_list()
planet_df.set_index("name", inplace=True)
planet_vocab = {card_name: idx for idx, card_name in enumerate(planet_names)}
planet_vocab[UNKNOWN_CARD_NAME] = len(planet_vocab)
