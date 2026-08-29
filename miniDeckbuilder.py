from conquestdb_data import df
import random
from cardDeckbuildingWeights import weights_main_faction, weights_ally_faction

valid_df = df[(df["card type"].isin(["Army", "Event", "Attachment", "Support"]))]
DEFAULT_NUM_CARDS_CHANGED = 10

def change_deck_content(warlord, ally, cards_in_deck_list, num_cards_changed=DEFAULT_NUM_CARDS_CHANGED):
    warlord_faction = df.loc[warlord]["faction"]
    current_valid_df = valid_df[((valid_df["faction"] == warlord_faction) & (valid_df["loyalty"] != "Signature")) | (
            (valid_df["faction"] == ally) & (valid_df["loyalty"] == "Common")) | (valid_df["faction"] == "Neutral")]
    valid_cards_that_can_be_added = current_valid_df.index.to_list()
    weights_cards_that_can_be_added = []
    for i in range(len(valid_cards_that_can_be_added)):
        card_name = valid_cards_that_can_be_added[i]
        card_faction = df.loc[card_name]["faction"]
        if card_faction == ally:
            weights_cards_that_can_be_added.append(weights_ally_faction[card_name])
        else:
            weights_cards_that_can_be_added.append(weights_main_faction[card_name])
    for i in range(num_cards_changed):
        cards_in_deck_list.pop(random.randint(0, len(cards_in_deck_list) - 1))
    for i in range(num_cards_changed):
        while True:
            if not valid_cards_that_can_be_added:
                break
            card_name = random.choices(valid_cards_that_can_be_added, weights=weights_cards_that_can_be_added, k=1)[0]
            if check_if_can_add_card_to_deck(card_name, cards_in_deck_list):
                cards_in_deck_list.append(card_name)
                break
            else:
                index = valid_cards_that_can_be_added.index(card_name)
                valid_cards_that_can_be_added.remove(card_name)
                del weights_cards_that_can_be_added[index]
    return cards_in_deck_list

def clean_deck(deck_message):
    deck_sections = deck_message.split(sep="----------------------------------------------------------------------")
    individual_parts = []
    for i in range(len(deck_sections)):
        individual_parts += deck_sections[i].split(sep="\n")
    individual_parts = [x for x in individual_parts if x]
    return individual_parts

def load_deck_from_text(deck_text):
    deck_text_split = clean_deck(deck_text)
    deck_name = deck_text_split[0]
    warlord = deck_text_split[1]
    factions = deck_text_split[2].split(sep=" (")
    if len(factions) == 2:
        factions[1] = factions[1][:-1]
    if len(factions) == 1:
        factions.append("")
    ally = factions[1]
    current_index = 4
    signature_cards = []
    while current_index < len(deck_text_split):
        if deck_text_split[current_index] == "Army":
            break
        if deck_text_split[current_index] == "Signature Squad":
            current_index += 1
            continue
        signature_cards.append(deck_text_split[current_index])
        current_index += 1
    current_index += 1
    cards_in_deck_list = []
    skippers = ["Army", "Event", "Attachment", "Support", "Synapse"]
    while current_index < len(deck_text_split):
        if deck_text_split[current_index] in skippers:
            current_index += 1
            continue
        if deck_text_split[current_index] == "Planet":
            break
        if len(deck_text_split[current_index]) > 3:
            current_name = deck_text_split[current_index][3:]
            current_amount = int(deck_text_split[current_index][0])
            for _ in range(current_amount):
                cards_in_deck_list.append(current_name)
        current_index += 1
    return {"warlord": warlord, "ally": ally, "deck_name": deck_name, "cards_in_deck": cards_in_deck_list, "signature_squad": signature_cards}

def convert_deck_data_to_text_data(data):
    warlord = data["warlord"]
    warlord_faction = df.loc[warlord]["faction"]
    ally = data["ally"]
    deck_name = data["deck_name"]
    cards_in_deck_list = data["cards_in_deck"]
    signature_squad = data["signature_squad"]
    deck_text = ""
    deck_text += deck_name + "\n"
    deck_text += "----------------------------------------------------------------------\n"
    deck_text += warlord + "\n"
    deck_text += warlord_faction
    if ally:
        deck_text += " (" + ally + ")"
    deck_text += "\n"
    deck_text += "----------------------------------------------------------------------\n"
    deck_text += "Signature Squad\n\n"
    for i in range(len(signature_squad)):
        deck_text += signature_squad[i] + "\n"
    deck_text += "----------------------------------------------------------------------\n"
    deck_text += "Army\n"
    deck_text += "\n"
    army_cards = []
    support_cards = []
    event_cards = []
    attachment_cards = []
    synapse_cards = []
    while cards_in_deck_list:
        card_name = cards_in_deck_list[0]
        card_amount = cards_in_deck_list.count(card_name)
        card_type = df.loc[card_name]["card type"]
        if card_type == "Army":
            army_cards.append(str(card_amount) + "x " + card_name)
        elif card_type == "Support":
            support_cards.append(str(card_amount) + "x " + card_name)
        elif card_type == "Event":
            event_cards.append(str(card_amount) + "x " + card_name)
        elif card_type == "Attachment":
            attachment_cards.append(str(card_amount) + "x " + card_name)
        elif card_type == "Synapse":
            synapse_cards.append(str(card_amount) + "x " + card_name)
        while card_name in cards_in_deck_list: cards_in_deck_list.remove(card_name)
    for i in range(len(army_cards)):
        deck_text += army_cards[i] + "\n"
    deck_text += "----------------------------------------------------------------------\n"
    deck_text += "Support\n"
    for i in range(len(support_cards)):
        deck_text += support_cards[i] + "\n"
    deck_text += "----------------------------------------------------------------------\n"
    deck_text += "Synapse\n"
    for i in range(len(synapse_cards)):
        deck_text += synapse_cards[i] + "\n"
    deck_text += "----------------------------------------------------------------------\n"
    deck_text += "Attachment\n"
    for i in range(len(attachment_cards)):
        deck_text += attachment_cards[i] + "\n"
    deck_text += "----------------------------------------------------------------------\n"
    deck_text += "Event\n"
    for i in range(len(event_cards)):
        deck_text += event_cards[i] + "\n"
    deck_text += "----------------------------------------------------------------------\n"
    deck_text += "Planet\n"
    return deck_text

def check_if_can_add_card_to_deck(card_name, cards_in_deck_list):
    if cards_in_deck_list.count(card_name) > 2:
        return False
    return True


def save_deck_locally(deck_name, deck_text):
    with open("decks/" + deck_name, "w") as deck_file:
        deck_file.write(deck_text)


def load_adjust_deck(deck_name, num_cards_changed=DEFAULT_NUM_CARDS_CHANGED, save=True):
    with open("decks/" + deck_name, "r") as file:
        deck_text = file.read()
    data = load_deck_from_text(deck_text)

    deck_name = data["deck_name"]
    warlord = data["warlord"]
    ally = data["ally"]
    cards_in_deck_list = data["cards_in_deck"]
    cards_in_deck_list = change_deck_content(warlord, ally, cards_in_deck_list, num_cards_changed=num_cards_changed)
    data["cards_in_deck"] = cards_in_deck_list
    deck_text = convert_deck_data_to_text_data(data)
    if save:
        save_deck_locally(deck_name, deck_text)
    return deck_text


