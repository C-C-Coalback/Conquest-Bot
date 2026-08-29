import CardObject
from unknown_card_name import UNKNOWN_CARD_NAME
from elementValues import VALUE_DAMAGE_WARLORD, VALUE_DAMAGE, VALUE_BASE_ATTACHMENTS, \
    VALUE_COST_ATTACHMENTS, VALUE_UNITS_IN_PLAY, VALUE_UNITS_IN_HQ
from conquestdb_data import planet_df


class Player:
    def __init__(self, player_name, number):
        self.name = player_name
        self.number = number
        self.headquarters = []
        self.cards_in_play = [[] for _ in range(7)]
        self.attachments_at_planets = [[] for _ in range(7)]
        self.resources = 0
        self.hand = []
        self.discard = []
        self.removed = []
        self.victory_display = []
        self.target_hand = -1
        self.target_discard = -1
        self.warlord_is_bloodied = False
        self.previous_actions = []

    def search_hand_low_cost_command_unit(self):
        for i in range(len(self.hand)):
            card = self.get_card_in_hand(i)
            if card is not None:
                if card.check_if_command_unit() and card.get_cost() <= 1:
                    return True
        return False

    def get_number(self):
        return self.number

    def get_previous_actions(self):
        return self.previous_actions

    def count_command_at_planet(self, planet_pos):
        total_command = 0
        for i in range(len(self.cards_in_play[planet_pos])):
            total_command += self.cards_in_play[planet_pos][i].get_command()
        return total_command

    def count_attack_at_planet(self, planet_pos):
        total_command = 0
        for i in range(len(self.cards_in_play[planet_pos])):
            total_command += self.cards_in_play[planet_pos][i].get_attack()
        return total_command

    def count_value_of_units_in_hq(self):
        total_value = 0
        for i in range(len(self.headquarters)):
            card = self.get_card_given_pos(-2, i)
            total_value += card.get_cost()
            damage_factor = VALUE_DAMAGE
            if card.get_card_type() == "Warlord":
                damage_factor = VALUE_DAMAGE_WARLORD
            total_value += damage_factor * card.get_damage()
            attachments = card.get_attachments()
            for j in range(len(attachments)):
                positive_attachment = 1
                if attachments[j].get_card_name() in ["Suffering", "Dire Mutation"]:
                    positive_attachment = -1
                total_value += positive_attachment * (
                    VALUE_BASE_ATTACHMENTS + (attachments[j].get_cost() * VALUE_COST_ATTACHMENTS)
                )
        return total_value

    def count_value_of_units_in_play(self, first_planet):
        total_value = 0
        for i in range(7):
            for j in range(len(self.cards_in_play[i])):
                card = self.get_card_given_pos(i, j)
                if i == first_planet:
                    total_value += card.get_attack() - card.get_command()
                total_value += card.get_cost()
                total_value += VALUE_DAMAGE * card.get_damage()
                attachments = card.get_attachments()
                for k in range(len(attachments)):
                    positive_attachment = 1
                    if attachments[k].get_card_name() in ["Suffering", "Dire Mutation"]:
                        positive_attachment = -1
                    total_value += positive_attachment * (
                        VALUE_BASE_ATTACHMENTS + (attachments[k].get_cost() * VALUE_COST_ATTACHMENTS)
                    )
        return total_value

    def get_name(self):
        return self.name

    def get_headquarters(self):
        return self.headquarters

    def get_cards_in_play(self):
        return self.cards_in_play

    def get_cards_at_planet(self, planet_pos, round_number=0):
        planet_pos = planet_pos + round_number
        if planet_pos > 6:
            return []
        return self.cards_in_play[planet_pos]

    def get_card_given_pos(self, planet_pos, unit_pos):
        if planet_pos == -2:
            if unit_pos >= len(self.headquarters):
                return None
            return self.headquarters[unit_pos]
        if planet_pos < 0 or planet_pos > 6:
            return None
        if unit_pos >= len(self.cards_in_play[planet_pos]):
            return None
        return self.cards_in_play[planet_pos][unit_pos]

    def get_attachment_given_pos(self, planet_pos, unit_pos, attachment_pos):
        if planet_pos == -2:
            return self.headquarters[unit_pos].get_attachment_at_pos(attachment_pos)
        return self.cards_in_play[planet_pos][unit_pos].get_attachment_at_pos(attachment_pos)

    def get_attachments_at_planets(self):
        return self.attachments_at_planets

    def get_attachments_at_planet(self, planet_pos):
        return self.attachments_at_planets[planet_pos]

    def get_attachment_at_planet_given_pos(self, planet_pos, attachment_pos):
        return self.attachments_at_planets[planet_pos][attachment_pos]

    def get_selected_card_hand(self):
        if self.target_hand != -1:
            return self.get_card_in_hand(self.target_hand)
        return None

    def get_hand(self):
        return self.hand

    def get_hidden_hand(self):
        return [UNKNOWN_CARD_NAME for _ in range(len(self.hand))]

    def get_card_in_hand(self, hand_pos):
        if hand_pos >= len(self.hand):
            return None
        return self.hand[hand_pos]

    def get_discard(self):
        return self.discard

    def get_card_in_discard(self, discard_pos):
        return self.discard[discard_pos]

    def get_removed(self):
        return self.removed

    def get_card_in_removed(self, removed_pos):
        return self.removed[removed_pos]

    def get_victory_display(self):
        return self.victory_display

    def count_icons_victory_display(self):
        num_red = 0
        num_blue = 0
        num_green = 0
        for i in range(len(self.victory_display)):
            planet_name = self.victory_display[i]
            if planet_df.loc[planet_name, "red"]:
                num_red += 1
            if planet_df.loc[planet_name, "green"]:
                num_green += 1
            if planet_df.loc[planet_name, "blue"]:
                num_blue += 1
        return num_red, num_blue, num_green

    def get_planet_in_victory_display(self, planet_pos):
        return self.victory_display[planet_pos]

    def get_resources(self):
        return self.resources

    def add_card_to_headquarters(self, card_name, ready=True, damage=0, faith=0, text="", target=False, attachments=None, hale=True):
        card = CardObject.Card(card_name, ready, damage, faith, text, target, hale=hale)
        if not hale:
            self.warlord_is_bloodied = True
        if attachments is not None:
            for i in range(len(attachments)):
                attachment_card_data = attachments[i].split(sep="+")[:3]
                attachment_card_name, attachment_ready, _ = attachment_card_data
                if attachment_ready == "R":
                    attachment_ready = True
                else:
                    attachment_ready = False
                attachment_card = CardObject.Card(attachment_card_name, attachment_ready, 0, 0, "", False)
                card.add_attachment(attachment_card)
        self.headquarters.append(card)

    def get_card_being_damaged(self):
        for i in range(len(self.headquarters)):
            if self.headquarters[i].get_target(as_num=False) == "red":
                return self.headquarters[i]
        for i in range(len(self.cards_in_play)):
            for j in range(len(self.cards_in_play[i])):
                if self.cards_in_play[i][j].get_target(as_num=False) == "red":
                    return self.cards_in_play[i][j]
        return None

    def add_card_to_planet(self, card_name, planet_pos, ready=True, damage=0, faith=0, text="", target=False, attachments=None, hale=True):
        card = CardObject.Card(card_name, ready, damage, faith, text, target, hale=hale)
        if not hale:
            self.warlord_is_bloodied = True
        if attachments is not None:
            for i in range(len(attachments)):
                attachment_card_data = attachments[i].split(sep="+")[:3]
                attachment_card_name, attachment_ready, _ = attachment_card_data
                if attachment_ready == "R":
                    attachment_ready = True
                else:
                    attachment_ready = False
                attachment_card = CardObject.Card(attachment_card_name, attachment_ready, 0, 0, "", False)
                card.add_attachment(attachment_card)
        self.cards_in_play[planet_pos].append(card)

    def add_card_to_hand(self, card_name):
        card = CardObject.Card(card_name, True, 0, 0, "", False, True)
        self.hand.append(card)

    def add_card_to_discard(self, card_name):
        self.discard.append(card_name)
