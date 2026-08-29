import pandas as pd

cards_df = pd.read_csv("conquestdb_card_data.csv")
cards_df.set_index("name", inplace=True)
cards_df["command"] = cards_df["command"].astype(int)
cards_df["health"] = cards_df["health"].astype(int)
cards_df["attack"] = cards_df["attack"].astype(int)
cards_df["cost"] = cards_df["cost"].astype(int)


class Card:
    def __init__(self, card_name, ready_state=True, damage=0, faith=0, text="", target=False, hale=True):
        self.card_name = card_name
        self.ready = ready_state
        self.damage = damage
        self.faith = faith
        self.additional_text_info = text
        self.attachments = []
        self.target = target
        self.hale = hale
        if card_name == "Unknown":
            self.cost = 0
            self.command = 0
            self.attack = 0
            self.health = 0
            self.card_type = "Army"
            self.faction = "Neutral"
            self.shields = 0
        else:
            self.card_type = cards_df.loc[card_name, "card type"]
            self.faction = cards_df.loc[card_name, "faction"]
            self.shields = 0
            if self.card_type in ["Event", "Attachment"]:
                self.shields = cards_df.loc[card_name, "shields"]
            if self.card_type == "Warlord":
                self.command = 0
                self.cost = 0
            else:
                self.cost = cards_df.loc[card_name, "cost"]
                self.command = cards_df.loc[card_name, "command"]
            self.command = max(0, self.command)
            self.cost = max(0, self.cost)
            self.attack = cards_df.loc[card_name, "attack"]
            self.health = cards_df.loc[card_name, "health"]
            split_text = self.additional_text_info.split("\n")
            extra_attack = 0
            for i in range(len(split_text)):
                if "Extra Attack" in split_text[i]:
                    extra_attack += int(split_text[i].split(" ")[-1])
            self.attack = self.attack + extra_attack
            extra_health = 0
            for i in range(len(split_text)):
                if "Extra Health" in split_text[i]:
                    extra_health += int(split_text[i].split(" ")[-1])
            self.health = self.health + extra_health
            self.health = max(0, self.health)
            self.attack = max(0, self.attack)

    def check_if_combat_unit(self):
        if self.get_card_type() == "Army":
            if self.get_cost() >= 3 and self.get_attack() >= 3:
                return True
        return False

    def check_if_command_unit(self):
        if self.get_card_type() == "Army":
            if self.get_command() > 0:
                if self.get_command() >= self.get_cost():
                    return True
        return False

    def get_faction(self):
        return self.faction

    def get_card_type(self):
        return self.card_type

    def get_shields(self):
        return self.shields

    def get_cost(self):
        return self.cost

    def get_command(self):
        command = self.command
        for i in range(len(self.attachments)):
            if self.attachments[i].get_card_name() == "Promotion":
                command += 2
        return command

    def get_attack(self):
        return self.attack

    def get_health(self):
        return self.health

    def get_cost(self):
        return self.cost

    def get_target(self, as_num=True):
        if as_num:
            if not self.target:
                return 0
            if self.target == "blue":
                return 1
            if self.target == "red":
                return 2
            if self.target == "green":
                return 3
            return 0
        return self.target

    def get_hale(self):
        return self.hale

    def get_name(self):
        return self.get_card_name()

    def get_card_name(self):
        return self.card_name

    def get_ready(self):
        return self.ready

    def get_damage(self):
        return self.damage

    def get_faith(self):
        return self.faith

    def get_text(self):
        return self.additional_text_info

    def get_attachments(self):
        return self.attachments

    def get_attachment_at_pos(self, position):
        return self.attachments[position]

    def add_attachment(self, card):
        self.attachments.append(card)
