from PlayerObject import Player
import random
from conquestdb_data import planet_vocab, planet_df
from maxValues import MAX_DAMAGE, MAX_HAND_SIZE
from elementValues import VALUE_CARDS, VALUE_RESOURCES, VALUE_UNITS_IN_PLAY, VALUE_UNITS_IN_HQ, \
    VALUE_DAMAGE_WARLORD, VALUE_BLOODIED_WARLORD, VALUE_ICONS_VICTORY_DISPLAY, VALUE_WINNING_COMMAND_AT_PLANET, REWARD_EOR_FACTOR, VALUE_BASE_CAPTURED_PLANET


class Game:
    def __init__(self, name_1, name_2, production=False):
        self.player_one = Player(name_1, 1)
        self.player_two = Player(name_2, 2)
        self.planets_in_play = ["Unknown" for _ in range(7)]
        self.round_number = 1
        self.initiative_holder = 1
        self.phase = ""
        self.info_box = ""
        self.choices = []
        self.choice_context = ""
        self.targeted_planet = 7
        self.infested_planets = [False for _ in range(7)]
        self.text_planets = ["" for _ in range(7)]
        self.active_player = ""
        self.active_context = ""
        self.active_options = []
        self.winner = ""
        self.action_between_combat_turns_window = False
        self.damage_that_can_be_shielded = 0
        self.mode = "UNKNOWN"
        self.last_eor_reward_estimate_player_one = 0
        self.last_round_id_nums = [0, 0]
        self.cards_in_search_box = []
        self.production = production

    def check_if_first_planet(self, planet_pos):
        if self.round_number - 1 == planet_pos:
            return True
        return False

    def get_players_winning_command_struggles_perspective_active_player(self):
        player_number = self.get_active_player_number()
        round_number = self.get_round_number()
        i = round_number - 1
        winning_command_struggles = []
        while i < 7 and len(winning_command_struggles) < 5:
            player_winning_command_struggle = self.determine_player_number_winning_command_struggle(i)
            if player_winning_command_struggle == 0:
                winning_command_struggles.append(0.5)
            elif player_winning_command_struggle == player_number:
                winning_command_struggles.append(1.0)
            else:
                winning_command_struggles.append(0.0)
            i = i + 1
        while len(winning_command_struggles) < 5:
            winning_command_struggles.append(0.5)
        return winning_command_struggles

                    
    def get_colors_first_planet(self):
        planet_name = self.get_planet_given_pos(self.round_number)
        if planet_name == "Unknown":
            return 0, 0, 0
        return int(planet_df.loc[planet_name, "red"] == True), int(planet_df.loc[planet_name, "blue"] == True), int(planet_df.loc[planet_name, "green"] == True)

    def get_targeted_planet_name(self):
        if self.targeted_planet == 7:
            return ""
        return self.get_planet_given_pos(self.targeted_planet)

    def get_targeted_planet(self):
        return self.targeted_planet

    def estimate_reward_end_of_round(self, player_number):
        total_reward = 0
        if player_number == 1:
            player = self.get_player_one()
            enemy_player = self.get_player_two()
        else:
            player = self.get_player_two()
            enemy_player = self.get_player_one()
        total_reward += VALUE_RESOURCES * (player.get_resources() - enemy_player.get_resources())
        total_reward += VALUE_CARDS * (len(player.get_hand()) - len(enemy_player.get_hand()))
        first_planet_pos = self.get_round_number() - 1
        total_reward += VALUE_UNITS_IN_PLAY * (
            player.count_value_of_units_in_play(first_planet_pos) -
            enemy_player.count_value_of_units_in_play(first_planet_pos)
        )
        total_reward += VALUE_UNITS_IN_HQ * (
            player.count_value_of_units_in_hq() - enemy_player.count_value_of_units_in_hq()
        )
        if player.warlord_is_bloodied:
            total_reward += VALUE_BLOODIED_WARLORD
        if enemy_player.warlord_is_bloodied:
            total_reward += -VALUE_BLOODIED_WARLORD
        for i in range(7):
            if self.planets_in_play[i] != "Unknown":
                player_winning_command_struggle = self.determine_player_number_winning_command_struggle(i)
                if player_winning_command_struggle == 0:
                    pass
                else:
                    planet_name = self.planets_in_play[i]
                    command_rewards = (planet_df.loc[planet_name, "cards"], planet_df.loc[planet_name, "resources"])
                    if player_winning_command_struggle == player_number:
                        total_reward += VALUE_WINNING_COMMAND_AT_PLANET[0] * command_rewards[0]
                        total_reward += VALUE_WINNING_COMMAND_AT_PLANET[1] * command_rewards[1]
                    else:
                        total_reward -= VALUE_WINNING_COMMAND_AT_PLANET[0] * command_rewards[0]
                        total_reward -= VALUE_WINNING_COMMAND_AT_PLANET[1] * command_rewards[1]
        total_reward += self.perform_victory_display_reward_calcs(player_number)
        total_reward = total_reward * REWARD_EOR_FACTOR
        if player_number == 1:
            total_reward = total_reward - self.last_eor_reward_estimate_player_one
            self.last_eor_reward_estimate_player_one = total_reward
        else:
            total_reward = total_reward + self.last_eor_reward_estimate_player_one
            self.last_eor_reward_estimate_player_one = -total_reward
        print("EOR Reward:", total_reward)
        if total_reward < -1:
            total_reward = -1
        if total_reward > 1:
            total_reward = 1
        print("Adjusted Reward:", total_reward)
        return total_reward

    def perform_victory_display_reward_calcs(self, player_number):
        total_reward = 0
        num_red, num_blue, num_green = self.player_one.count_icons_victory_display()
        factor_display = 1
        if player_number == 2:
            factor_display = -1
        total_reward += factor_display * (num_red ** 2)
        total_reward += factor_display * (num_blue ** 2)
        total_reward += factor_display * (num_green ** 2)
        total_reward = total_reward * VALUE_ICONS_VICTORY_DISPLAY
        total_reward += factor_display * len(self.player_one.victory_display) * VALUE_BASE_CAPTURED_PLANET
        num_red, num_blue, num_green = self.player_two.count_icons_victory_display()
        enemy_reward = 0
        factor_display = 1
        if player_number == 1:
            factor_display = -1
        enemy_reward += factor_display * (num_red ** 2)
        enemy_reward += factor_display * (num_blue ** 2)
        enemy_reward += factor_display * (num_green ** 2)
        enemy_reward = enemy_reward * VALUE_ICONS_VICTORY_DISPLAY
        enemy_reward += factor_display * len(self.player_one.victory_display) * VALUE_BASE_CAPTURED_PLANET
        total_reward = total_reward + enemy_reward
        return total_reward

    def determine_player_number_winning_command_struggle(self, planet_pos):
        command_one = self.player_one.count_command_at_planet(planet_pos)
        command_two = self.player_two.count_command_at_planet(planet_pos)
        if command_one > command_two:
            return 1
        elif command_two > command_one:
            return 2
        return 0

    def get_mode(self):
        return self.mode
    
    def get_searched_cards(self):
        return self.cards_in_search_box

    def get_damage_that_can_be_shielded(self):
        return min(self.damage_that_can_be_shielded, MAX_DAMAGE)

    def get_winner_number(self):
        if self.winner == self.player_one.get_name():
            return 1
        elif self.winner == self.player_two.get_name():
            return 2
        return -1

    def get_active_player_number(self):
        if self.active_player == self.player_one.get_name():
            return 1
        elif self.active_player == self.player_two.get_name():
            return 2
        return -1

    def determine_if_active_player_has_initiative(self):
        if self.initiative_holder == 1:
            if self.active_player == self.player_one.get_name():
                return True
            return False
        if self.active_player == self.player_two.get_name():
            return True
        return False

    def get_scalars_perspective_player(self, player):
        enemy_player = self.player_two
        if enemy_player.get_name() == player.get_name():
            enemy_player = self.player_one
        scalars = [player.get_resources(), enemy_player.get_resources(),
                   player.target_hand, player.target_discard, enemy_player.target_discard,
                   1 if self.determine_if_active_player_has_initiative() else 0,
                   self.get_targeted_planet(), self.action_between_combat_turns_window,
                   self.get_damage_that_can_be_shielded(), self.choice_context,
                   self.get_round_number(), self.get_phase(), self.get_mode()]
        num_red, num_blue, num_green = player.count_icons_victory_display()
        scalars = scalars + [num_red, num_blue, num_green]
        num_red, num_blue, num_green = enemy_player.count_icons_victory_display()
        scalars = scalars + [num_red, num_blue, num_green]
        scalars = scalars + [player.warlord_is_bloodied, enemy_player.warlord_is_bloodied]
        scalars = scalars + [len(enemy_player.get_hand()) / MAX_HAND_SIZE]
        return scalars

    def get_active_player(self):
        if self.active_player == self.player_one.get_name():
            return self.player_one
        return self.player_two

    def get_inactive_player(self):
        if self.active_player != self.player_one.get_name():
            return self.player_one
        return self.player_two

    def determine_player(self, player_name):
        if player_name == self.player_one.get_name():
            player = self.player_one
        else:
            player = self.player_two
        return player

    def determine_enemy_player(self, player_name):
        if player_name == self.player_one.get_name():
            player = self.player_two
        else:
            player = self.player_one
        return player

    def get_cards_at_planet_for_player(self, player_name, planet_pos):
        player = self.determine_player(player_name)
        return player.get_cards_at_planet(planet_pos)

    def get_cards_in_hq_for_player(self, player_name):
        player = self.determine_player(player_name)
        return player.get_headquarters()

    def get_cards_in_discard_for_player(self, player_name):
        player = self.determine_player(player_name)
        return player.get_hand()

    def get_cards_in_hand_for_player(self, player_name):
        player = self.determine_player(player_name)
        return player.get_hand()

    def get_enemy_cards_in_hand_for_player(self, player_name):
        player = self.determine_enemy_player(player_name)
        return player.get_hidden_hand()

    def create_automated_choice_message(self, choice):
        if "SPECIAL_ACTION_" in choice:
            choice = choice.replace("SPECIAL_ACTION_", "SPECIAL_ACTION/")
        if self.action_between_combat_turns_window:
            message = "AUTOMATED_SPECIAL_ACTION_CHOICE/" + self.active_player + "/" + choice
            message = '{"message": \"' + message + '\"}'
            return message
        message = "AUTOMATED_CHOICE/" + self.active_player + "/" + choice
        message = '{"message": \"' + message + '\"}'
        return message

    def get_active_options(self):
        return self.active_options

    def update_game_from_game_update_message(self, game_update_string: str):
        split_game_string = game_update_string.split(sep="/")
        if split_game_string[0] == "GAME_INFO":
            if split_game_string[1] == "RESOURCES":
                if split_game_string[2] == "1":
                    self.player_one.resources = int(split_game_string[3])
                else:
                    self.player_two.resources = int(split_game_string[3])
            elif split_game_string[1] == "INITIATIVE":
                if split_game_string[2] == "1":
                    self.initiative_holder = 1
                else:
                    self.initiative_holder = 2
            elif split_game_string[1] == "PLANETS":
                planets_with_icons = split_game_string[2:]
                self.targeted_planet = 7
                for i in range(len(planets_with_icons)):
                    separated_planet_data = planets_with_icons[i].split(sep="|")
                    planet_name, infested, targeted_text, p1_attachments, p2_attachments, text = separated_planet_data
                    if planet_name not in planet_vocab:
                        planet_name = "Unknown"
                    self.planets_in_play[i] = planet_name
                    if targeted_text != "":
                        self.targeted_planet = i
                    if infested == "I":
                        self.infested_planets[i] = True
                    else:
                        self.infested_planets[i] = False
                    self.text_planets[i] = text
            elif split_game_string[1] == "VICTORY_DISPLAY":
                player = self.player_one
                if split_game_string[2] == "2":
                    player = self.player_two
                captured_planets = split_game_string[3:]
                if self.production:
                    for i in range(len(captured_planets)):
                        if captured_planets[i] not in planet_vocab:
                            captured_planets[i] = "Unknown"
                player.victory_display = captured_planets
            elif split_game_string[1] == "HAND":
                player = self.player_one
                if split_game_string[2] == "2":
                    player = self.player_two
                hand_data = split_game_string[4:]
                player.hand = []
                player.target_hand = -1
                for i in range(len(hand_data)):
                    card_name, target, _ = hand_data[i].split(sep="|")
                    player.add_card_to_hand(card_name)
                    if target:
                        player.target_hand = i
            elif split_game_string[1] == "SEARCH":
                self.cards_in_search_box = []
                search_data = split_game_string[4:]
                for i in range(len(search_data)):
                    self.cards_in_search_box.append(search_data[i])
            elif split_game_string[1] == "DISCARD":
                player = self.player_one
                if split_game_string[2] == "2":
                    player = self.player_two
                discard_data = split_game_string[4:]
                player.discard = []
                any_target = False
                for i in range(len(discard_data)):
                    card_name, target = discard_data[i].split(sep="|")
                    player.add_card_to_discard(card_name)
                    if target:
                        player.target_discard = i
                        any_target = True
                if not any_target:
                    player.target_discard = 50
            elif split_game_string[1] == "HQ":
                player = self.player_one
                if split_game_string[2] == "2":
                    player = self.player_two
                hq_data = split_game_string[4:]
                player.headquarters = []
                for i in range(len(hq_data)):
                    current_hq_data = hq_data[i].split(sep="|")
                    card_name = current_hq_data[0]
                    ready = current_hq_data[1]
                    if ready == "R" or ready == "PR":
                        ready = True
                    else:
                        ready = False
                    damage = int(current_hq_data[2])
                    faith = int(current_hq_data[3])
                    hale = current_hq_data[4]
                    if hale == "H":
                        hale = True
                    else:
                        hale = False
                    border = current_hq_data[5]
                    text = current_hq_data[6]
                    target = current_hq_data[7]
                    attachments = current_hq_data[8:]
                    player.add_card_to_headquarters(card_name, ready, damage, faith, text, target, attachments, hale=hale)
            elif split_game_string[1] == "IN_PLAY":
                player = self.player_one
                if split_game_string[2] == "2":
                    player = self.player_two
                planet_data = split_game_string[5:]
                planet_pos = int(split_game_string[3])
                player.cards_in_play[planet_pos] = []
                for i in range(len(planet_data)):
                    current_planet_data = planet_data[i].split(sep="|")
                    card_name = current_planet_data[0]
                    ready = current_planet_data[1]
                    if ready == "R" or ready == "PR":
                        ready = True
                    else:
                        ready = False
                    damage = int(current_planet_data[2])
                    faith = int(current_planet_data[3])
                    hale = current_planet_data[4]
                    if hale == "H":
                        hale = True
                    else:
                        hale = False
                    border = current_planet_data[5]
                    text = current_planet_data[6]
                    target = current_planet_data[7]
                    attachments = current_planet_data[8:]
                    player.add_card_to_planet(card_name, planet_pos, ready, damage, faith, text, target, attachments, hale=hale)
            elif split_game_string[1] == "VICTORY_MESSAGE":
                self.winner = split_game_string[2]
                return "FINISHED"
            elif split_game_string[1] == "MISC_AUTOMATED_DATA":
                eor_training_required = False
                round_number = int(split_game_string[2]) + 1
                if self.round_number != round_number:
                    eor_training_required = True
                self.round_number = round_number
                self.phase = split_game_string[3]
                self.mode = split_game_string[4]
                misc_data_context, misc_data_extra = split_game_string[5].split(sep="|||")
                self.choice_context = ""
                self.damage_that_can_be_shielded = 0
                if misc_data_context != "None":
                    if misc_data_context == "Damage":
                        self.damage_that_can_be_shielded = int(misc_data_context[1])
                    if misc_data_context == "CHOICE":
                        self.choice_context = misc_data_extra
                if eor_training_required:
                    return "EOR_TRAINING"
            elif split_game_string[1] == "AUTOMATED_DATA":
                event_context = split_game_string[2]
                name_player = split_game_string[3]
                clickable_elements = split_game_string[4:]
                clickable_elements = "/".join(clickable_elements).split(sep="|||")
                clickable_elements = [item for item in clickable_elements if item]
                self.action_between_combat_turns_window = False
                if event_context == "Action Window Between Combat Turns":
                    self.action_between_combat_turns_window = True
                self.active_player = name_player
                self.active_context = event_context
                self.active_options = clickable_elements
                if self.active_context == "SETUP":
                    return "LOADDECK"
                return "ACTION_REQUIRED"
        return ""

    def resolve_choice(self):
        if not self.active_options:
            return "pass-P1"
        return random.choice(self.active_options)

    def get_player_one(self):
        return self.player_one

    def get_player_two(self):
        return self.player_two

    def get_planets(self):
        planets_in_play = []
        for i in range(5):
            planets_in_play.append(self.get_planet_given_pos(i))
        return planets_in_play

    def get_planet_given_pos(self, position):
        position = position + self.get_round_number() - 1
        if position > 6:
            return "Unknown"
        return self.planets_in_play[position]

    def get_round_number(self):
        return self.round_number

    def get_initiative_holder(self):
        return self.initiative_holder

    def get_phase(self):
        return self.phase

    def get_info_box(self):
        return self.info_box

    def get_choices(self):
        return self.choices

    def get_choice_context(self):
        return self.choice_context

    def get_choice(self, choice_pos):
        return self.choices[choice_pos]
