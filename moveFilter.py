def filter_obvious_bad_moves(game):
    player = game.get_active_player()
    enemy_player = game.get_inactive_player()
    if game.active_context == "Deploy Turn":
        if player.target_hand != -1:
            return filter_bad_deploy_locations(game, player, enemy_player)
        else:
            filter_bad_attachments(game, player, enemy_player)
            if len(game.active_options) > 1:
                if player.get_resources() > 2 or (player.get_resources() > 0 and player.search_hand_low_cost_command_unit()):
                    if "pass-P1" in game.active_options:
                        game.active_options.remove("pass-P1")
        return None
    elif game.active_context == "Choice":
        return filter_choices(game, player, enemy_player)
    elif game.active_context == "Search":
        if len(game.active_options) > 1:
            if "pass-P1" in game.active_options:
                game.active_options.remove("pass-P1")
    elif game.active_context == "Retreat Turn":
        return filter_retreat_turn(game, player, enemy_player)
    elif game.active_context == "Damage":
        return filter_possible_shields(game, player, enemy_player)
    elif game.active_context == "Combat Turn":
        for i in range(len(game.active_options)):
            if "PLANETS" in game.active_options[i]:
                game.active_options = [game.active_options[i]]
                return None
    return None


def filter_bad_attachments(game, player, enemy_player):
    if player.get_resources() < 3:
        return None
    active_options = game.get_active_options()
    removed_options = []
    for i in range(len(active_options)):
        option_data = active_options[i].split(sep="/")
        if len(option_data) == 3:
            if option_data[0] == "HAND":
                card_in_hand = player.get_card_in_hand(int(option_data[2]))
                if card_in_hand.get_card_type() == "Attachment":
                    removed_options.append(active_options[i])
    if len(removed_options) < len(active_options):  # Never remove all of the options
        for i in range(len(removed_options)):
            game.active_options.remove(removed_options[i])
    return None


def filter_possible_shields(game, player, enemy_player):
    if game.damage_that_can_be_shielded == 1:
        active_options = game.get_active_options()
        removed_options = []
        one_shield_card_present = False
        for i in range(len(active_options)):
            option_data = active_options[i].split(sep="/")
            if len(option_data) == 3:
                if option_data[0] == "HAND":
                    if player.get_card_in_hand(int(option_data[2])).get_shields() == 1:
                        one_shield_card_present = True
        if one_shield_card_present:
            for i in range(len(active_options)):
                option_data = active_options[i].split(sep="/")
                if len(option_data) == 3:
                    if option_data[0] == "HAND":
                        if player.get_card_in_hand(int(option_data[2])).get_shields() > 1:
                            removed_options.append(active_options[i])
        if len(removed_options) < len(active_options):  # Never remove all of the options
            for i in range(len(removed_options)):
                game.active_options.remove(removed_options[i])
    card = player.get_card_being_damaged()
    if card is not None:
        remaining_health = card.get_health() - card.get_damage()
        excess_damage = -remaining_health
        minimum_shield_value_to_live = excess_damage + 1
        active_options = game.get_active_options()
        removed_options = []
        for i in range(len(active_options)):
            option_data = active_options[i].split(sep="/")
            if len(option_data) == 3:
                if option_data[0] == "HAND":
                    card_in_hand = player.get_card_in_hand(int(option_data[2]))
                    if card_in_hand.get_shields() < minimum_shield_value_to_live or card_in_hand.get_card_name() == "Indomitable":
                        removed_options.append(active_options[i])
        if len(removed_options) < len(active_options):  # Never remove all of the options
            for i in range(len(removed_options)):
                game.active_options.remove(removed_options[i])
    return None

def filter_retreat_turn(game, player, enemy_player):
    icons_red_enemy, icons_blue_enemy, icons_green_enemy = enemy_player.count_icons_victory_display()
    icon_red_first, icon_blue_first, icon_green_first = game.get_colors_first_planet()
    icons_red_pot_enemy = icon_red_first + icons_red_enemy
    icons_blue_pot_enemy = icon_blue_first + icons_blue_enemy
    icons_green_pot_enemy = icon_green_first + icons_green_enemy
    print("retreat filter")
    print(game.targeted_planet)
    print(game.round_number)
    if game.targeted_planet == game.round_number - 1:
        print("first planet, filter retreat")
        if icons_red_pot_enemy == 3 or icons_blue_pot_enemy == 3 or icons_green_pot_enemy == 3:
            print("prevent retreat")
            game.active_options = ["pass-P1"]
            return None
        if player.count_attack_at_planet(game.targeted_planet) > enemy_player.count_attack_at_planet(game.targeted_planet):
            print("prevent retreat")
            game.active_options = ["pass-P1"]
            return None
    elif game.targeted_planet == game.round_number:
        print("next planet filtering")
        if player.count_attack_at_planet(game.targeted_planet) > enemy_player.count_attack_at_planet(game.targeted_planet):
            print("prevent retreat")
            game.active_options = ["pass-P1"]
            return None
    else:
        active_options = game.get_active_options()
        removed_options = []
        for i in range(len(active_options)):
            option_data = active_options[i].split(sep="/")
            if len(option_data) == 4:
                planet_pos = int(option_data[2])
                unit_pos = int(option_data[3])
                if player.get_card_given_pos(planet_pos, unit_pos).get_card_type() == "Warlord":
                    removed_options.append(active_options[i])
        if len(removed_options) < len(active_options):  # Never remove all of the options
            for i in range(len(removed_options)):
                game.active_options.remove(removed_options[i])
        return None
    return None


def filter_choices(game, player, enemy_player):
    print(game.choice_context)
    if game.choice_context == "Resolve Battle Ability?":
        if game.get_targeted_planet_name() != "Y'varn":
            game.active_options = ["CHOICE/0"]
            return None
    if game.choice_context == "Earth Caste Technician":
        game.active_options = ["CHOICE/0"]
        return None
    if game.choice_context == "Promethium Mine":
        game.active_options = ["CHOICE/0"]
        return None
    if game.choice_context == "Captain Cato Sicarius":
        game.active_options = ["CHOICE/0"]
        return None
    if game.choice_context == "Retreat Warlord?":
        if game.targeted_planet != 7:
            if player.warlord_is_bloodied:
                return None
            if enemy_player.count_attack_at_planet(game.targeted_planet) < 3:
                game.active_options = ["CHOICE/1"]
                return None


def filter_bad_deploy_locations(game, player, enemy_player):
    card = player.get_selected_card_hand()
    if card is None:
        return None
    icons_red_enemy, icons_blue_enemy, icons_green_enemy = enemy_player.count_icons_victory_display()
    icon_red_first, icon_blue_first, icon_green_first = game.get_colors_first_planet()
    icons_red_pot_enemy = icon_red_first + icons_red_enemy
    icons_blue_pot_enemy = icon_blue_first + icons_blue_enemy
    icons_green_pot_enemy = icon_green_first + icons_green_enemy
    print("filter deploy locs")
    if card.get_card_type() == "Attachment":
        print("filter attachment locs")
        active_options = game.get_active_options()
        removed_options = []
        for i in range(len(active_options)):
            option_data = active_options[i].split(sep="/")
            if len(option_data) > 1:
                if player.number != int(option_data[1]):
                    removed_options.append(active_options[i])
        if len(removed_options) < len(active_options):  # Never remove all of the options
            for i in range(len(removed_options)):
                game.active_options.remove(removed_options[i])
    if icons_red_pot_enemy == 3 or icons_blue_pot_enemy == 3 or icons_green_pot_enemy == 3:
        # Enemy wins if captures first planet.
        if card.get_attack() == 0:
            return None
        victory_planet_choice = "PLANETS/" + str(game.round_number - 1)
        if victory_planet_choice in game.active_options:
            game.active_options = [victory_planet_choice]
        return None
    if game.round_number > 3:
        return None
    if card.get_command() >= card.get_cost():
        # filter out locations where deploying the command unit has no effect on command state.
        active_options = game.get_active_options()
        removed_options = []
        print("filter command locs")
        for i in range(len(active_options)):
            option_data = active_options[i].split(sep="/")
            if len(option_data) == 2:
                if option_data[0] == "PLANETS":
                    planet_num = int(option_data[1])
                    own_command = player.count_command_at_planet(planet_num)
                    enemy_command = enemy_player.count_command_at_planet(planet_num)
                    if own_command > enemy_command:
                        removed_options.append(active_options[i])
        if len(removed_options) < len(active_options):  # Never remove all of the options
            for i in range(len(removed_options)):
                game.active_options.remove(removed_options[i])
        return None
    return None
