import unittest
import GameObject
import moveFilter

class TestMoveFilter(unittest.TestCase):
    def test_filter_pointless_command_deployment(self):
        game = GameObject.Game("1", "2", False)
        game.active_player = "1"
        game.active_options = ["PLANETS/0", "PLANETS/1", "PLANETS/2", "PLANETS/3", "PLANETS/4"]
        game.active_context = "Deploy Turn"
        game.player_one.add_card_to_hand("Rogue Trader")
        game.player_one.target_hand = 0
        moveFilter.filter_obvious_bad_moves(game)
        self.assertEqual(game.active_options, ["PLANETS/0", "PLANETS/1", "PLANETS/2", "PLANETS/3", "PLANETS/4"])
        game.player_one.add_card_to_planet("Void Pirate", 1)
        moveFilter.filter_obvious_bad_moves(game)
        self.assertEqual(game.active_options, ["PLANETS/0", "PLANETS/2", "PLANETS/3", "PLANETS/4"])
        for i in range(5):
            game.player_one.add_card_to_planet("Void Pirate", i)
        game.active_options = ["PLANETS/0", "PLANETS/1", "PLANETS/2", "PLANETS/3", "PLANETS/4"]
        moveFilter.filter_obvious_bad_moves(game)
        self.assertEqual(game.active_options, ["PLANETS/0", "PLANETS/1", "PLANETS/2", "PLANETS/3", "PLANETS/4"])
        game.player_two.add_card_to_planet("Void Pirate", 2)
        moveFilter.filter_obvious_bad_moves(game)
        self.assertEqual(game.active_options, ["PLANETS/2"])

    def test_filter_if_enemy_victory_planet(self):
        game = GameObject.Game("1", "2", False)
        game.active_player = "1"
        game.active_options = ["PLANETS/0", "PLANETS/1", "PLANETS/2", "PLANETS/3", "PLANETS/4"]
        game.active_context = "Deploy Turn"
        game.player_one.add_card_to_hand("Land Raider")
        game.player_one.target_hand = 0
        game.planets_in_play = ["Barlus" for _ in range(7)]
        game.player_two.victory_display = ["Barlus", "Barlus"]
        moveFilter.filter_obvious_bad_moves(game)
        self.assertEqual(game.active_options, ["PLANETS/0"])

    def test_filter_battle_ability_choice(self):
        game = GameObject.Game("1", "2", False)
        game.active_player = "1"
        game.active_options = ["CHOICE/0", "CHOICE/1"]
        game.active_context = "Choice"
        game.choice_context = "Resolve Battle Ability?"
        game.targeted_planet = 0
        game.planets_in_play = ["Barlus" for _ in range(7)]
        moveFilter.filter_obvious_bad_moves(game)
        self.assertEqual(game.active_options, ["CHOICE/0"])
        game.planets_in_play = ["Y'varn" for _ in range(7)]
        game.active_options = ["CHOICE/0", "CHOICE/1"]
        moveFilter.filter_obvious_bad_moves(game)
        self.assertEqual(game.active_options, ["CHOICE/0", "CHOICE/1"])

    def test_filter_warlord_retreating_if_no_threat(self):
        game = GameObject.Game("1", "2", False)
        game.active_player = "1"
        game.active_options = ["CHOICE/0", "CHOICE/1"]
        game.active_context = "Choice"
        game.choice_context = "Retreat Warlord?"
        game.targeted_planet = 0
        game.player_two.add_card_to_planet("Void Pirate", 0)
        moveFilter.filter_obvious_bad_moves(game)
        self.assertEqual(game.active_options, ["CHOICE/1"])
        game.active_options = ["CHOICE/0", "CHOICE/1"]
        game.player_two.add_card_to_planet("Blood Angels Veterans", 0)
        moveFilter.filter_obvious_bad_moves(game)
        self.assertEqual(game.active_options, ["CHOICE/0", "CHOICE/1"])

    def test_filter_retreat_turn_at_victory_planet(self):
        game = GameObject.Game("1", "2", False)
        game.active_player = "1"
        game.active_options = ["IN_PLAY/1/0/0", "IN_PLAY/1/0/1", "pass-P1"]
        game.active_context = "Retreat Turn"
        game.targeted_planet = 0
        game.planets_in_play = ["Barlus" for _ in range(7)]
        game.player_two.add_card_to_planet("10th Company Scout", 0)
        game.player_one.add_card_to_planet("10th Company Scout", 0)
        game.player_one.add_card_to_planet("Blood Angels Veterans", 0)
        game.player_two.victory_display = ["Barlus", "Barlus"]
        moveFilter.filter_obvious_bad_moves(game)
        self.assertEqual(game.active_options, ["pass-P1"])

    def test_filter_pointless_retreats(self):
        game = GameObject.Game("1", "2", False)
        game.active_player = "1"
        game.active_options = ["IN_PLAY/1/1/0", "IN_PLAY/1/1/1", "pass-P1"]
        game.active_context = "Retreat Turn"
        game.targeted_planet = 1
        game.planets_in_play = ["Barlus" for _ in range(7)]
        game.player_one.add_card_to_planet("10th Company Scout", 1)
        game.player_one.add_card_to_planet("Captain Cato Sicarius", 1)
        moveFilter.filter_obvious_bad_moves(game)
        self.assertEqual(game.active_options, ["IN_PLAY/1/1/0", "pass-P1"])

    def test_filter_one_shield_only(self):
        game = GameObject.Game("1", "2", False)
        game.active_player = "1"
        game.active_options = ["HAND/1/0", "HAND/1/1", "pass-P1"]
        game.active_context = "Damage"
        game.player_one.add_card_to_planet("10th Company Scout", 0, target="red", damage=1)
        game.player_one.add_card_to_hand("Ion Rifle")
        game.player_one.add_card_to_hand("Drop Pod Assault")
        game.damage_that_can_be_shielded = 1
        moveFilter.filter_obvious_bad_moves(game)
        self.assertEqual(game.active_options, ["HAND/1/0", "pass-P1"])
        game.active_options = ["HAND/1/0", "HAND/1/1", "pass-P1"]
        game.player_one.cards_in_play[0] = []
        game.player_one.add_card_to_planet("10th Company Scout", 0, target="red", damage=4)
        game.damage_that_can_be_shielded = 4
        moveFilter.filter_obvious_bad_moves(game)
        self.assertEqual(game.active_options, ["pass-P1"])


if __name__ == '__main__':
    unittest.main()