import keras

from model_definitions import create_model, model_make_choice, resolve_training
import GameObject

inputs_per_player = {1: [], 2: []}
action_idx_per_player = {1: [], 2: []}
value_est_per_player = {1: [], 2: []}
rewards_per_player = {1: [], 2: []}
rewards_stay_fix = {1: [], 2: []}

model = create_model(simplified=True)
model.summary()
optimizer = keras.optimizers.Adam(learning_rate=0.001)

game = GameObject.Game(name_1="conqueror1", name_2="conqueror2")
possible_actions = ["HAND/1/0", "IN_PLAY/1/0/0", "pass-P1", "HAND/1/1", "ATTACHMENT/IN_PLAY/1/0/0/0"]
game.update_game_from_game_update_message("GAME_INFO/AUTOMATED_DATA/context/conqueror1/" + "|||".join(possible_actions))
game.update_game_from_game_update_message("GAME_INFO/HAND/1/10th Company Scout||playable/Eager Recruit||playable")
game.update_game_from_game_update_message("GAME_INFO/SEARCH/conqueror1/DRAW/Honored Librarian/Eager Recruit/Indomitable")
game.update_game_from_game_update_message("GAME_INFO/IN_PLAY/1/0/conqueror1/10th Company Scout|R|0|0|H||Extra Attack (EOR): 2\nExtra Health (EOP): 3||Ion Rifle+R+R+")
game.player_one.resources = 7
game.player_two.resources = 4
game.initiative_holder = 1
game.update_game_from_game_update_message("GAME_INFO/HAND/2/Chaos Fanatics||playable/Eager Recruit||playable")
game.update_game_from_game_update_message("GAME_INFO/HQ/1/conqueror1/Eager Recruit|R|0|0|H||None|/Captain Cato Sicarius|R|0|0|H||None|")
game.update_game_from_game_update_message("GAME_INFO/IN_PLAY/1/0/conqueror1/Eager Recruit|R|0|0|H||None|")
game.update_game_from_game_update_message("GAME_INFO/VICTORY_DISPLAY/1/Barlus/Ferrin")
outputs = model_make_choice(model, game)

action_idx = outputs["action_idx"]
current_player = outputs["player_choosing"]
if outputs["flag"] == "Multi":
    inputs = outputs["state"]
    policy_probs = outputs["policy_probs"]
    value_estimate = outputs["critic_value"]

    inputs_per_player[current_player].append(inputs)
    action_idx_per_player[current_player].append(action_idx)
    value_est_per_player[current_player].append(value_estimate)
    rewards_per_player[current_player].append(0)
    rewards_stay_fix[current_player].append(False)

choice = outputs["choice"]

print(choice)

game.update_game_from_game_update_message("GAME_INFO/VICTORY_MESSAGE/conqueror1/Concession")

resolve_training(model, game, optimizer, inputs_per_player, action_idx_per_player, rewards_per_player, rewards_stay_fix, game_completed=False)
resolve_training(model, game, optimizer, inputs_per_player, action_idx_per_player, rewards_per_player, rewards_stay_fix, game_completed=True)
