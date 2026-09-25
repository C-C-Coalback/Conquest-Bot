from discord import SyncWebhook
from scipy.stats import binomtest
from discordhook import link
if not link:
    raise ImportError("Discord Link Not Sent")
webhook = SyncWebhook.from_url(link)

def greetings():
    webhook.send("Hello World")

def model_completed_training(model_settings):
    model_name = model_settings["Model Name"].replace("trained_models/", "")
    games_trained = str(model_settings["Games Trained"])
    transformer = "Transformer" if model_settings["Transformer"] else "Standard FFN"
    eog_training = model_settings["EOG Rewards"]
    eor_training = model_settings["EOR Rewards"]
    custom_training = model_settings["Custom Move Rewards"]
    eor_rounds = str(model_settings["EOR Rounds"])
    trained_decks = ""
    notes = model_settings["Notes"]
    text = "Model Completed Training.\nDetails:\n"
    text = text + "Model Name: " + model_name + "\n"
    text = text + "Games Trained: " + games_trained + "\n"
    text = text + "Model Type: " + transformer + "\n"
    num_reward_types = eog_training + eor_training + custom_training
    reward_string = ""
    if eog_training:
        reward_string = reward_string + "EOG"
        if num_reward_types > 1:
            reward_string = reward_string + ", "
        num_reward_types = num_reward_types - 1
    if eor_training:
        reward_string = reward_string + "EOR (" + eor_rounds + " rounds)"
        if num_reward_types > 1:
            reward_string = reward_string + ", "
        num_reward_types = num_reward_types - 1
    if custom_training:
        reward_string = reward_string + "Custom"
    reward_string = reward_string + "\n"
    text = text + "Reward Types: " + reward_string
    text = text + "Notes: " + notes
    webhook.send(text)

def model_completed_testing(testing_environment):
    model_name = testing_environment["Model Name"].replace("trained_models/", "")
    filter_first = testing_environment["Filter First"]
    opponent = testing_environment["Opponent"].replace("trained_models/", "")
    filter_second = testing_environment["Filter Second"]
    file_results = testing_environment["File"]

    with open(file_results, "r") as f:
        content = f.read()
    conq1_wins = content.count("conqueror1")
    conq2_wins = content.count("conqueror2")
    total_wins = conq1_wins + conq2_wins
    text = "Model Completed Comparison Test.\nDetails:\n"
    text = text + "Model Name: " + model_name
    if filter_first:
        text = text + " (+ filter)"
    text = text + "\n"
    text = text + "Opponent: " + opponent
    if filter_second:
        text = text + " (+ filter)"
    text = text + "\n"
    text = text + "Games Played: " + str(total_wins) + "\n"
    text = text + "Results: " + str(conq1_wins) + "-" + str(conq2_wins) + "\n"
    result = binomtest(conq1_wins, total_wins, 0.5, alternative="greater")
    text = text + "p-value: " + str(result.pvalue)
    webhook.send(text)
