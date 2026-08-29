import keras
import asyncio
import websockets
import requests
from GameObject import Game
import ast
import datetime
from model_definitions import create_model, model_make_choice
import os
import shutil
import DeckManipulation
import miniDeckbuilder
import traceback
import sys
from credentials import PASSWORD

from protocolSettings import ws_protocol, site_name, URL

real_site = False
recorded_game_results_file = "real_game_results_tracker.txt"
if not PASSWORD:
    raise ValueError("No password provided, please configure credentials.py")
if len(sys.argv) > 1:
    if sys.argv[1] == "iridial":
        real_site = True
if real_site:
    print("is real site")
    site_name = "www.iridial.net"
    http_protocol = "https"
    ws_protocol = "wss"
    URL = http_protocol + "://" + site_name + "/"
resp = requests.post(
    URL + "api/auth-token/",
    data={"username": "Conqueror", "password": PASSWORD},
)
print(resp)
token = resp.json()["token"]
cards_changed_each_time = 0
max_timeouts = 20
file_name_model = "trained_models/RevisedTransformerModel100.keras"
valid_deck_names = ["CatoChamp"]
shutil.copytree("default_decks", "decks", dirs_exist_ok=True)
if not os.path.exists("decks/CatoCore"):
    raise SystemError
if not os.path.exists("trained_models"):
    os.mkdir("trained_models")

auth_token_string = "?token=" + token
bot_name = "Conqueror"
lobby_url = ws_protocol + "://" + site_name + "/ws/play/" + auth_token_string
if not os.path.exists(file_name_model):
    raise FileNotFoundError
model = keras.models.load_model(file_name_model)

waiting_time = 2000  # milliseconds
seconds_timeout = 3

async def run_temp_connection(url, name_1, name_2):
    # code restructed since we have multiple games going at once i.e. no globals
    # could probably use a list but it gets annoying with handling reconnects
    # though i don't bother handling non-lobby reconnects anyway so mehhhh
    # TODO: Add reconnect logic
    print("Opening temporary connection to " + url)
    game = Game(name_1, name_2, production=True)
    num_times_a_timeout_occurred = 0
    game_finished = False
    done = False
    try:
        async with websockets.connect(url) as websocket:
            while not done:
                try:
                    message = await asyncio.wait_for(
                        websocket.recv(),
                        timeout=seconds_timeout
                    )
                except asyncio.TimeoutError:
                    if game.active_player == bot_name:
                        num_times_a_timeout_occurred += 1
                        if num_times_a_timeout_occurred > max_timeouts:
                            message = "CHAT_MESSAGE/I appear to be stuck in an infinite loop and cannot resolve it, so I must leave."
                            message = '{"message": \"' + message + '\"}'
                            await websocket.send(message)
                            break
                        else:
                            message = "CHAT_MESSAGE//force-send-auto-data"
                            message = '{"message": \"' + message + '\"}'
                            print("send time out message:", message)
                            await websocket.send(message)
                    continue  # Go back to listening
                message = ast.literal_eval(message)["message"]
                print("Received message:", message)
                if message == "server: An error has occurred on the server side. Your game may become unstable or unplayable.":
                    message = "CHAT_MESSAGE/I cannot fix errors, and must leave to avoid causing more problems."
                    message = '{"message": \"' + message + '\"}'
                    await websocket.send(message)
                    break
                else:
                    action_required = game.update_game_from_game_update_message(message)
                    if action_required:
                        if action_required == "FINISHED":
                            game_finished = True
                        elif action_required == "LOADDECK":
                            deck_name = DeckManipulation.choose_deck(valid_deck_names)
                            deck_content = miniDeckbuilder.load_adjust_deck(deck_name, cards_changed_each_time)
                            DeckManipulation.send_deck(bot_name, deck_content, custom_URL=URL)
                            message = "CHAT_MESSAGE//loaddeckbot/" + game.active_player + "/" + deck_name
                            message = '{"message": \"' + message + '\"}'
                            await websocket.send(message)
                        else:
                            if game.active_player == bot_name and not game_finished:
                                await asyncio.sleep(waiting_time / 1000)
                                outputs = model_make_choice(model, game, training=False, filter_moves=True)
                                choice = outputs["choice"]
                                if not real_site:
                                    if outputs["flag"] == "Multi":
                                        inputs = outputs["state"]
                                        if "HAND" in choice:
                                            import numpy as np
                                            np.save("temp_2_input.npy", inputs)
                                message = game.create_automated_choice_message(choice)
                                await websocket.send(message)
                            if game_finished:
                                with open(recorded_game_results_file, "a") as f:
                                    f.write(game.player_one.get_name() + ", " + game.player_two.get_name() + ", " + game.winner + "\n")
                                    f.close()
                                message = "CHAT_MESSAGE/GGs"
                                message = '{"message": \"' + message + '\"}'
                                await websocket.send(message)
                                done = True
    except Exception as e:
        print("error:", e)
        with open("errors.txt", "a") as f:
            f.write(traceback.format_exc())

async def handle_main_message(websocket, message):
    message = ast.literal_eval(message)["message"]
    print(message)
    if message.startswith("Create lobby/"):
        print("lobby created")
        split_message = message.split(sep="/")
        human_username = split_message[1]
        ai_opponent = "false"
        if len(split_message) > 8:
            ai_opponent = split_message[8]
        print(ai_opponent)
        if ai_opponent == "true":
            message = "Join lobby/" + human_username + "/" + ""
            message = '{"message": \"' + message + '\"}'
            await websocket.send(message)
    elif message.startswith("Move to game/"):
        split_message = message.split(sep="/")
        game_id = split_message[1]
        name_1 = split_message[2]
        name_2 = split_message[3]
        if name_2 == bot_name or name_1 == bot_name:
            game_url = ws_protocol + "://" + site_name + "/ws/play/" + game_id + "/" + auth_token_string

            task = asyncio.create_task(run_temp_connection(game_url, name_1, name_2))
            print("Create game connection to " + game_url)

async def connect_lobby(lobby_url):
    while True:
        try:
            async with websockets.connect(lobby_url) as websocket:
                async for message in websocket:
                    await handle_main_message(websocket, message)
        except websockets.ConnectionClosed:
            print("Lost connection, reconnecting in 30s")
            await asyncio.sleep(30)
        except OSError as e:
            print("Exception", e)
            await asyncio.sleep(300)


asyncio.run(connect_lobby(lobby_url))