import asyncio
import websockets
import requests
from GameObject import Game
import ast
import datetime
import os
import shutil
import DeckManipulation
import miniDeckbuilder
import random
import moveFilter

from protocolSettings import bot_room_name, ws_protocol, site_name, training, URL, name_1, name_2


simplified = True
training = False
filter_moves = True
cards_changed_each_time = 0
num_games = 100
game_count = 0
initial_game_number = 0
num_times_a_timeout_occurred = 0
max_timeouts = 20
valid_deck_names = ["CatoChamp"]
recorded_game_results_file = "game_results/noomodelsfiltervnon.txt"
if not os.path.exists("decks"):
    shutil.copytree("default_decks", "decks")
if not os.path.exists("decks/CatoCore"):
    raise SystemError

data = {
    "name1": name_1,
    "name2": name_2,
    "id": bot_room_name,
    "private": False
}

response = requests.post(URL + "api/create_bot_room/", data=data)
print(response)
if response.status_code == 200:
    received_data = response.json()
    print(received_data)
    game_id = received_data["id"]
    game_url = ws_protocol + "://" + site_name + "/ws/play/" + game_id + "/"
    game = Game(name_1, name_2)
    game_finished = False
    performing_eor_model_training = False

    waiting_time = 0  # milliseconds
    start_time_all_games = datetime.datetime.now()
    start_time_current_game = start_time_all_games
    seconds_timeout = 3
    deck_1 = ""
    deck_2 = ""

    async def receive_messages(websocket, send_queue, done_event):
        global game_finished
        global waiting_time
        global game
        global game_count
        global num_times_a_timeout_occurred
        global start_time_current_game
        global performing_eor_model_training
        global deck_1, deck_2
        async for message in websocket:
            try:
                message = ast.literal_eval(message)["message"]
                print("Received message:", message)
                if "/slowdown " in message:
                    idx = message.index("/slowdown ")
                    end = idx + len("/slowdown ")
                    waiting_time = int(message[end:])
                if message == "server: An error has occurred on the server side. Your game may become unstable or unplayable.":
                    game_finished = True
                    message = "CHAT_MESSAGE//reset-game"
                    message = '{"message": \"' + message + '\"}'
                    await websocket.send(message)
                    num_times_a_timeout_occurred = 0
                    game_finished = False
                    game = Game(name_1, name_2)
                else:
                    action_required = game.update_game_from_game_update_message(message)
                    if action_required:
                        if num_times_a_timeout_occurred > max_timeouts:
                            game_finished = True
                            message = "CHAT_MESSAGE//reset-game"
                            message = '{"message": \"' + message + '\"}'
                            await websocket.send(message)
                            num_times_a_timeout_occurred = 0
                            game_finished = False
                            game = Game(name_1, name_2)
                        elif action_required == "EOR_TRAINING":
                            pass
                        elif action_required == "FINISHED":
                            game_finished = True
                            num_times_a_timeout_occurred = 0
                            message = "CHAT_MESSAGE//reset-game"
                            message = '{"message": \"' + message + '\"}'
                            await websocket.send(message)
                            game_count += 1
                            print("games completed:", game_count)
                            if game_count < 1000:
                                with open(recorded_game_results_file, "a") as f:
                                    f.write(deck_1 + ", " + deck_2 + ", " + game.winner + "\n")
                                    f.close()
                            if game_count < num_games:
                                game_finished = False
                                game = Game(name_1, name_2)
                            else:
                                await asyncio.sleep(0.2)
                                done_event.set()
                                await websocket.close()
                                return
                        elif action_required == "LOADDECK":
                            deck_name = DeckManipulation.choose_deck(valid_deck_names)
                            if game.active_player == game.player_one.name:
                                deck_1 = deck_name
                            else:
                                deck_2 = deck_name
                            deck_content = miniDeckbuilder.load_adjust_deck(deck_name, cards_changed_each_time)
                            DeckManipulation.send_deck(game.active_player, deck_content)
                            message = "CHAT_MESSAGE//loaddeckbot/" + game.active_player + "/" + deck_name
                            # message = "CHAT_MESSAGE//loadrandombot/" + game.active_player
                            message = '{"message": \"' + message + '\"}'
                            await send_queue.put(message)
                        elif not game_finished:
                            if game.active_player == game.player_one.name:
                                filter_moves = True
                            else:
                                filter_moves = False
                            if filter_moves:
                                moveFilter.filter_obvious_bad_moves(game)
                            possible_actions = game.get_active_options()
                            if not possible_actions:
                                game.active_options = ["pass-P1"]
                                possible_actions = ["pass-P1"]
                            choice = random.choice(possible_actions)
                            message = game.create_automated_choice_message(choice)
                            await send_queue.put(message)
            except Exception as e:
                print("error:", e)
                with open("errors.txt", "a") as f:
                    f.write(str(e) + "\n")


    async def send_messages(websocket, send_queue, done_event):
        global waiting_time
        global num_times_a_timeout_occurred
        global game_finished
        global performing_eor_model_training
        global seconds_timeout
        while not done_event.is_set():
            try:
                message = await asyncio.wait_for(send_queue.get(), timeout=seconds_timeout)
            except asyncio.TimeoutError:
                print("timed out")
                num_times_a_timeout_occurred += 1
                if not game_finished and not performing_eor_model_training:
                    message = "CHAT_MESSAGE//force-send-auto-data"
                    message = '{"message": \"' + message + '\"}'
                    print("send time out message:", message)
                    await websocket.send(message)
                performing_eor_model_training = False
                continue
            await asyncio.sleep(waiting_time / 1000)
            await websocket.send(message)
            send_queue.task_done()
            print("sent:", message)

    async def connect():
        send_queue = asyncio.Queue()
        done_event = asyncio.Event()
        async with websockets.connect(game_url) as websocket:
            await asyncio.gather(
                receive_messages(websocket, send_queue, done_event),
                send_messages(websocket, send_queue, done_event),
                return_exceptions=True
            )
    asyncio.run(connect())
    end_time_all_games = datetime.datetime.now()
    print("Time taken to play all " + str(num_games - initial_game_number) + " games: ", end_time_all_games - start_time_all_games)
