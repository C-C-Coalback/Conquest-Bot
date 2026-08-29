import requests
from protocolSettings import URL
import random

def send_deck(username, deck_text, custom_URL=""):
    data = {
        "name": username,
        "deck_text": deck_text
    }
    if custom_URL == "":
        custom_URL = URL
    response = requests.post(custom_URL + "api/send_deck_text/", data=data)
    print(response)
    if response.status_code == 200:
        received_data = response.json()
        print(received_data)


def request_deck(deck_name):
    data = {
        "name": "conqueror",
        "deck_name": deck_name
    }

    response = requests.post(URL + "api/request_deck/", data=data)
    print(response)
    if response.status_code == 200:
        received_data = response.json()
        print(received_data)


def choose_deck(decks):
    return random.choice(decks)
