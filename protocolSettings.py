bot_room_name = "conqueror"
bot_deck_name = "ConquerorStraken1"
training = True

site_name = "127.0.0.1:8000"
http_protocol = "http"
ws_protocol = "ws"

if site_name == "www.iridial.net":
    http_protocol = "https"
    ws_protocol = "wss"

URL = http_protocol + "://" + site_name + "/"

name_1 = "conqueror1"
name_2 = "conqueror2"