from conquestdb_data import df

valid_df = df[(df["card type"].isin(["Army", "Event", "Attachment", "Support"]))]
valid_df = valid_df[(valid_df["loyalty"] != "Signature")]
card_names = valid_df.index.to_list()
weights_main_faction = {}
weights_ally_faction = {}
for card_name in card_names:
    weights_main_faction[card_name] = 1
    weights_ally_faction[card_name] = 1

# Space Marines weights
weights_main_faction["Drop Pod Assault"] = 5
weights_main_faction["Indomitable"] = 5
weights_main_faction["Iron Halo"] = 2

# Astra Militarum weights
weights_main_faction["Preemptive Barrage"] = 5
weights_main_faction["Suppressive Fire"] = 3
weights_ally_faction["Suppressive Fire"] = 3
weights_main_faction["Bodyguard"] = 2

# Orks weights
weights_main_faction["Battle Cry"] = 5

# Chaos weights
weights_main_faction["Tzeentch's Firestorm"] = 5

# Dark Eldar weights
weights_main_faction["Incubus Warrior"] = 3
weights_ally_faction["Incubus Warrior"] = 3
weights_main_faction["Murder of Razorwings"] = 2
weights_main_faction["Archon's Terror"] = 5
weights_ally_faction["Archon's Terror"] = 5
weights_main_faction["Raid"] = 5
weights_main_faction["Suffering"] = 5

# Eldar weights
weights_main_faction["Nullify"] = 5
weights_main_faction["Gift of Isha"] = 5
weights_ally_faction["Nullify"] = 0
weights_main_faction["Biel-Tan Guardians"] = 5
weights_main_faction["Soaring Falcon"] = 3
weights_ally_faction["Soaring Falcon"] = 3

# Tau weights
weights_main_faction["Recon Drone"] = 5
weights_main_faction["Vash'ya Trailblazer"] = 3
weights_ally_faction["Vash'ya Trailblazer"] = 3
weights_main_faction["Vior'la Marksman"] = 2
weights_ally_faction["Vior'la Marksman"] = 2
weights_main_faction["Earth Caste Technician"] = 3
weights_ally_faction["Earth Caste Technician"] = 3
weights_main_faction["Deception"] = 5
weights_main_faction["Repulsor Impact Field"] = 5
weights_main_faction["Ion Rifle"] = 3
weights_ally_faction["Ion Rifle"] = 2

# Neutral weights
weights_main_faction["Void Pirate"] = 3
weights_main_faction["Rogue Trader"] = 3
weights_main_faction["Promotion"] = 3