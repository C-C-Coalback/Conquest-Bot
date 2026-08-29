


VALUE_CARDS = 1.25
VALUE_RESOURCES = 1
VALUE_DAMAGE = -0.1
VALUE_DAMAGE_WARLORD = -0.2
VALUE_BLOODIED_WARLORD = -5
VALUE_WINNING_COMMAND_AT_PLANET = (VALUE_CARDS, VALUE_RESOURCES)  # Winning command at these planets.
VALUE_BASE_CAPTURED_PLANET = 3  # Prevents model from completely ignoring first planet.
VALUE_ICONS_VICTORY_DISPLAY = 1  # Should be replaced with an estimate of how good each icon is in relation to winning the current planet flop
VALUE_UNITS_IN_PLAY = VALUE_RESOURCES * 1.25  # Value of units in play will be more than their costs to encourage playing cards.
VALUE_UNITS_IN_HQ = VALUE_UNITS_IN_PLAY * 0.75  # Units in HQ less valuable than units in play
VALUE_BASE_ATTACHMENTS = 1
VALUE_COST_ATTACHMENTS = 1
REWARD_EOR_FACTOR = 0.05
