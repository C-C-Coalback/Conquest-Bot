This repository contains all of the code for building a Conquest AI with a neural network. It does NOT contain the code for the environment used to train the network; please see [here](https://github.com/C-C-Coalback/Conquest-LCG-Site) for that repository.

# Installation Instructions

 Requires Python 3.13.1. You will need to install a local copy of iridial.net and all of its dependencies. Please see the README.md of iridial.net for further instructions on how to do this. Once you have installed iridial.net, continue with these instructions.

 Open up a command terminal or your preferred IDE. Git clone the repository via
 ~~~
git clone https://github.com/C-C-Coalback/Conquest-Bot ConquestBot
cd ConquestBot
~~~

Create a virtual environment of your choice and activate it. Then run ```pip install -r requirements.txt``` to install the python packages that this repository requires. Finally, check that the keras/model training is working correctly by running ```py test.py```. If you see a model summary and some sample training messages without errors, it should be working correctly. Ignore any warnings about missing GPUs; they are not required.

 Your set-up should now be working.

# Usage

 ## Set iridial.net running.

 You will need two command terminals open for this step, not including the one that will be required to set the automated opponent training/running. In the first terminal, run ```docker run --rm -p 6379:6379 redis:7``` to create the Redis environment for websockets. Leave this process running. In the second terminal, run ```daphne -b 127.0.0.1 -p 8000 conquest_site.asgi:application``` to set iridial.net running.

 Check that iridial.net is running by navigating to 127.0.0.1:8000 in your browser. If the page loads correctly, the website is running. Lastly, check that the websockets are working by clicking "play" in the navigation bar, and see if there is an "Anonymous" player in the lobby. If there is, then you are all set. If not, there is an issue with the websockets; there should be an error message in the terminal that will help you diagnose what the problem is.

 ## Run a training loop.

 Set iridial.net running if it is not already. Configure the main.py file to specify where to save the model, the number of games, what decks to use, whether and how many cards to change out between games, what type of model to train, and some additional settings such as maximum number of timeouts before exiting the game. Then open a third terminal and run ```py main.py```. After a small delay, you should see a large amount of data passing through the terminal, indicating that the model is now training. A backup of the model is created every 50 games. Come back later when the model is done training.

 It should be noted that the "simplified" setting must be kept as True. It was previously used to distinguish between the more complex model that used embedding layers for the card names, attachments, etc. and the final version that does not use these settings. It has been left in as I would like to return to this approach at some point, but for now this model type is unusable.

If you want to resume training a previous model, simply set the name of the model to the location of that model, and the existing model will be loaded and overwrite the newly-generated model before training begins.

If you are training the model on a Linux machine, you may encounter an issue where the websocket expires after 24 hours. To get around this issue, instead run ```sh run_main.sh```, which is a shell script that will resume the training after this happens. You will need to change the TARGET variable to the correct number of loops.

If you would like the training loop to save a new copies of the model instead of writing to a single backup, run the altMain.py script instead. This is identical to the main.py script except for this difference.

## Testing a trained model.

There are multiple ways to test how well a model is performing. The first is to test the model against an opponent making moves at random. Configure and run comparisonTest.py to achieve this.

If you would like to test the performance without models (for example, to check how strong the move filter is), run the comparisonTestNoModels.py script.

If you would like to test which of two models is stronger, run the comparisonTestDiffModels.py script.

## Playing against the model yourself.

First, you will need to create an account on your locally-hosted website for the model. When this is done, add the password to the credentials.py file, and specify the username of the model in Connector.py. Then run ```py Connector.py```. You will now be able to play against the AI by selecting "AI opponent" in the lobby, then clicking "Create Lobby". You should see the AI in the lobby, and it should then join your game.

## Deploying the model to iridial.net.

The model can be deployed to the actual iridial.net website by the same process as before, but running ```py Connector.py iridial``` instead. I recommend only doing this to check that it works; do not keep the AI running, as it will be very confusing for players if there are multiple AI opponents on the website.

# Existing Models

Some pre-trained models are provided. The best of these is the RevisedTransformerModel100, which has been trained for 100 games and uses the transformer block architecture.

The results of some of the matchups have been recorded in the text files found in the game_results folder. For each file, each line represents one game, containing three items: the name of the deck that the first model played, the name of the deck that the second model played, and the winner of the game. conqueror1 is always the first player, and conqueror2 is the second player. If the results are from tests against a non-model based opponent, then conqueror1 is the model being tested. If the results are from a test between two models, the file should usually specify in the title the models used, such as "simple_100_v_trans_250", in this case meaning that conqueror1 was playing a non-transformer based model trained for 100 games while conqueror2 was playing a transformer based model trained for 250 games.

# Contributions

If you have trained a model that you have found is superior to the currently trained model, please consider reaching out with a copy of your model. I will require documentation as to what changes were made, both training-wise as well as in the model layers. If there are significant changes, such as to the move filter, please create a branch so that I can more easily evaluate the changes.

I am accepting pull requests, either for new features, approaches or tests. Please do not create pull requests containing AI-generated code or documentation.