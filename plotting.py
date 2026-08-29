import matplotlib.pyplot as plt

base_win_rate = 0.9
game_results_dir = "game_results"
files_simple = ["results_simple_" + str(25 * (i + 1)) + ".txt" for i in range(9)]
files_trans = ["results_trans_" + str(25 * (i + 1)) + ".txt" for i in range(9)]
x = [0] + [25 * (i + 1) for i in range(9)]
y_simple = [base_win_rate * 40]
y_trans = [base_win_rate * 40]
for i in range(len(files_simple)):
    with open(game_results_dir + "/" + files_simple[i], "r") as f:
        text = f.read()
    conq1_wins = text.count("conqueror1")
    conq2_wins = text.count("conqueror2")
    total_wins = conq1_wins + conq2_wins
    win_rate = conq1_wins / total_wins
    y_simple.append(conq1_wins)
for i in range(len(files_trans)):
    with open(game_results_dir + "/" + files_trans[i], "r") as f:
        text = f.read()
    conq1_wins = text.count("conqueror1")
    conq2_wins = text.count("conqueror2")
    total_wins = conq1_wins + conq2_wins
    if total_wins != 40:
        print("wrong number of games recorded")
    win_rate = conq1_wins / total_wins
    y_trans.append(conq1_wins)

plt.plot(x, y_simple, '-ro', label="Regular")
plt.plot(x, y_trans, '-bo', label='Transformer')
plt.title("Number of games won out of 40 after n games of training")
plt.xlabel("Games trained")
plt.ylabel("Games won")
plt.legend()
plt.show()