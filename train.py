import pygame.event
from matplotlib import pyplot as plt
from prettytable import PrettyTable

import evol
import gen
import nn
from params import *

import random
import joblib

# нейронка
network = nn.NN(LAYERS)
LENGTH_CHROM = network.get_total_weights()  # количество генов в хромосоме

random.seed(RANDOM_SEED)

SIM_TIME = 100000  # скорость отрисовки окружения
RENDER = True  # рендер процесса обучения
TRAIN = True  # обучение модели / запуск симуляции с обученной моделью

selection_methods = [gen.selection_tourn, gen.selection_sus, gen.selection_ranked]
mate_methods = [gen.mate_bin, gen.mate_uni, gen.mate_blend]

if TRAIN:
    # создание популяций
    populations_weights = []
    for i in range(TEAMS):
        populations_weights.append(gen.create_population(LENGTH_CHROM))


    # запуск окружения
    env = evol.Evol(populations_weights, network, LENGTH_CHROM, RENDER)

    best = 0
    team_scores_total = []
    moves_per_game = []
    max_score_per_game = []
    avg_score_per_game = []
    recorded_games = []
    for i in range(GAMES):
        count = 0
        done = False
        painted_alive = []
        recorded_game = []
        while not done and count <= TIME_LIMIT:
            done, frame = env.step(SIM_TIME, RENDER)
            recorded_game.append(frame)
            if RENDER:
                for j in pygame.event.get():
                    if j.type == pygame.QUIT:
                        done = True
            count += 1
            if count % EVOLVE_STEPS == 0:
                selection_methods[SELECTION_METHOD](env.populations, env.populations_alive)
                mate_methods[CROSSOVER_METHOD](env.populations, LENGTH_CHROM, env.populations_alive)
                gen.mutate(env.populations, LENGTH_CHROM)

                if len(painted_alive) > 0:
                    info, done = env.get_info(painted_alive[-1])
                else:
                    info, done = env.get_info()
                painted_alive.append(info[:])

                th = ["Команда", "Процент закрашенных клеток", "Живых агентов"]
                table = PrettyTable(th)
                for team in range(TEAMS):
                    table.add_row([team + 1, round(painted_alive[-1][team][0] / SIZE / SIZE * 100, 2), painted_alive[-1][team][1]])
                print("Игра", i + 1, "Итерация", count // EVOLVE_STEPS)
                print(table)

        moves_per_game.append(len(recorded_game))

        if GAMES < 10:
            if i == 0 or i == GAMES:
                recorded_games.append(recorded_game)
        elif (i + 1) % (GAMES // 10) == 0 or i == 0:
            recorded_games.append(recorded_game)
        info, done = env.get_info()
        painted_alive.append(info[:])
        weights_scores = env.get_weight_scores()
        best = weights_scores[:]

        populations_weights = []
        for j in range(TEAMS):
            population = []
            for k in range(POPULATION_SIZE):
                population.append(best[j][k][0])
            populations_weights.append(population)

        joblib.dump(populations_weights, "sav/best_weights.sav")

        th = ["Команда", "Процент закрашенных клеток", "Живых агентов", "Максимальный счет среди агентов", "Средний счет в команде"]
        table = PrettyTable(th)
        team_scores = []
        max_scores = []
        avg_scores = []
        for team in range(TEAMS):
            scores = []
            team_scores.append(painted_alive[-1][team][0])
            for j in range(POPULATION_SIZE):
                scores.append(weights_scores[team][j][1])
            max_scores.append(max(scores))
            avg_scores.append(round(sum(scores) / len(scores), 2))
            table.add_row([team + 1, round(painted_alive[-1][team][0] / SIZE / SIZE * 100, 2), painted_alive[-1][team][1], max(scores), round(sum(scores) / len(scores), 2)])
        print("Игра", i + 1)
        print(table)
        team_scores_total.append(team_scores)
        max_score_per_game.append(max_scores)
        avg_score_per_game.append(avg_scores)
        new_populations = env.get_next_game_populations(painted_alive[-1], TOURN_SIZE)
        env.reset(new_populations, RENDER)

    joblib.dump(recorded_games, "sav/recorded_train_games.sav")
    joblib.dump(team_scores_total, "sav/train_scores.sav")
    joblib.dump(moves_per_game, "sav/moves_per_game.sav")
    joblib.dump(max_score_per_game, "sav/max_score_per_game.sav")
    joblib.dump(avg_score_per_game, "sav/avg_score_per_game.sav")

    # график обучения
    for i in range(TEAMS):
        total_cells = []
        for j in range(len(team_scores_total)):
            total_cells.append(team_scores_total[j][i])
        if i == 0:
            color = (1, 1, 0)
        elif i == 1:
            color = (0, 1, 0)
        elif i == 2:
            color = (0, 1, 1)
        elif i == 3:
            color = (0, 0, 1)
        plt.plot(total_cells, color=color)

    plt.xlabel('Игра')
    plt.ylabel('Количество закрашенных клеток')
    plt.ylim(top=(SIZE*SIZE))
    plt.show()

    # график кол-ва ходов
    plt.plot(moves_per_game, color=(1, 0, 0))
    plt.xlabel('Игра')
    plt.ylabel('Количество шагов')
    plt.ylim(top=(TIME_LIMIT))
    plt.show()

    # график макс. и сред. значения очков в команде
    for i in range(TEAMS):
        max_score = []
        for j in range(len(max_score_per_game)):
            max_score.append(max_score_per_game[j][i])
        if i == 0:
            color = (1, 1, 0)
        elif i == 1:
            color = (0, 1, 0)
        elif i == 2:
            color = (0, 1, 1)
        elif i == 3:
            color = (0, 0, 1)
        plt.plot(max_score, color=color)

    for i in range(TEAMS):
        avg_score = []
        for j in range(len(avg_score_per_game)):
            avg_score.append(avg_score_per_game[j][i])
        if i == 0:
            color = (0.5, 0.5, 0)
        elif i == 1:
            color = (0, 0.5, 0)
        elif i == 2:
            color = (0, 0.5, 0.5)
        elif i == 3:
            color = (0, 0, 0.5)
        plt.plot(avg_score, color=color)

    plt.xlabel('Игра')
    plt.ylabel('Количество очков')
    plt.show()

    env.close()

# выбор собственной модели, best_weights.sav - последняя полученная модель при обучении
populations_weights = joblib.load("sav/best_weights.sav")

env = evol.Evol(populations_weights, network, LENGTH_CHROM, True)
count = 0
done = False
painted_alive = []
while not done and count <= TIME_LIMIT:
    done, frame = env.step(SIM_TIME, True)
    for i in pygame.event.get():
        if i.type == pygame.QUIT:
            done = True
    count += 1
    if count % EVOLVE_STEPS == 0:
        if len(painted_alive) > 0:
            info, done = env.get_info(painted_alive[-1])
        else:
            info, done = env.get_info()
        painted_alive.append(info[:])
        th = ["Команда", "Процент закрашенных клеток", "Живых агентов"]
        table = PrettyTable(th)
        for team in range(TEAMS):
            table.add_row([team + 1, round(painted_alive[-1][team][0] / SIZE / SIZE * 100, 2), painted_alive[-1][team][1]])
        print("Ход", count)
        print(table)

info, done = env.get_info()
painted_alive.append(info[:])
th = ["Команда", "Процент закрашенных клеток", "Живых агентов"]
table = PrettyTable(th)
for team in range(TEAMS):
    table.add_row([team + 1, round(painted_alive[-1][team][0] / SIZE / SIZE * 100, 2), painted_alive[-1][team][1]])
print("Итог")
print(table)
env.close()

# график последней игры
for i in range(TEAMS):
    total_cells = []
    for j in range(len(painted_alive)):
        total_cells.append(painted_alive[j][i][0])
    if i == 0:
        color = (1, 1, 0)
    elif i == 1:
        color = (0, 1, 0)
    elif i == 2:
        color = (0, 1, 1)
    elif i == 3:
        color = (0, 0, 1)
    plt.plot(total_cells, color=color)

plt.xlabel('Этапы')
plt.ylabel('Количество закрашенных клеток')
plt.ylim(top=(SIZE*SIZE))
plt.show()
