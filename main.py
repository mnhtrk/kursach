import pygame.event
from matplotlib import pyplot as plt
from prettytable import PrettyTable

import evol
import gen
import nn

import random

# нейронка
INPUT_LAYER = 16  # количество входных значений
HIDDEN_LAYER = [12, 8]  # количество нейронов в скрытых слоях и количество скрытых слоев
                   # (например, [12, 6] - два скрытых слоя с 12 и 6 нейронами соответственно)
OUTPUT_LAYER = 4  # количество выходных значений
LAYERS = [INPUT_LAYER, HIDDEN_LAYER, OUTPUT_LAYER]
network = nn.NN(LAYERS)
LENGTH_CHROM = network.get_total_weights()  # количество генов в хромосоме

# ген. алгоритм
POPULATION_SIZE = 300  # количество агентов в команде
P_CROSSOVER = 0.8  # вероятность кроссинговера
P_MUTATION = 0.2  # вероятность мутации
TEAMS = 4  # до 4 команд
TOURN_SIZE = 3  # количество особоей для турнирного отбора
EVOLVE_STEPS = 15  # каждые n шагов срабатывает алгоритм эволюции

RANDOM_SEED = 42
random.seed(RANDOM_SEED)

SIM_TIME = 100000000  # скорость отрисовки окружения
RANDOM_SPAWN = True  # случайный спавн / спавн по секторам
DEADLY_WALLS = False  # смерть при столкновении со стенами


# создание популяций
populations_weights = []
for i in range(TEAMS):
    populations_weights.append(gen.create_population(POPULATION_SIZE, LENGTH_CHROM))


# запуск окружения
env = evol.Evol(populations_weights, network, POPULATION_SIZE, TEAMS, RANDOM_SPAWN, DEADLY_WALLS)
count = 0
done = False
painted_alive = []
while not done:
    done = env.step(SIM_TIME)
    for i in pygame.event.get():
        if i.type == pygame.QUIT:
            done = True
    count += 1
    if count % EVOLVE_STEPS == 0:
        gen.selection_tourn(env.populations, TOURN_SIZE)
        gen.mate_bin(env.populations, LENGTH_CHROM, P_CROSSOVER, env.populations_alive, TEAMS, POPULATION_SIZE)
        gen.mutate(env.populations, LENGTH_CHROM, P_MUTATION)
        info = env.get_score()
        painted_alive.append(info[:])
        th = ["Команда", "Процент закрашенных клеток", "Живых агентов"]
        table = PrettyTable(th)
        for team in range(TEAMS):
            table.add_row([team + 1, round(painted_alive[-1][team][0] / env.size / env.size * 100, 2), painted_alive[-1][team][1]])
        print("Итерация", count // EVOLVE_STEPS)
        print(table)

env.close()


# график
for i in range(TEAMS):
    total_cells = []
    for j in range(len(painted_alive)):
        total_cells.append(painted_alive[j][i][0])
    if i == 0:
        color = 'red'
    elif i == 1:
        color = 'green'
    elif i == 2:
        color = 'blue'
    elif i == 3:
        color = 'yellow'
    plt.plot(total_cells, color=color)

plt.xlabel('Поколение')
plt.ylabel('Количество закрашенных клеток')
plt.show()

env.close()
