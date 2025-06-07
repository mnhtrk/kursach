import random
from params import RANDOM_SEED, POPULATION_SIZE, TOURN_SIZE, TEAMS, P_CROSSOVER, P_MUTATION, P_UNI_CROSSOVER

random.seed(RANDOM_SEED)

# случайная хромосома
def create_ind(length_chrom):
    return [random.uniform(-1.0, 1.0) for _ in range(length_chrom)]


# создание популяции
def create_population(length_chrom):
    population = []
    for ind in range(POPULATION_SIZE):
        population.append(create_ind(length_chrom))

    return population


# турнирный отбор
def selection_tourn(populations, populations_alive):
    for population in range(len(populations)):
        if populations_alive[population] > 0:
            old_weights = []
            for agent in populations[population]:
                if agent.alive:
                    old_weights.append([agent.weights, agent.score])

            for agent in populations[population]:
                if agent.alive:
                    new_ind = []
                    for j in range(TOURN_SIZE):
                        new_ind_index = random.randint(0, len(old_weights) - 1)
                        new_ind.append(old_weights[new_ind_index])

                    new_ind.sort(key=lambda x: x[1])
                    agent.weights = new_ind[-1][0]
                    agent.score = 0


# стохастическая универсальная выборка
def selection_sus(populations, populations_alive):
    for population in range(TEAMS):
        if populations_alive[population] > 0:
            old_weights = []
            for agent in populations[population]:
                if agent.alive:
                    old_weights.append([agent.weights, agent.score])

            old_weights.sort(key=lambda x: x[1])
            lowest_score = old_weights[0][1]
            sum_s = 0

            for i in range(len(old_weights)):
                old_weights[i][1] += abs(lowest_score) + 1
                sum_s += old_weights[i][1]

            distance = sum_s / len(old_weights)
            start = random.uniform(0, float(distance))
            points = [(start + i * distance) % sum_s for i in range(len(old_weights))]

            chosen = []
            for p in points:
                i = 0
                sum_ = old_weights[i][1]
                while sum_ < p:
                    i += 1
                    sum_ += old_weights[i][1]
                chosen.append(old_weights[i][0])

            weight_ind = 0
            for agent in populations[population]:
                if agent.alive:
                    agent.weights = chosen[weight_ind]
                    agent.score = 0
                    weight_ind += 1


# ранжированный отбор
def selection_ranked(populations, populations_alive):
    for population in range(len(populations)):
        if populations_alive[population] > 0:
            old_weights = []
            for agent in populations[population]:
                if agent.alive:
                    old_weights.append([agent.weights, agent.score])

            old_weights.sort(key=lambda x: x[1])
            sum_x = len(old_weights) * (len(old_weights) + 1) / 2
            points = [(i * (i + 1) / 2) * (100 / sum_x) for i in range(len(old_weights))]

            for agent in populations[population]:
                if agent.alive:
                    p = random.uniform(0, 100)
                    for i in range(len(points) - 1, -1, -1):
                        if p >= points[i]:
                            agent.weights = old_weights[i][0]
                            agent.score = 0
                            break



# двухточечное скрещивание
def mate_bin(populations, length_chrom, populations_alive):
    for population in range(TEAMS):
        for agent1 in range(POPULATION_SIZE):
            if random.uniform(0, 1) <= P_CROSSOVER and populations[population][agent1].alive and populations_alive[population] >= 2:
                agent2 = random.randint(0, POPULATION_SIZE - 1)
                while agent2 == agent1 or not populations[population][agent2].alive:
                    agent2 = random.randint(0, POPULATION_SIZE - 1)
                dot1 = random.randint(0, length_chrom - 3)
                dot2 = random.randint(dot1 + 1, length_chrom - 2)
                populations[population][agent1].weights = \
                    populations[population][agent1].weights[:dot1] + \
                    populations[population][agent2].weights[dot1:dot2] + \
                    populations[population][agent1].weights[dot2:]
                populations[population][agent2].weights = \
                    populations[population][agent2].weights[:dot1] + \
                    populations[population][agent1].weights[dot1:dot2] + \
                    populations[population][agent2].weights[dot2:]


# равномерное скрещивание
def mate_uni(populations, length_chrom, populations_alive):
    for population in range(TEAMS):
        for agent1 in range(POPULATION_SIZE):
            if random.uniform(0, 1) <= P_CROSSOVER and populations[population][agent1].alive and populations_alive[population] >= 2:
                agent2 = random.randint(0, POPULATION_SIZE - 1)
                while agent2 == agent1 or not populations[population][agent2].alive:
                    agent2 = random.randint(0, POPULATION_SIZE - 1)
                for i in range(length_chrom):
                    if random.uniform(0, 1) <= P_UNI_CROSSOVER:
                        temp = populations[population][agent1].weights[i]
                        populations[population][agent1].weights[i] = populations[population][agent2].weights[i]
                        populations[population][agent2].weights[i] = temp


# скрещивание смешением
def mate_blend(populations, length_chrom, populations_alive):
    for population in range(TEAMS):
        for agent1 in range(POPULATION_SIZE):
            if random.uniform(0, 1) <= P_CROSSOVER and populations[population][agent1].alive and populations_alive[population] >= 2:
                agent2 = random.randint(0, POPULATION_SIZE - 1)
                while agent2 == agent1 or not populations[population][agent2].alive:
                    agent2 = random.randint(0, POPULATION_SIZE - 1)
                for i in range(length_chrom):
                    temp_p1 = min(populations[population][agent1].weights[i], populations[population][agent2].weights[i])
                    temp_p2 = max(populations[population][agent1].weights[i], populations[population][agent2].weights[i])
                    p1 = temp_p1 - 0.5 * (temp_p2 - temp_p1)
                    p2 = temp_p1 + 0.5 * (temp_p2 - temp_p1)
                    if p1 < -1.0:
                        p1 = -1.0
                    if p2 > 1.0:
                        p2 = 1.0
                    populations[population][agent1].weights[i] = random.uniform(p1, p2)
                    populations[population][agent2].weights[i] = random.uniform(p1, p2)

# мутация
def mutate(populations, length_chrom):
    for population in populations:
        for agent in population:
            if random.uniform(0, 1) <= P_MUTATION and agent.alive:
                mut_index = random.randint(0, length_chrom - 1)
                agent.weights[mut_index] = random.triangular(-1.0, 1.0, random.gauss(agent.weights[mut_index], 0.2))
