import random

RANDOM_SEED = 42
random.seed(RANDOM_SEED)


# случайная хромосома
def create_ind(length_chrom):
    return [random.uniform(-1.0, 1.0) for _ in range(length_chrom)]


# создание популяции
def create_population(population_size, length_chrom):
    population = []
    for ind in range(population_size):
        population.append(create_ind(length_chrom))

    return population


# турнирный отбор
def selection_tourn(populations, tourn_size):
    for population in populations:
        old_weights = []
        for agent in population:
            if agent.alive:
                old_weights.append([agent.weights, agent.score])

        for agent in population:
            if agent.alive:
                new_ind = []
                for j in range(tourn_size):
                    new_ind_index = random.randint(0, len(old_weights) - 1)
                    new_ind.append(old_weights[new_ind_index])

                new_ind.sort(key=lambda x: x[1])
                agent.weights = new_ind[-1][0]
                agent.score = 0


# двухточечное скрещивание
def mate_bin(populations, length_chrom, p_cross, populations_alive, teams, population_size):
    for population in range(teams):
        for agent1 in range(population_size):
            if random.uniform(0, 1) <= p_cross and populations[population][agent1].alive and populations_alive[population] >= 2:
                agent2 = random.randint(0, population_size - 1)
                while agent2 == agent1 or not populations[population][agent2].alive:
                    agent2 = random.randint(0, population_size - 1)
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


# мутация
def mutate(populations, length_chrom, p_mut):
    for population in populations:
        for agent in population:
            if random.uniform(0, 1) <= p_mut and agent.alive:
                mut_index = random.randint(0, length_chrom - 1)
                agent.weights[mut_index] = random.triangular(-1.0, 1.0, random.gauss(agent.weights[mut_index], 0.2))
