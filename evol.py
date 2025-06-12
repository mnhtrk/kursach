import pygame
from gym import Env
import numpy as np
import random

from params import *


random.seed(RANDOM_SEED)


class Cell():
    def __init__(self):
        self.conc = 0
        self.color = 0
        self.bot_color = 0
        self.dead_bots = 0


class Agent():
    def __init__(self, x, y, weights, color):
        self.x = x
        self.y = y
        self.weights = weights
        self.color = color
        self.observation = []
        self.hp = HP
        self.score = 0
        self.total_score = 0
        self.alive = True

    # значения агента, которые подаются в нейронку
    def set_obs(self, field, population):
        # расстояние до ближайшей клетки другого цвета / не полностью заполненной клетки такого же цвета по прямой
        dist_up, dist_right, dist_down, dist_left = 0, 0, 0, 0
        for i in range(self.y - 1, -1, -1):
            if ((field[i][self.x].color != self.color or
                field[i][self.x].color == self.color and field[i][self.x].conc < CONC) and
                field[i][self.x].color != 5 and field[i][self.x].color != 6):
                dist_up = self.norm_obs(self.y - i)
                break

        for i in range(self.x + 1, SIZE):
            if ((field[self.y][i].color != self.color or
                field[self.y][i].color == self.color and field[self.y][i].conc < CONC) and
                field[self.y][i].color != 5 and field[self.y][i].color != 6):
                dist_right = self.norm_obs(i - self.x)
                break

        for i in range(self.y + 1, SIZE):
            if ((field[i][self.x].color != self.color or
                field[i][self.x].color == self.color and field[i][self.x].conc < CONC) and
                field[i][self.x].color != 5 and field[i][self.x].color != 6):
                dist_down = self.norm_obs(i - self.y)
                break

        for i in range(self.x - 1, -1, -1):
            if ((field[self.y][i].color != self.color or
                field[self.y][i].color == self.color and field[self.y][i].conc < CONC) and
                field[self.y][i].color != 5 and field[self.y][i].color != 6):
                dist_left = self.norm_obs(self.x - i)
                break

        # расстояние до ближайшей клетки другого цвета / не полностью заполненной клетки такого же цвета по диагонали
        dist_up_right, dist_down_right, dist_down_left, dist_up_left = 0, 0, 0, 0
        for i in range(1, SIZE):
            if self.y - i == -1 or self.x + i == SIZE:
                break
            elif ((field[self.y - i][self.x + i].color != self.color or
                field[self.y - i][self.x + i].color == self.color and field[self.y - i][self.x + i].conc < CONC) and
                field[self.y - i][self.x + i].color != 5 and field[self.y - i][self.x + i].color != 6):
                dist_up_right = self.norm_obs(i)
                break

        for i in range(1, SIZE):
            if self.y + i == SIZE or self.x + i == SIZE:
                break
            elif ((field[self.y + i][self.x + i].color != self.color or
                field[self.y + i][self.x + i].color == self.color and field[self.y + i][self.x + i].conc < CONC) and
                field[self.y + i][self.x + i].color != 5 and field[self.y + i][self.x + i].color != 6):
                dist_down_right = self.norm_obs(i)
                break

        for i in range(1, SIZE):
            if self.y + i == SIZE or self.x - i == -1:
                break
            elif ((field[self.y + i][self.x - i].color != self.color or
                field[self.y + i][self.x - i].color == self.color and field[self.y + i][self.x - i].conc < CONC) and
                field[self.y + i][self.x - i].color != 5 and field[self.y + i][self.x - i].color != 6):
                dist_down_left = self.norm_obs(i)
                break

        for i in range(1, SIZE):
            if self.y - i == -1 or self.x - i == -1:
                break
            elif ((field[self.y - i][self.x - i].color != self.color or
                field[self.y - i][self.x - i].color == self.color and field[self.y - i][self.x - i].conc < CONC) and
                field[self.y - i][self.x - i].color != 5 and field[self.y - i][self.x - i].color != 6):
                dist_up_left = self.norm_obs(i)
                break

        # расстояние до ближайшей клетки красного цвета по прямой и по диагонали
        dist_red_up, dist_red_right, dist_red_down, dist_red_left = 0, 0, 0, 0
        dist_red_up_right, dist_red_down_right, dist_red_down_left, dist_red_up_left = 0, 0, 0, 0
        if RED_FIELD:
            for i in range(self.y - 1, -1, -1):
                if field[i][self.x].color == 7:
                    dist_red_up = self.norm_obs(self.y - i)
                    break

            for i in range(self.x + 1, SIZE):
                if field[self.y][i].color == 7:
                    dist_red_right = self.norm_obs(i - self.x)
                    break

            for i in range(self.y + 1, SIZE):
                if field[i][self.x].color == 7:
                    dist_red_down = self.norm_obs(i - self.y)
                    break

            for i in range(self.x - 1, -1, -1):
                if field[self.y][i].color == 7:
                    dist_red_left = self.norm_obs(self.x - i)
                    break

            for i in range(1, SIZE):
                if self.y - i == -1 or self.x + i == SIZE:
                    break
                elif field[self.y - i][self.x + i].color == 7:
                    dist_red_up_right = self.norm_obs(i)
                    break

            for i in range(1, SIZE):
                if self.y + i == SIZE or self.x + i == SIZE:
                    break
                elif field[self.y + i][self.x + i].color == 7:
                    dist_red_down_right = self.norm_obs(i)
                    break

            for i in range(1, SIZE):
                if self.y + i == SIZE or self.x - i == -1:
                    break
                elif field[self.y + i][self.x - i].color == 7:
                    dist_red_down_left = self.norm_obs(i)
                    break

            for i in range(1, SIZE):
                if self.y - i == -1 or self.x - i == -1:
                    break
                elif field[self.y - i][self.x - i].color == 7:
                    dist_red_up_left = self.norm_obs(i)
                    break

        # расстояние до ближайшего мертвого агента такого же цвета по прямой
        teammate_dead_up, teammate_dead_right, teammate_dead_down, teammate_dead_left = 0, 0, 0, 0
        for agent in population:
            if not agent.alive:
                if agent.x == self.x and agent.y < self.y:
                    if self.norm_obs(self.y - agent.y) > teammate_dead_up or teammate_dead_up == 0:
                        teammate_dead_up = self.norm_obs(self.y - agent.y)
                if agent.x > self.x and agent.y == self.y:
                    if self.norm_obs(agent.x - self.x) > teammate_dead_right or teammate_dead_right == 0:
                        teammate_dead_right = self.norm_obs(agent.x - self.x)
                if agent.x == self.x and agent.y > self.y:
                    if self.norm_obs(agent.y - self.y) > teammate_dead_down or teammate_dead_down == 0:
                        teammate_dead_down = self.norm_obs(agent.y - self.y)
                if agent.x < self.x and agent.y == self.y:
                    if self.norm_obs(self.x - agent.x) > teammate_dead_left or teammate_dead_left == 0:
                        teammate_dead_left = self.norm_obs(self.x - agent.x)

        # расстояние до ближайшего живого агента такого же цвета по прямой
        teammate_alive_up, teammate_alive_right, teammate_alive_down, teammate_alive_left = 0, 0, 0, 0
        for i in range(self.y - 1, -1, -1):
            if field[i][self.x].bot_color == self.color:
                teammate_alive_up = self.norm_obs(self.y - i)
                break

        for i in range(self.x + 1, SIZE):
            if field[self.y][i].bot_color == self.color:
                teammate_alive_right = self.norm_obs(i - self.x)
                break

        for i in range(self.y + 1, SIZE):
            if field[i][self.x].bot_color == self.color:
                teammate_alive_down = self.norm_obs(i - self.y)
                break

        for i in range(self.x - 1, -1, -1):
            if field[self.y][i].bot_color == self.color:
                teammate_alive_left = self.norm_obs(self.x - i)
                break

        # расстояние до ближайшего живого агента такого же цвета по диагонали
        teammate_alive_up_right, teammate_alive_down_right, teammate_alive_down_left, teammate_alive_up_left = 0, 0, 0, 0
        for i in range(1, SIZE):
            if self.y - i == -1 or self.x + i == SIZE:
                break
            elif field[self.y - i][self.x + i].bot_color == self.color:
                teammate_alive_up_right = self.norm_obs(i)
                break

        for i in range(1, SIZE):
            if self.y + i == SIZE or self.x + i == SIZE:
                break
            elif field[self.y + i][self.x + i].bot_color == self.color:
                teammate_alive_down_right = self.norm_obs(i)
                break

        for i in range(1, SIZE):
            if self.y + i == SIZE or self.x - i == -1:
                break
            elif field[self.y + i][self.x - i].bot_color == self.color:
                teammate_alive_down_left = self.norm_obs(i)
                break

        for i in range(1, SIZE):
            if self.y - i == -1 or self.x - i == -1:
                break
            elif field[self.y - i][self.x - i].bot_color == self.color:
                teammate_alive_up_left = self.norm_obs(i)
                break

        # расстояние до ближайшего живого агента другого цвета по прямой
        enemy_alive_up, enemy_alive_right, enemy_alive_down, enemy_alive_left = 0, 0, 0, 0
        for i in range(self.y - 1, -1, -1):
            if field[i][self.x].bot_color != self.color:
                enemy_alive_up = self.norm_obs(self.y - i)
                break

        for i in range(self.x + 1, SIZE):
            if field[self.y][i].bot_color != self.color:
                enemy_alive_right = self.norm_obs(i - self.x)
                break

        for i in range(self.y + 1, SIZE):
            if field[i][self.x].bot_color != self.color:
                enemy_alive_down = self.norm_obs(i - self.y)
                break

        for i in range(self.x - 1, -1, -1):
            if field[self.y][i].bot_color != self.color:
                enemy_alive_left = self.norm_obs(self.x - i)
                break

        # расстояние до ближайшего живого агента другого цвета по диагонали
        enemy_alive_up_right, enemy_alive_down_right, enemy_alive_down_left, enemy_alive_up_left = 0, 0, 0, 0
        for i in range(1, SIZE):
            if self.y - i == -1 or self.x + i == SIZE:
                break
            elif field[self.y - i][self.x + i].bot_color != self.color:
                enemy_alive_up_right = self.norm_obs(i)
                break

        for i in range(1, SIZE):
            if self.y + i == SIZE or self.x + i == SIZE:
                break
            elif field[self.y + i][self.x + i].bot_color != self.color:
                enemy_alive_down_right = self.norm_obs(i)
                break

        for i in range(1, SIZE):
            if self.y + i == SIZE or self.x - i == -1:
                break
            elif field[self.y + i][self.x - i].bot_color != self.color:
                enemy_alive_down_left = self.norm_obs(i)
                break

        for i in range(1, SIZE):
            if self.y - i == -1 or self.x - i == -1:
                break
            elif field[self.y - i][self.x - i].bot_color != self.color:
                enemy_alive_up_left = self.norm_obs(i)
                break

        # расстояние до стен
        wall_up = self.norm_obs(self.y + 1)
        wall_right = self.norm_obs(SIZE - self.x)
        wall_down = self.norm_obs(SIZE - self.y)
        wall_left = self.norm_obs(self.x + 1)
        for i in range(self.y - 1, -1, -1):
            if field[i][self.x].color == 5 or field[i][self.x].color == 6:
                wall_up = self.norm_obs(self.y - i)
                break

        for i in range(self.x + 1, SIZE):
            if field[self.y][i].color == 5 or field[self.y][i].color == 6:
                wall_right = self.norm_obs(i - self.x)
                break

        for i in range(self.y + 1, SIZE):
            if field[i][self.x].color == 5 or field[i][self.x].color == 6:
                wall_down = self.norm_obs(i - self.y)
                break

        for i in range(self.x - 1, -1, -1):
            if field[self.y][i].color == 5 or field[self.y][i].color == 6:
                wall_left = self.norm_obs(self.x - i)
                break

        return np.array([dist_up, dist_right, dist_down, dist_left,
                         dist_up_right, dist_down_right, dist_down_left, dist_up_left,
                         dist_red_up, dist_red_right, dist_red_down, dist_red_left,
                         dist_red_up_right, dist_red_down_right, dist_red_down_left, dist_red_up_left,
                         teammate_dead_up, teammate_dead_right, teammate_dead_down, teammate_dead_left,
                         teammate_alive_up, teammate_alive_right, teammate_alive_down, teammate_alive_left,
                         teammate_alive_up_right, teammate_alive_down_right, teammate_alive_down_left, teammate_alive_up_left,
                         enemy_alive_up, enemy_alive_right, enemy_alive_down, enemy_alive_left,
                         enemy_alive_up_right, enemy_alive_down_right, enemy_alive_down_left, enemy_alive_up_left,
                         wall_up, wall_right, wall_down, wall_left])

    # нормализация значений
    def norm_obs(self, value):
        return 1 / value


class Evol(Env):
    def __init__(self, populations, network, length_chrom, render):
        self.window = None
        self.clock = None
        self.pix_square_size = int(WINDOW_SIZE / SIZE)

        self.field = []
        for i in range(SIZE):
            row = []
            for j in range(SIZE):
                row.append(Cell())
            self.field.append(row)

        self.gen_structs()

        if render:
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((WINDOW_SIZE, WINDOW_SIZE))
            self.clock = pygame.time.Clock()
            self.canvas = pygame.Surface((WINDOW_SIZE, WINDOW_SIZE))

        self.network = network

        self.deadly_walls = DEADLY_WALLS
        self.random_spawn = RANDOM_SPAWN

        self.teams = TEAMS
        self.population_size = POPULATION_SIZE
        self.hall_of_fame = HALL_OF_FAME
        self.length_chrom = length_chrom

        self.populations = []
        self.populations_alive = []

        spawn_cords = [[SIZE // 16, SIZE // 16 * 7, SIZE // 16, SIZE // 16 * 7],
                       [SIZE // 16 * 9, SIZE // 16 * 15, SIZE // 16, SIZE // 16 * 7],
                       [SIZE // 16, SIZE // 16 * 7, SIZE // 16 * 9, SIZE // 16 * 15],
                       [SIZE // 16 * 9, SIZE // 16 * 15, SIZE // 16 * 9, SIZE // 16 * 15]]
        for i in range(self.teams):
            agents = []
            for agent in populations[i]:
                if self.random_spawn:
                    x = random.randint(0, SIZE - 1)
                    y = random.randint(0, SIZE - 1)
                else:
                    x = random.randint(spawn_cords[i][0], spawn_cords[i][1])
                    y = random.randint(spawn_cords[i][2], spawn_cords[i][3])
                while self.spawn_check(x, y):
                    if self.random_spawn:
                        x = random.randint(0, SIZE - 1)
                        y = random.randint(0, SIZE - 1)
                    else:
                        x = random.randint(spawn_cords[i][0], spawn_cords[i][1])
                        y = random.randint(spawn_cords[i][2], spawn_cords[i][3])
                agents.append(Agent(x, y, agent, i + 1))
            self.populations.append(agents)
            self.populations_alive.append(self.population_size)

        for population in self.populations:
            for agent in population:
                self.change_cell_values(self.field[agent.y][agent.x], agent.color)

        self.info = []
        for i in range(self.teams):
            self.info.append(0)
            for agent in self.populations[i]:
                agent.observation = agent.set_obs(self.field, self.populations[i])


    # генерация структур
    def gen_structs(self):
        if STONES:
            for i in range(1, SIZE - 1):
                for j in range(1, SIZE - 1):
                    if random.uniform(0, 1) <= P_STONE:
                        self.field[i][j].color = 5
                    if ((self.field[i][j - 1].color == 5 or
                         self.field[i + 1][j].color == 5 or
                         self.field[i][j + 1].color == 5 or
                         self.field[i - 1][j].color == 5) and
                         random.uniform(0, 1) <= P_STONE_EXT):
                        self.field[i][j].color = 5
            for i in range(SIZE - 2, 0, - 1):
                for j in range(SIZE - 2, 0, - 1):
                    if ((self.field[i][j - 1].color == 5 or
                         self.field[i + 1][j].color == 5 or
                         self.field[i][j + 1].color == 5 or
                         self.field[i - 1][j].color == 5) and
                         random.uniform(0, 1) <= P_STONE_EXT):
                        self.field[i][j].color = 5

        if SECTOR_WALLS:
            for i in range(SIZE):
                for j in range(SIZE):
                    if (SIZE // 2 - SECTOR_WALLS_SIZE <= i <= SIZE // 2 + SECTOR_WALLS_SIZE or
                        SIZE // 2 - SECTOR_WALLS_SIZE <= j <= SIZE // 2 + SECTOR_WALLS_SIZE):
                        self.field[i][j].color = 6

        if RED_FIELD:
            side = RED_FIELD_SIZE // 2
            if RANDOM_RED_FIELD:
                red_fields = []
                for n in range(NUMBER_RED_FIELD):
                    x = random.randint(0, SIZE - 1)
                    y = random.randint(0, SIZE - 1)
                    red_fields.append([x, y])
                    for i in range(y - side - 1, y + side + 2):
                        for j in range(x - side - 1, x + side + 2):
                            if -1 < i < SIZE and -1 < j < SIZE:
                                if (j == x - side - 1 or j == x + side + 1 or
                                    i == y - side - 1 or i == y + side + 1):
                                    self.field[i][j].color = 6
                for n in range(NUMBER_RED_FIELD):
                    for i in range(red_fields[n][1] - side, red_fields[n][1] + side + 1):
                        for j in range(red_fields[n][0] - side, red_fields[n][0] + side + 1):
                            if -1 < i < SIZE and -1 < j < SIZE:
                                    self.field[i][j].color = 7
                                    self.field[i][j].conc = CONC
            else:
                x = SIZE // 2
                y = SIZE // 2
                for i in range(y - side - 1, y + side + 2):
                    for j in range(x - side - 1, x + side + 2):
                        if -1 < i < SIZE and -1 < j < SIZE:
                            if (j == x - side - 1 or j == x + side + 1 or
                                    i == y - side - 1 or i == y + side + 1):
                                self.field[i][j].color = 6
                            else:
                                self.field[i][j].color = 7
                                self.field[i][j].conc = CONC


    # коллизия при спавне
    def spawn_check(self, x, y):
        if 5 <= self.field[y][x].color <= 7:
            return True

        return False


    # количество закрашенных полей для всех комманд, количество живых агентов
    def get_info(self, prev_info = []):
        team_score = []
        done = False
        for population_color in range(1, self.teams + 1):
            score = 0
            for i in range(SIZE):
                for j in range(SIZE):
                    if self.field[i][j].color == population_color and self.field[i][j].conc == CONC:
                        score += 1
            team_score.append(score)

        for population in self.populations:
            for agent in population:
                agent.score = 0

        for i in range(self.teams):
            self.info[i] = [team_score[i], self.populations_alive[i]]

        if prev_info == self.info:
            done = True

        return self.info, done


    # возвращает пары значений веса / общий счет для каждого агента
    def get_weight_scores(self):
        weight_scores = []
        for population in self.populations:
            team_w_s = []
            for agent in population:
                team_w_s.append([agent.weights, agent.total_score])
            weight_scores.append(team_w_s)
        return weight_scores


    # возвращает веса для следующей игры
    # от каждой команды в зал славы отбираются n лучших особоей по итогам всей игры
    # 5% новой популяции победившей команды состоит из рандомных особей из лучших 25% особей зала славы
    # 5% новой популяции остальных команд состоят из рандомных особей из остальных 75% особей зала славы
    # остальная часть всех популяций заполняется с помощью турнирного отбора всех особей всех команд
    def get_next_game_populations(self, team_score_alive):
        team_score = []
        for i in range(self.teams):
            team_score.append(team_score_alive[i][0])
        team_score_sorted = team_score[:]
        team_score_sorted.sort(reverse=True)
        winner_team_ind = team_score.index(team_score_sorted[0])

        hall_of_fame = []
        for population in self.populations:
            team_best = []
            for agent in population:
                team_best.append([agent.weights, agent.total_score])
            team_best.sort(reverse=True, key=lambda x: x[1])
            team_best = team_best[:self.hall_of_fame]
            hall_of_fame += team_best

        new_populations = []

        old_weights = []
        for population in self.populations:
            for agent in population:
                old_weights.append([agent.weights, agent.total_score])

        for i in range(self.teams):
            new_population = []
            if i == winner_team_ind:
                transfered_agents = self.population_size // 20
                for j in range(transfered_agents):
                    new_population.append(hall_of_fame[random.randint(0, len(hall_of_fame) // 4 - 1)][0])
            else:
                transfered_agents = self.population_size // 20
                for j in range(transfered_agents):
                    new_population.append(hall_of_fame[random.randint(len(hall_of_fame) // 4, len(hall_of_fame) - 1)][0])

            for j in range(transfered_agents, self.population_size):
                new_ind = []
                for k in range(TOURN_SIZE):
                    new_ind_index = random.randint(0, len(old_weights) - 1)
                    new_ind.append(old_weights[new_ind_index])

                new_ind.sort(key=lambda x: x[1])
                new_population.append(new_ind[-1][0])

            new_populations.append(new_population)
        return new_populations


    # отрисовка окружения
    def render(self, time):
        for i in range(SIZE):
            for j in range(SIZE):
                if self.field[i][j].color != 0:
                    if self.field[i][j].color == 1:
                        color_cell = (200 // CONC * self.field[i][j].conc, 200 // CONC * self.field[i][j].conc, 0)
                    elif self.field[i][j].color == 2:
                        color_cell = (0, 200 // CONC * self.field[i][j].conc, 0)
                    elif self.field[i][j].color == 3:
                        color_cell = (0, 200 // CONC * self.field[i][j].conc, 200 // CONC * self.field[i][j].conc)
                    elif self.field[i][j].color == 4:
                        color_cell = (0, 0, 200 // CONC * self.field[i][j].conc)
                    elif self.field[i][j].color == 5:
                        color_cell = (255, 0, 127)
                    elif self.field[i][j].color == 6:
                        color_cell = (255, 102, 255)
                    elif self.field[i][j].color == 7:
                        color_cell = (200 // CONC * self.field[i][j].conc, 0, 0)
                    pygame.draw.rect(
                        self.canvas,
                        color_cell,
                        (j * self.pix_square_size, i * self.pix_square_size,
                         self.pix_square_size, self.pix_square_size),
                    )

        for population in self.populations:
            for agent in population:
                if agent.alive:
                    if agent.color == 1:
                        color_bot = (agent.hp * 255 // MAX_HP, agent.hp * 255 // MAX_HP, 0)
                    elif agent.color == 2:
                        color_bot = (0, agent.hp * 255 // MAX_HP, 0)
                    elif agent.color == 3:
                        color_bot = (0, agent.hp * 255 // MAX_HP, agent.hp * 255 // MAX_HP)
                    elif agent.color == 4:
                        color_bot = (0, 0, agent.hp * 255 // MAX_HP)
                else:
                    color_bot = (100, 100, 100)
                pygame.draw.rect(
                    self.canvas,
                    (255, 255, 255),
                    (agent.x * self.pix_square_size + 1, agent.y * self.pix_square_size + 1,
                     self.pix_square_size - 2, self.pix_square_size - 2),
                )
                pygame.draw.rect(
                    self.canvas,
                    color_bot,
                    (agent.x * self.pix_square_size + 2, agent.y * self.pix_square_size + 2,
                     self.pix_square_size - 4, self.pix_square_size - 4),
                )

        self.window.blit(self.canvas, self.canvas.get_rect())
        pygame.event.pump()
        pygame.display.update()
        self.clock.tick(time)


    # изменяет значения для клетки
    def change_cell_values(self, cell, bot_color):
        if cell.color == 0 and bot_color != 5 and bot_color != 0:
            cell.color = bot_color

        if bot_color != 0 and bot_color != 5:
            if cell.color != bot_color:
                cell.conc -= 1
                if cell.conc <= 0:
                    cell.color = bot_color
                    cell.conc = 1
            else:
                if cell.conc < CONC:
                    cell.conc += 1

        if cell.dead_bots > 0:
            cell.bot_color = 5
        else:
            cell.bot_color = bot_color


    # награды
    # +1 за атаку на вражеское поле, +2 за усиление своего поля, +5 за атаку на красное поле / -5 при максимальном хп,
    # еще +2, если при перекраске рядом находится союзник

    # +CONC за каждое перекрашенное вражеское поле при использовании скила

    # +20 за уничтожение врага, еще +20, если при уничтожении рядом находится союзник

    # +20 за воскрешение союзника

    # -100 за смерть
    def step(self, time, render):
        population_ind = 0
        for population in self.populations:
            for agent in population:
                if agent.alive:
                    self.network.set_weights(agent.weights)
                    action = self.network.predict(agent.observation)
                    action = np.where(action == np.max(action))[0][0]

                    self.change_cell_values(self.field[agent.y][agent.x], 0)

                    # передвижение и столкновение со стенами
                    if action == 0:
                        agent.hp -= 1
                        agent.y -= 1
                        if (agent.y == -1 or
                            self.field[agent.y][agent.x].color == 5 or
                            self.field[agent.y][agent.x].color == 6):
                            if agent.y != -1 and self.field[agent.y][agent.x].color == 6:
                                self.field[agent.y][agent.x].color = agent.color
                            agent.y += 1
                            if self.deadly_walls:
                                agent.alive = False
                                agent.hp = 0
                                self.field[agent.y][agent.x].dead_bots += 1
                                self.change_cell_values(self.field[agent.y][agent.x], 5)
                                self.populations_alive[population_ind] -= 1
                                agent.score -= 100
                                agent.total_score -= 100
                        elif self.field[agent.y][agent.x].bot_color == agent.color:
                            agent.y += 1
                    elif action == 1:
                        agent.hp -= 1
                        agent.x += 1
                        if (agent.x == SIZE or
                            self.field[agent.y][agent.x].color == 5 or
                            self.field[agent.y][agent.x].color == 6):
                            if agent.x != SIZE and self.field[agent.y][agent.x].color == 6:
                                self.field[agent.y][agent.x].color = agent.color
                            agent.x -= 1
                            if self.deadly_walls:
                                agent.alive = False
                                agent.hp = 0
                                self.field[agent.y][agent.x].dead_bots += 1
                                self.change_cell_values(self.field[agent.y][agent.x], 5)
                                self.populations_alive[population_ind] -= 1
                                agent.score -= 100
                                agent.total_score -= 100
                        elif self.field[agent.y][agent.x].bot_color == agent.color:
                            agent.x -= 1
                    elif action == 2:
                        agent.hp -= 1
                        agent.y += 1
                        if (agent.y == SIZE or
                            self.field[agent.y][agent.x].color == 5 or
                            self.field[agent.y][agent.x].color == 6):
                            if agent.y != SIZE and self.field[agent.y][agent.x].color == 6:
                                self.field[agent.y][agent.x].color = agent.color
                            agent.y -= 1
                            if self.deadly_walls:
                                agent.alive = False
                                agent.hp = 0
                                self.field[agent.y][agent.x].dead_bots += 1
                                self.change_cell_values(self.field[agent.y][agent.x], 5)
                                self.populations_alive[population_ind] -= 1
                                agent.score -= 100
                                agent.total_score -= 100
                        elif self.field[agent.y][agent.x].bot_color == agent.color:
                            agent.y -= 1
                    elif action == 3:
                        agent.hp -= 1
                        agent.x -= 1
                        if (agent.x == -1 or
                            self.field[agent.y][agent.x].color == 5 or
                            self.field[agent.y][agent.x].color == 6):
                            if agent.x != -1 and self.field[agent.y][agent.x].color == 6:
                                self.field[agent.y][agent.x].color = agent.color
                            agent.x += 1
                            if self.deadly_walls:
                                agent.alive = False
                                agent.hp = 0
                                self.field[agent.y][agent.x].dead_bots += 1
                                self.change_cell_values(self.field[agent.y][agent.x], 5)
                                self.populations_alive[population_ind] -= 1
                                agent.score -= 100
                                agent.total_score -= 100
                        elif self.field[agent.y][agent.x].bot_color == agent.color:
                            agent.x += 1
                    elif action == 4:
                        if agent.hp >= SKILL_COST:
                            circ_off = SKILL_RADIUS
                            for i in range(agent.y - SKILL_RADIUS, agent.y + SKILL_RADIUS + 1):
                                for j in range(agent.x - SKILL_RADIUS + abs(circ_off), agent.x + SKILL_RADIUS - abs(circ_off) + 1):
                                    if -1 < i < SIZE and -1 < j < SIZE:
                                        if self.field[i][j].color != 5 and self.field[i][j].color != 6:
                                            if self.field[i][j].color != agent.color and self.field[i][j].color != 7:
                                                agent.score += CONC
                                                agent.total_score += CONC
                                            self.field[i][j].color = agent.color
                                            self.field[i][j].conc = CONC
                                circ_off -= 1
                            agent.hp -= SKILL_COST
                            if agent.hp <= 0:
                                agent.alive = False
                                self.field[agent.y][agent.x].dead_bots += 1
                                self.change_cell_values(self.field[agent.y][agent.x], 5)
                                self.populations_alive[population_ind] -= 1
                        else:
                            agent.hp -= 1
                    elif action == 5:
                        agent.hp -= 1

                    if agent.alive:
                        # изменение цвета клетки
                        if (self.field[agent.y][agent.x].color != agent.color or
                            self.field[agent.y][agent.x].color == agent.color and self.field[agent.y][agent.x].conc < CONC):
                            teammate_near = False
                            for j in range(20, 28):
                                if agent.observation[j] == 1:
                                    teammate_near = True
                                    break

                            if self.field[agent.y][agent.x].color == 7:
                                agent.hp += 6
                                if agent.hp <= MAX_HP:
                                    agent.score += 5
                                    agent.total_score += 5
                                    if teammate_near:
                                        agent.score += 2
                                        agent.total_score += 2
                                else:
                                    agent.score -= 5
                                    agent.total_score -= 5
                            else:
                                agent.score += 1
                                agent.total_score += 1
                                agent.hp += 1
                                if self.field[agent.y][agent.x].color == agent.color and self.field[agent.y][agent.x].conc < CONC:
                                    agent.score += 1
                                    agent.total_score += 1
                                    agent.hp += 3
                                if teammate_near:
                                    agent.score += 2
                                    agent.total_score += 2

                            if agent.hp > MAX_HP:
                                agent.hp = MAX_HP


                        for i in range(self.teams):
                            # уничтожение агента другого цвета
                            if population_ind != i:
                                for enemy_agent in self.populations[i]:
                                    if enemy_agent.x == agent.x and enemy_agent.y == agent.y and enemy_agent.alive:
                                        enemy_agent.alive = False
                                        agent.hp += enemy_agent.hp
                                        if agent.hp > MAX_HP:
                                            agent.hp = MAX_HP
                                        enemy_agent.hp = 0
                                        self.field[agent.y][agent.x].dead_bots += 1
                                        self.change_cell_values(self.field[agent.y][agent.x], 5)
                                        enemy_agent.score -= 100
                                        enemy_agent.total_score -= 100
                                        self.populations_alive[i] -= 1

                                        agent.score += 20
                                        agent.total_score += 20

                                        teammate_near = False
                                        for j in range(20, 28):
                                            if agent.observation[j] == 1:
                                                teammate_near = True
                                                break

                                        if teammate_near:
                                            agent.score += 20
                                            agent.total_score += 20

                            # возрождение союзного агента
                            else:
                                for teammate in self.populations[i]:
                                    if teammate.x == agent.x and teammate.y == agent.y and not teammate.alive and agent.hp > 1:
                                        teammate.alive = True
                                        teammate.hp = agent.hp // 2
                                        agent.hp -= teammate.hp
                                        self.field[agent.y][agent.x].dead_bots -= 1
                                        agent.score += 20
                                        agent.total_score += 20
                                        self.populations_alive[i] += 1


                    if agent.hp <= 0 and agent.alive:
                        agent.alive = False
                        agent.hp = 0
                        self.field[agent.y][agent.x].dead_bots -= 1
                        self.change_cell_values(self.field[agent.y][agent.x], 5)
                        self.populations_alive[population_ind] -= 1
                        agent.score -= 100
                        agent.total_score -= 100


                    if agent.alive:
                        self.change_cell_values(self.field[agent.y][agent.x], agent.color)
                        agent.observation = agent.set_obs(self.field, population)
            population_ind += 1

        if render:
            self.render(time)

        done = False
        dead_teams = 0
        for team in self.populations_alive:
            if team == 0:
                dead_teams += 1
        if dead_teams >= self.teams - 1:
            done = True


        frame = []
        for i in range(SIZE):
            r = []
            for j in range(SIZE):
                c = ""
                c += str(self.field[i][j].color)
                c += str(self.field[i][j].conc)
                c += str(self.field[i][j].bot_color)
                r.append(c)
            frame.append(r)

        for population in self.populations:
            for agent in population:
                if agent.alive and frame[agent.y][agent.x][2] != 5 and len(frame[agent.y][agent.x]) == 3:
                    frame[agent.y][agent.x] += str(agent.hp)
        return done, frame


    def reset(self, populations, render):
        for row in self.field:
            for cell in row:
                cell.conc = 0
                cell.color = 0
                cell.bot_color = 0
                cell.dead_bots = 0

        self.gen_structs()

        if render:
            self.canvas.fill((0, 0, 0))
        for population in self.populations:
            for agent in population:
                agent.x = -1
                agent.y = -1
                agent.alive = True
                agent.hp = HP
                agent.score = 0
                agent.total_score = 0

        spawn_cords = [[SIZE // 16, SIZE // 16 * 7, SIZE // 16, SIZE // 16 * 7],
                       [SIZE // 16 * 9, SIZE // 16 * 15, SIZE // 16, SIZE // 16 * 7],
                       [SIZE // 16, SIZE // 16 * 7, SIZE // 16 * 9, SIZE // 16 * 15],
                       [SIZE // 16 * 9, SIZE // 16 * 15, SIZE // 16 * 9, SIZE // 16 * 15]]
        for i in range(self.teams):
            for weight_ind, agent in enumerate(self.populations[i]):
                if self.random_spawn:
                    x = random.randint(0, SIZE - 1)
                    y = random.randint(0, SIZE - 1)
                else:
                    x = random.randint(spawn_cords[i][0], spawn_cords[i][1])
                    y = random.randint(spawn_cords[i][2], spawn_cords[i][3])
                while self.spawn_check(x, y):
                    if self.random_spawn:
                        x = random.randint(0, SIZE - 1)
                        y = random.randint(0, SIZE - 1)
                    else:
                        x = random.randint(spawn_cords[i][0], spawn_cords[i][1])
                        y = random.randint(spawn_cords[i][2], spawn_cords[i][3])

                agent.x = x
                agent.y = y
                agent.weights = populations[i][weight_ind]

            self.populations_alive[i] = self.population_size

        for population in self.populations:
            for agent in population:
                self.change_cell_values(self.field[agent.y][agent.x], agent.color)

        self.info = []
        for i in range(self.teams):
            self.info.append([])
            for agent in self.populations[i]:
                agent.observation = agent.set_obs(self.field, self.populations[i])