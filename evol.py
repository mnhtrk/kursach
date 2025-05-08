import pygame

from gym import Env
import numpy as np
import random

RANDOM_SEED = 42
random.seed(RANDOM_SEED)


class Agent():
    def __init__(self, x, y, weights, color):
        self.x = x
        self.y = y
        self.weights = weights
        self.color = color
        self.observation = []
        self.score = 0
        self.alive = True

    # значения агента, которые подаются в нейронку
    def set_obs(self, field, population):
        # расстояние до ближайшей клетки другого цвета по прямой
        dist_up, dist_right, dist_down, dist_left = 0, 0, 0, 0
        for i in range(self.y - 1, -1, -1):
            if field[i][self.x] != self.color:
                dist_up = self.norm_obs(self.y - i)
                break

        for i in range(self.x + 1, len(field)):
            if field[self.y][i] != self.color:
                dist_right = self.norm_obs(i - self.x)
                break

        for i in range(self.y + 1, len(field)):
            if field[i][self.x] != self.color:
                dist_down = self.norm_obs(i - self.y)
                break

        for i in range(self.x - 1, -1, -1):
            if field[self.y][i] != self.color:
                dist_left = self.norm_obs(self.x - i)
                break

        # расстояние до ближайшей клетки другого цвета по диагонали
        dist_up_right, dist_down_right, dist_down_left, dist_up_left = 0, 0, 0, 0
        for i in range(1, len(field)):
            if self.y - i == -1 or self.x + i == len(field):
                break
            elif field[self.y - i][self.x + i] != self.color:
                dist_up_right = self.norm_obs(i)
                break

        for i in range(1, len(field)):
            if self.y + i == len(field) or self.x + i == len(field):
                break
            elif field[self.y + i][self.x + i] != self.color:
                dist_down_right = self.norm_obs(i)
                break

        for i in range(1, len(field)):
            if self.y + i == len(field) or self.x - i == -1:
                break
            elif field[self.y + i][self.x - i] != self.color:
                dist_down_left = self.norm_obs(i)
                break

        for i in range(1, len(field)):
            if self.y - i == -1 or self.x - i == -1:
                break
            elif field[self.y - i][self.x - i] != self.color:
                dist_up_left = self.norm_obs(i)
                break

        # расстояние до ближайшего мертвого агента такого же цвета
        teammate_up, teammate_right, teammate_down, teammate_left = 0, 0, 0, 0
        for agent in population:
            if not agent.alive:
                if agent.x == self.x and agent.y < self.y:
                    teammate_up = self.norm_obs(self.y - agent.y)
                if agent.x > self.x and agent.y == self.y:
                    teammate_right = self.norm_obs(agent.x - self.x)
                if agent.x == self.x and agent.y > self.y:
                    teammate_down = self.norm_obs(agent.y - self.y)
                if agent.x < self.x and agent.y == self.y:
                    teammate_left = self.norm_obs(self.x - agent.x)

        # расстояние до стен
        wall_up = self.norm_obs(self.y + 1)
        wall_right = self.norm_obs(len(field) - self.x)
        wall_down = self.norm_obs(len(field) - self.y)
        wall_left = self.norm_obs(self.x + 1)

        return np.array([dist_up, dist_right, dist_down, dist_left,
                         dist_up_right, dist_down_right, dist_down_left, dist_up_left,
                         teammate_up, teammate_right, teammate_down, teammate_left,
                         wall_up, wall_right, wall_down, wall_left])

    # нормализация значений
    def norm_obs(self, value):
        return 1 / value


class Evol(Env):
    def __init__(self, populations, network, population_size, teams, random_spawn, deadly_walls):
        self.size = 128
        self.window_size = 768
        self.window = None
        self.clock = None
        self.pix_square_size = int(self.window_size / self.size)

        self.field = np.zeros((self.size, self.size))

        pygame.init()
        pygame.display.init()
        self.window = pygame.display.set_mode((self.window_size, self.window_size))
        self.clock = pygame.time.Clock()
        self.canvas = pygame.Surface((self.window_size, self.window_size))

        self.network = network

        self.deadly_walls = deadly_walls

        self.teams = teams
        self.population_size = population_size
        self.populations = []
        self.populations_alive = []
        spawn_cords = [[self.size // 16, self.size // 16 * 7, self.size // 16, self.size // 16 * 7],
                       [self.size // 16 * 9, self.size // 16 * 15, self.size // 16, self.size // 16 * 7],
                       [self.size // 16, self.size // 16 * 7, self.size // 16 * 9, self.size // 16 * 15],
                       [self.size // 16 * 9, self.size // 16 * 15, self.size // 16 * 9, self.size // 16 * 15]]
        for i in range(self.teams):
            agents = []
            for ind in populations[i]:
                if random_spawn:
                    x = random.randint(0, self.size - 1)
                    y = random.randint(0, self.size - 1)
                else:
                    x = random.randint(spawn_cords[i][0], spawn_cords[i][1])
                    y = random.randint(spawn_cords[i][2], spawn_cords[i][3])
                while self.cell_check(x, y):
                    if random_spawn:
                        x = random.randint(0, self.size - 1)
                        y = random.randint(0, self.size - 1)
                    else:
                        x = random.randint(spawn_cords[i][0], spawn_cords[i][1])
                        y = random.randint(spawn_cords[i][2], spawn_cords[i][3])

                agents.append(Agent(x, y, ind, i + 1))
            self.populations.append(agents)
            self.populations_alive.append(population_size)
        self.info = []
        for i in range(self.teams):
            self.info.append(0)
            for agent in self.populations[i]:
                agent.observation = agent.set_obs(self.field, self.populations[i])

    # коллизия с другими агентами при спавне
    def cell_check(self, x, y):
        for population in self.populations:
            for agent in population:
                if x == agent.x and y == agent.y:
                    return True

    # количество закрашенных полей для всех комманд, количество живых агентов
    def get_score(self):
        total_score = []
        for population_color in range(1, self.teams + 1):
            score = 0
            for i in range(self.size):
                for j in range(self.size):
                    if self.field[i][j] == population_color:
                        score += 1
            total_score.append(score)

        for i in range(self.teams):
            self.info[i] = [total_score[i], self.populations_alive[i]]

        return self.info

    # отрисовка окружения
    def render(self, time):
        for i in range(self.size):
            for j in range(self.size):
                if self.field[i][j] != 0:
                    if self.field[i][j] == 1:
                        color = (100, 0, 0)
                    elif self.field[i][j] == 2:
                        color = (0, 100, 0)
                    elif self.field[i][j] == 3:
                        color = (0, 0, 100)
                    elif self.field[i][j] == 4:
                        color = (100, 100, 0)
                    pygame.draw.rect(
                        self.canvas,
                        color,
                        (j * self.pix_square_size, i * self.pix_square_size,
                         self.pix_square_size, self.pix_square_size),
                    )

        for population in self.populations:
            for agent in population:
                if agent.alive:
                    if agent.color == 1:
                        color = (255, 0, 0)
                    elif agent.color == 2:
                        color = (0, 255, 0)
                    elif agent.color == 3:
                        color = (0, 0, 255)
                    elif agent.color == 4:
                        color = (255, 255, 0)
                else:
                    color = (100, 100, 100)
                pygame.draw.rect(
                    self.canvas,
                    color,
                    (agent.x * self.pix_square_size, agent.y * self.pix_square_size,
                     self.pix_square_size, self.pix_square_size),
                )

        self.window.blit(self.canvas, self.canvas.get_rect())
        pygame.event.pump()
        pygame.display.update()
        self.clock.tick(time)

    # ход
    def step(self, time):
        population_ind = 0
        for population in self.populations:
            for agent in population:
                if agent.alive:
                    self.network.set_weights(agent.weights)
                    action = self.network.predict(agent.observation)
                    action = np.where(action == np.max(action))[0][0]

                    # передвижение
                    if action == 0:
                        agent.y = agent.y - 1
                    elif action == 1:
                        agent.x = agent.x + 1
                    elif action == 2:
                        agent.y = agent.y + 1
                    elif action == 3:
                        agent.x = agent.x - 1

                    # столкновение со стеной
                    if agent.y == -1:
                        agent.y = 0
                        if self.deadly_walls:
                            agent.alive = False
                            self.populations_alive[population_ind] -= 1
                    elif agent.x == self.size:
                        agent.x = self.size - 1
                        if self.deadly_walls:
                            agent.alive = False
                            self.populations_alive[population_ind] -= 1
                    elif agent.y == self.size:
                        agent.y = self.size - 1
                        if self.deadly_walls:
                            agent.alive = False
                            self.populations_alive[population_ind] -= 1
                    elif agent.x == -1:
                        agent.x = 0
                        if self.deadly_walls:
                            agent.alive = False
                            self.populations_alive[population_ind] -= 1

                    if agent.alive:
                        # изменение цвета клетки
                        if self.field[agent.y, agent.x] != agent.color:
                            agent.score += 1
                        self.field[agent.y, agent.x] = agent.color

                        for i in range(self.teams):
                            # уничтожение агента другого цвета
                            if population_ind != i:
                                for enemy_agent in self.populations[i]:
                                    if enemy_agent.x == agent.x and enemy_agent.y == agent.y and enemy_agent.alive:
                                        enemy_agent.alive = False
                                        enemy_agent.score = 0
                                        self.populations_alive[i] -= 1
                            # возрождение союзного агента
                            else:
                                for teammate in self.populations[i]:
                                    if teammate.x == agent.x and teammate.y == agent.y and not teammate.alive:
                                        teammate.alive = True
                                        agent.score += 2
                                        self.populations_alive[i] += 1

                    agent.observation = agent.set_obs(self.field, population)
            population_ind += 1

        self.render(time)
        done = False
        if len(set(self.field.reshape(self.size * self.size))) == 1 or \
                list(set(self.populations_alive))[0] == 0 and len(set(self.populations_alive)) == 1:
            done = True

        return done
