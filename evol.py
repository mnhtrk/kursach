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
        self.total_score = 0
        self.alive = True

    # значения агента, которые подаются в нейронку
    def set_obs(self, field, population):
        # расстояние до ближайшей клетки другого цвета по прямой
        dist_up, dist_right, dist_down, dist_left = 0, 0, 0, 0
        for i in range(self.y - 1, -1, -1):
            if abs(field[i][self.x]) != self.color:
                dist_up = self.norm_obs(self.y - i)
                break

        for i in range(self.x + 1, len(field)):
            if abs(field[self.y][i]) != self.color:
                dist_right = self.norm_obs(i - self.x)
                break

        for i in range(self.y + 1, len(field)):
            if abs(field[i][self.x]) != self.color:
                dist_down = self.norm_obs(i - self.y)
                break

        for i in range(self.x - 1, -1, -1):
            if abs(field[self.y][i]) != self.color:
                dist_left = self.norm_obs(self.x - i)
                break

        # расстояние до ближайшей клетки другого цвета по диагонали
        dist_up_right, dist_down_right, dist_down_left, dist_up_left = 0, 0, 0, 0
        for i in range(1, len(field)):
            if self.y - i == -1 or self.x + i == len(field):
                break
            elif abs(field[self.y - i][self.x + i]) != self.color:
                dist_up_right = self.norm_obs(i)
                break

        for i in range(1, len(field)):
            if self.y + i == len(field) or self.x + i == len(field):
                break
            elif abs(field[self.y + i][self.x + i]) != self.color:
                dist_down_right = self.norm_obs(i)
                break

        for i in range(1, len(field)):
            if self.y + i == len(field) or self.x - i == -1:
                break
            elif abs(field[self.y + i][self.x - i]) != self.color:
                dist_down_left = self.norm_obs(i)
                break

        for i in range(1, len(field)):
            if self.y - i == -1 or self.x - i == -1:
                break
            elif abs(field[self.y - i][self.x - i]) != self.color:
                dist_up_left = self.norm_obs(i)
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
            if field[i][self.x] == -self.color:
                teammate_alive_up = self.norm_obs(self.y - i)
                break

        for i in range(self.x + 1, len(field)):
            if field[self.y][i] == -self.color:
                teammate_alive_right = self.norm_obs(i - self.x)
                break

        for i in range(self.y + 1, len(field)):
            if field[i][self.x] == -self.color:
                teammate_alive_down = self.norm_obs(i - self.y)
                break

        for i in range(self.x - 1, -1, -1):
            if field[self.y][i] == -self.color:
                teammate_alive_left = self.norm_obs(self.x - i)
                break

        # расстояние до ближайшего живого агента такого же цвета по диагонали
        teammate_alive_up_right, teammate_alive_down_right, teammate_alive_down_left, teammate_alive_up_left = 0, 0, 0, 0
        for i in range(1, len(field)):
            if self.y - i == -1 or self.x + i == len(field):
                break
            elif field[self.y - i][self.x + i] == -self.color:
                teammate_alive_up_right = self.norm_obs(i)
                break

        for i in range(1, len(field)):
            if self.y + i == len(field) or self.x + i == len(field):
                break
            elif field[self.y + i][self.x + i] == -self.color:
                teammate_alive_down_right = self.norm_obs(i)
                break

        for i in range(1, len(field)):
            if self.y + i == len(field) or self.x - i == -1:
                break
            elif field[self.y + i][self.x - i] == -self.color:
                teammate_alive_down_left = self.norm_obs(i)
                break

        for i in range(1, len(field)):
            if self.y - i == -1 or self.x - i == -1:
                break
            elif field[self.y - i][self.x - i] == -self.color:
                teammate_alive_up_left = self.norm_obs(i)
                break

        # расстояние до ближайшего живого агента другого цвета по прямой
        enemy_alive_up, enemy_alive_right, enemy_alive_down, enemy_alive_left = 0, 0, 0, 0
        for i in range(self.y - 1, -1, -1):
            if field[i][self.x] != -self.color and field[i][self.x] < 0:
                enemy_alive_up = self.norm_obs(self.y - i)
                break

        for i in range(self.x + 1, len(field)):
            if field[self.y][i] != -self.color and field[self.y][i] < 0:
                enemy_alive_right = self.norm_obs(i - self.x)
                break

        for i in range(self.y + 1, len(field)):
            if field[i][self.x] != -self.color and field[i][self.x] < 0:
                enemy_alive_down = self.norm_obs(i - self.y)
                break

        for i in range(self.x - 1, -1, -1):
            if field[self.y][i] != -self.color and field[self.y][i] < 0:
                enemy_alive_left = self.norm_obs(self.x - i)
                break

        # расстояние до ближайшего живого агента другого цвета по диагонали
        enemy_alive_up_right, enemy_alive_down_right, enemy_alive_down_left, enemy_alive_up_left = 0, 0, 0, 0
        for i in range(1, len(field)):
            if self.y - i == -1 or self.x + i == len(field):
                break
            elif field[self.y - i][self.x + i] != -self.color and field[self.y - i][self.x + i] < 0:
                enemy_alive_up_right = self.norm_obs(i)
                break

        for i in range(1, len(field)):
            if self.y + i == len(field) or self.x + i == len(field):
                break
            elif field[self.y + i][self.x + i] != -self.color and field[self.y + i][self.x + i] < 0:
                enemy_alive_down_right = self.norm_obs(i)
                break

        for i in range(1, len(field)):
            if self.y + i == len(field) or self.x - i == -1:
                break
            elif field[self.y + i][self.x - i] != -self.color and field[self.y + i][self.x - i] < 0:
                enemy_alive_down_left = self.norm_obs(i)
                break

        for i in range(1, len(field)):
            if self.y - i == -1 or self.x - i == -1:
                break
            elif field[self.y - i][self.x - i] != -self.color and field[self.y - i][self.x - i] < 0:
                enemy_alive_up_left = self.norm_obs(i)
                break

        # расстояние до стен
        wall_up = self.norm_obs(self.y + 1)
        wall_right = self.norm_obs(len(field) - self.x)
        wall_down = self.norm_obs(len(field) - self.y)
        wall_left = self.norm_obs(self.x + 1)

        return np.array([dist_up, dist_right, dist_down, dist_left,
                         dist_up_right, dist_down_right, dist_down_left, dist_up_left,
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
    def __init__(self, populations, network, population_size, teams, random_spawn, deadly_walls, hall_of_fame, length_chrom, render):
        self.size = 128
        self.window_size = 768
        self.window = None
        self.clock = None
        self.pix_square_size = int(self.window_size / self.size)

        self.field = np.zeros((self.size, self.size))

        if render:
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
            self.clock = pygame.time.Clock()
            self.canvas = pygame.Surface((self.window_size, self.window_size))

        self.network = network

        self.deadly_walls = deadly_walls
        self.random_spawn = random_spawn

        self.teams = teams
        self.population_size = population_size
        self.hall_of_fame = hall_of_fame
        self.length_chrom = length_chrom

        self.populations = []
        self.populations_alive = []

        spawn_cords = [[self.size // 16, self.size // 16 * 7, self.size // 16, self.size // 16 * 7],
                       [self.size // 16 * 9, self.size // 16 * 15, self.size // 16, self.size // 16 * 7],
                       [self.size // 16, self.size // 16 * 7, self.size // 16 * 9, self.size // 16 * 15],
                       [self.size // 16 * 9, self.size // 16 * 15, self.size // 16 * 9, self.size // 16 * 15]]
        for i in range(self.teams):
            agents = []
            for agent in populations[i]:
                if self.random_spawn:
                    x = random.randint(0, self.size - 1)
                    y = random.randint(0, self.size - 1)
                else:
                    x = random.randint(spawn_cords[i][0], spawn_cords[i][1])
                    y = random.randint(spawn_cords[i][2], spawn_cords[i][3])
                while self.spawn_check(x, y):
                    if self.random_spawn:
                        x = random.randint(0, self.size - 1)
                        y = random.randint(0, self.size - 1)
                    else:
                        x = random.randint(spawn_cords[i][0], spawn_cords[i][1])
                        y = random.randint(spawn_cords[i][2], spawn_cords[i][3])

                agents.append(Agent(x, y, agent, i + 1))

            self.populations.append(agents)
            self.populations_alive.append(self.population_size)

        self.info = []
        for i in range(self.teams):
            self.info.append(0)
            for agent in self.populations[i]:
                agent.observation = agent.set_obs(self.field, self.populations[i])

    # коллизия при спавне
    def spawn_check(self, x, y):
        for population in self.populations:
            for agent in population:
                if x == agent.x and y == agent.y:
                    return True
        return False

    # количество закрашенных полей для всех комманд, количество живых агентов
    def get_info(self, prev_info = []):
        team_score = []
        done = False
        for population_color in range(1, self.teams + 1):
            score = 0
            for i in range(self.size):
                for j in range(self.size):
                    if abs(self.field[i][j]) == population_color:
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

    # отрисовка окружения
    def render(self, time):
        for i in range(self.size):
            for j in range(self.size):
                if self.field[i][j] != 0:
                    if abs(self.field[i][j]) == 1:
                        color = (100, 0, 0)
                    elif abs(self.field[i][j]) == 2:
                        color = (0, 100, 0)
                    elif abs(self.field[i][j]) == 3:
                        color = (0, 0, 100)
                    elif abs(self.field[i][j]) == 4:
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

    # награды
    # +1 за перекрашенное поле, еще +2, если при перекраске рядом находится союзник
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

                    self.field[agent.y][agent.x] = agent.color

                    # передвижение и столкновение со стенами
                    if action == 0:
                        agent.y -= 1
                        if agent.y == -1:
                            agent.y += 1
                            if self.deadly_walls:
                                agent.alive = False
                                self.populations_alive[population_ind] -= 1
                                agent.score -= 100
                                agent.total_score -= 100
                        elif self.field[agent.y][agent.x] == -agent.color:
                            agent.y += 1
                    elif action == 1:
                        agent.x += 1
                        if agent.x == self.size:
                            agent.x -= 1
                            if self.deadly_walls:
                                agent.alive = False
                                self.populations_alive[population_ind] -= 1
                                agent.score -= 100
                                agent.total_score -= 100
                        elif self.field[agent.y][agent.x] == -agent.color:
                            agent.x -= 1
                    elif action == 2:
                        agent.y += 1
                        if agent.y == self.size:
                            agent.y -= 1
                            if self.deadly_walls:
                                agent.alive = False
                                self.populations_alive[population_ind] -= 1
                                agent.score -= 100
                                agent.total_score -= 100
                        elif self.field[agent.y][agent.x] == -agent.color:
                            agent.y -= 1
                    elif action == 3:
                        agent.x -= 1
                        if agent.x == -1:
                            agent.x += 1
                            if self.deadly_walls:
                                agent.alive = False
                                self.populations_alive[population_ind] -= 1
                                agent.score -= 100
                                agent.total_score -= 100
                        elif self.field[agent.y][agent.x] == -agent.color:
                            agent.x += 1

                    if agent.alive:
                        # изменение цвета клетки
                        if abs(self.field[agent.y][agent.x]) != agent.color:
                            agent.score += 1
                            agent.total_score += 1

                            teammate_near = False
                            for j in range(12, 20):
                                if agent.observation[j] == 1:
                                    teammate_near = True
                                    break

                            if teammate_near:
                                agent.score += 2
                                agent.total_score += 2
                        self.field[agent.y][agent.x] = -agent.color

                        for i in range(self.teams):
                            # уничтожение агента другого цвета
                            if population_ind != i:
                                for enemy_agent in self.populations[i]:
                                    if enemy_agent.x == agent.x and enemy_agent.y == agent.y and enemy_agent.alive:
                                        enemy_agent.alive = False
                                        enemy_agent.score -= 100
                                        enemy_agent.total_score -= 100
                                        self.populations_alive[i] -= 1

                                        agent.score += 20
                                        agent.total_score += 20

                                        teammate_near = False
                                        for j in range(12, 20):
                                            if agent.observation[j] == 1:
                                                teammate_near = True
                                                break

                                        if teammate_near:
                                            agent.score += 20
                                            agent.total_score += 20

                            # возрождение союзного агента
                            else:
                                for teammate in self.populations[i]:
                                    if teammate.x == agent.x and teammate.y == agent.y and not teammate.alive:
                                        teammate.alive = True
                                        agent.score += 20
                                        agent.total_score += 20
                                        self.populations_alive[i] += 1

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

        return done

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
    def get_next_game_populations(self, team_score_alive, tourn_size):
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
                for k in range(tourn_size):
                    new_ind_index = random.randint(0, len(old_weights) - 1)
                    new_ind.append(old_weights[new_ind_index])

                new_ind.sort(key=lambda x: x[1])
                new_population.append(new_ind[-1][0])

            new_populations.append(new_population)
        return new_populations


    def reset(self, populations, render):
        self.field = np.zeros((self.size, self.size))

        if render:
            self.canvas.fill((0, 0, 0))
        for population in self.populations:
            for agent in population:
                agent.x = -1
                agent.y = -1
                agent.alive = True
                agent.score = 0
                agent.total_score = 0

        spawn_cords = [[self.size // 16, self.size // 16 * 7, self.size // 16, self.size // 16 * 7],
                       [self.size // 16 * 9, self.size // 16 * 15, self.size // 16, self.size // 16 * 7],
                       [self.size // 16, self.size // 16 * 7, self.size // 16 * 9, self.size // 16 * 15],
                       [self.size // 16 * 9, self.size // 16 * 15, self.size // 16 * 9, self.size // 16 * 15]]
        for i in range(self.teams):
            for weight_ind, agent in enumerate(self.populations[i]):
                if self.random_spawn:
                    x = random.randint(0, self.size - 1)
                    y = random.randint(0, self.size - 1)
                else:
                    x = random.randint(spawn_cords[i][0], spawn_cords[i][1])
                    y = random.randint(spawn_cords[i][2], spawn_cords[i][3])
                while self.spawn_check(x, y):
                    if self.random_spawn:
                        x = random.randint(0, self.size - 1)
                        y = random.randint(0, self.size - 1)
                    else:
                        x = random.randint(spawn_cords[i][0], spawn_cords[i][1])
                        y = random.randint(spawn_cords[i][2], spawn_cords[i][3])

                agent.x = x
                agent.y = y
                agent.weights = populations[i][weight_ind]

            self.populations_alive[i] = self.population_size

        self.info = []
        for i in range(self.teams):
            self.info.append([])
            for agent in self.populations[i]:
                agent.observation = agent.set_obs(self.field, self.populations[i])