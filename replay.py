import pygame.event
from matplotlib import pyplot as plt
import pygame
from params import WINDOW_SIZE, SIZE, TEAMS, CONC, MAX_HP, GAMES, TIME_LIMIT
import joblib

TIME = 1000

pix_square_size = int(WINDOW_SIZE / SIZE)
pygame.init()
pygame.display.init()
window = pygame.display.set_mode((WINDOW_SIZE, WINDOW_SIZE))
clock = pygame.time.Clock()
canvas = pygame.Surface((WINDOW_SIZE, WINDOW_SIZE))

# клетка - xyzN; x - цвет клетки, y - концентрация клетки, z - цвет бота на клетке, N - кол-во хп
def render(time, frame):
    for i in range(SIZE):
        for j in range(SIZE):
            if frame[i][j][0] != "0":
                if frame[i][j][0] == "1":
                    color_cell = (200 // CONC * int(frame[i][j][1]), 200 // CONC * int(frame[i][j][1]), 0)
                elif frame[i][j][0] == "2":
                    color_cell = (0, 200 // CONC * int(frame[i][j][1]), 0)
                elif frame[i][j][0] == "3":
                    color_cell = (0, 200 // CONC * int(frame[i][j][1]), 200 // CONC * int(frame[i][j][1]))
                elif frame[i][j][0] == "4":
                    color_cell = (0, 0, 200 // CONC * int(frame[i][j][1]))
                elif frame[i][j][0] == "5":
                    color_cell = (255, 0, 127)
                elif frame[i][j][0] == "6":
                    color_cell = (255, 102, 255)
                elif frame[i][j][0] == "7":
                    color_cell = (200 // CONC * int(frame[i][j][1]), 0, 0)
                pygame.draw.rect(
                    canvas,
                    color_cell,
                    (j * pix_square_size, i * pix_square_size,
                     pix_square_size, pix_square_size),
                )

            if frame[i][j][2] != "0":
                if frame[i][j][2] == "1":
                    color_bot = (int(frame[i][j][3:]) * 255 // MAX_HP, int(frame[i][j][3:]) * 255 // MAX_HP, 0)
                elif frame[i][j][2] == "2":
                    color_bot = (0, int(frame[i][j][3:]) * 255 // MAX_HP, 0)
                elif frame[i][j][2] == "3":
                    color_bot = (0, int(frame[i][j][3:]) * 255 // MAX_HP, int(frame[i][j][3:]) * 255 // MAX_HP)
                elif frame[i][j][2] == "4":
                    color_bot = (0, 0, int(frame[i][j][3:]) * 255 // MAX_HP)
                elif frame[i][j][2] == "5":
                    color_bot = (100, 100, 100)
                pygame.draw.rect(
                    canvas,
                    (255, 255, 255),
                    (j * pix_square_size + 1, i * pix_square_size + 1,
                     pix_square_size - 2, pix_square_size - 2),
                )
                pygame.draw.rect(
                    canvas,
                    color_bot,
                    (j * pix_square_size + 2, i * pix_square_size + 2,
                     pix_square_size - 4, pix_square_size - 4),
                )

    window.blit(canvas, canvas.get_rect())
    pygame.event.pump()
    pygame.display.update()
    clock.tick(time)

# recorded_train_games.sav
recorded_games = joblib.load("sav/recorded_train_games.sav")
# train_scores.sav
team_scores_total = joblib.load("sav/train_scores.sav")
# moves_per_game.sav
moves_per_game = joblib.load("sav/moves_per_game.sav")
# max_score_per_game.sav
max_score_per_game = joblib.load("sav/max_score_per_game.sav")
# avg_score_per_game.sav
avg_score_per_game = joblib.load("sav/avg_score_per_game.sav")

game_count = 0
for game in recorded_games:
    if GAMES < 10:
        if game_count == 0:
            print("Игра", game_count)
        else:
            print("Игра", GAMES)
    else:
        print("Игра", game_count * GAMES // 10)
    canvas.fill((0, 0, 0))
    frame = 0
    done = False
    while not done:
        for i in pygame.event.get():
            if i.type == pygame.QUIT:
                done = True
        render(TIME, game[frame])
        frame += 1
        if frame == len(game):
            done = True
    game_count += 1

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
