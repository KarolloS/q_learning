import random as rand
import numpy as np


def next_move():
    m = input("what is your next move?")
    if m == 'w':
        return 0
    elif m == 's':
        return 1
    elif m == 'a':
        return 2
    elif m == 'd':
        return 3
    else:
        print("Wrong Movement!")
        return -1


def perform_move(a_pos, next_m, g_size):
    loc = np.transpose(np.nonzero(a_pos))
    loc = loc[0]

    w_status = 0

    new_a_pos = np.copy(a_pos)
    if next_m == 0:
        if loc[0] > 0:
            new_a_pos[loc[0]-1, loc[1]] = 1
            new_a_pos[loc[0], loc[1]] = 0
        else:
            w_status = 1
    elif next_m == 1:
        if loc[0] < g_size-1:
            new_a_pos[loc[0]+1, loc[1]] = 1
            new_a_pos[loc[0], loc[1]] = 0
        else:
            w_status = 1
    elif next_m == 2:
        if loc[1] > 0:
            new_a_pos[loc[0], loc[1]-1] = 1
            new_a_pos[loc[0], loc[1]] = 0
        else:
            w_status = 1
    elif next_m == 3:
        if loc[1] < g_size-1:
            new_a_pos[loc[0], loc[1]+1] = 1
            new_a_pos[loc[0], loc[1]] = 0
        else:
            w_status = 1

    return new_a_pos, w_status


def game_over(a_pos, g_pos, o_pos):
    a_loc = np.transpose(np.nonzero(a_pos))
    g_loc = np.transpose(np.nonzero(g_pos))
    o_loc = np.transpose(np.nonzero(o_pos))

    for i in range(len(np.transpose(o_loc)[0])):
        if np.array_equal(a_loc[0], o_loc[i]):
            return True

    if np.array_equal(a_loc[0], g_loc[0]):
        return True

    return False


def get_reward(a_pos, g_pos, o_pos, w_status, v1, v2, v3):
    a_loc = np.transpose(np.nonzero(a_pos))
    g_loc = np.transpose(np.nonzero(g_pos))
    o_loc = np.transpose(np.nonzero(o_pos))

    for i in range(len(np.transpose(o_loc)[0])):
        if np.array_equal(a_loc[0], o_loc[i]):
            return -v1 + v2 + w_status*v3

    if np.array_equal(a_loc[0], g_loc[0]):
        return v1 + v2 + w_status*v3

    return v2 + w_status*v3


def render(a_pos, g_pos, o_pos, grid_size):
    grid = np.empty((grid_size, grid_size), dtype=str)
    for i in range(grid_size):
        for j in range(grid_size):
            grid[i, j] = ' '

    idx = np.nonzero(g_pos)
    grid[idx] = '+'

    idx = np.nonzero(o_pos)
    grid[idx] = 'O'

    idx = np.nonzero(a_pos)
    grid[idx] = 'P'

    print(grid)


def main():

    score = 0
    s = 4
    n = 1

    agent_pos = np.zeros((s, s))
    agent_pos[rand.randrange(s), rand.randrange(s)] = 1

    goal_pos = np.zeros((s, s))
    goal_pos[rand.randrange(s), rand.randrange(s)] = 1
    while np.array_equal(agent_pos, goal_pos):
        goal_pos = np.zeros((s, s))
        goal_pos[rand.randrange(s), rand.randrange(s)] = 1

    obstacle_pos = np.zeros((s, s))
    for i in range(n):
        temp = np.zeros((s, s))
        temp[rand.randrange(s), rand.randrange(s)] = 1
        while np.array_equal(temp, agent_pos) or np.array_equal(temp, goal_pos) or np.amax(obstacle_pos + temp) > 1:
            temp = np.zeros((s, s))
            temp[rand.randrange(s), rand.randrange(s)] = 1
        obstacle_pos += temp

    render(agent_pos, goal_pos, obstacle_pos, s)

    while not game_over(agent_pos, goal_pos, obstacle_pos):
        m = next_move()
        agent_pos, wall_status = perform_move(agent_pos, m, s)
        render(agent_pos, goal_pos, obstacle_pos, s)

        score += get_reward(agent_pos, goal_pos, obstacle_pos, wall_status, 1, -0.05, -0.5)
        print("Your score: " + str(score))

    print("Game Finished!")


main()
