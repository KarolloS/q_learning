import random as rand
import tensorflow as tf
import time
import numpy as np


# calculate next agent position based on the next move
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


# check if the game is over
def game_over(a_pos, g_pos, o_pos):
    a_loc = np.transpose(np.nonzero(a_pos))
    g_loc = np.transpose(np.nonzero(g_pos))
    o_loc = np.transpose(np.nonzero(o_pos))

    # agent in the obstacle
    for i in range(len(np.transpose(o_loc)[0])):
        if np.array_equal(a_loc[0], o_loc[i]):
            return True

    # agent reached the goal
    if np.array_equal(a_loc[0], g_loc[0]):
        return True

    return False


# obtain final reward for either loosing or wining the game
def get_reward(a_pos, g_pos, o_pos, w_status, r1, r2, r3):
    a_loc = np.transpose(np.nonzero(a_pos))
    g_loc = np.transpose(np.nonzero(g_pos))
    o_loc = np.transpose(np.nonzero(o_pos))

    # agent in the obstacle
    for i in range(len(np.transpose(o_loc)[0])):
        if np.array_equal(a_loc[0], o_loc[i]):
            return -r1 + r2 + r3*w_status

    # agent reached the goal
    if np.array_equal(a_loc[0], g_loc[0]):
        return r1 + r2 + r3*w_status

    return r2 + r3*w_status


# 'draw' play grid in console
def render(a_pos, g_pos, o_pos, g_size):
    grid = np.empty((g_size, g_size), dtype=str)
    for i in range(g_size):
        for j in range(g_size):
            grid[i, j] = ' '

    # goal
    idx = np.nonzero(g_pos)
    grid[idx] = '+'

    # obstacles
    idx = np.nonzero(o_pos)
    grid[idx] = 'O'

    # agent
    idx = np.nonzero(a_pos)
    grid[idx] = 'P'

    print(grid)


###############################################################################
# Set parameters
training_iteration = 10000
training_bound = float('inf')
gamma_1 = 0.95
gamma_2 = 0.98

v1 = 1  # goal/obstacle reward
v2 = -0.05  # movement reward
v3 = -0.25  # wall reward

n = 1  # number of obstacles
grid_size = 5  # size of the grid

batch_size = 32
buffer_size = 64
buffer = []  # stores tuples of (S, A, R, S')
h = 0

# TF graph input
x = tf.placeholder("float", [None, 3*grid_size**2], name='input')  # obstacles + goal + start
y = tf.placeholder("float", [None, 4], name='output')  # down, up, left, right

# Create a model
# Set model weights
W1 = tf.Variable(tf.random_normal([3*grid_size**2, 1024], stddev=0.1), name='weight_1')
W2 = tf.Variable(tf.random_normal([1024, 512], stddev=0.1), name='weight_2')
W3 = tf.Variable(tf.random_normal([512, 4], stddev=0.1), name='weight_3')

with tf.variable_scope("W3W2W1x"):
    # Construct a linear model
    h1 = tf.nn.relu(tf.matmul(x, W1))
    h2 = tf.nn.relu(tf.matmul(h1, W2))
    model = tf.matmul(h2, W3)

# Add summary ops to collect data
w1_h = tf.summary.histogram(W1.op.name, W1)
w2_h = tf.summary.histogram(W2.op.name, W2)
w3_h = tf.summary.histogram(W2.op.name, W3)

# Create loss function
with tf.variable_scope("cost_function"):
    # Minimize error
    cost_function = tf.reduce_mean(tf.pow(y - model, 2))
    # Create a summary to monitor the cost function
    tf.summary.scalar(cost_function.op.name, cost_function)

with tf.variable_scope("train"):
    # Gradient descent
    # optimizer = tf.train.GradientDescentOptimizer(0.05).minimize(cost_function)
    optimizer = tf.train.AdadeltaOptimizer(1.0).minimize(cost_function)

# Initializing the variables
init = tf.global_variables_initializer()

# Merge all summaries into a single operator
merged_summary_op = tf.summary.merge_all()

# Launch the graph
# with tf.Session(config=tf.ConfigProto(device_count={'GPU': 0})) as sess:  # don't use GPU
with tf.Session() as sess:  # use GPU
    sess.run(init)

    # Set the logs writer
    summary_writer = tf.summary.FileWriter('log', sess.graph)

    # ########## TRAIN

    print("Training started...")
    start_time = time.time()

    # Training cycle
    for iteration in range(training_iteration):

        score = 0  # game score

        # define agent starting position
        agent_pos = np.zeros((grid_size, grid_size))
        agent_pos[rand.randrange(grid_size), rand.randrange(grid_size)] = 1

        # define goal position
        goal_pos = np.zeros((grid_size, grid_size))
        goal_pos[rand.randrange(grid_size), rand.randrange(grid_size)] = 1
        while np.array_equal(agent_pos, goal_pos):
            goal_pos = np.zeros((grid_size, grid_size))
            goal_pos[rand.randrange(grid_size), rand.randrange(grid_size)] = 1

        # define obstacle position
        obstacle_pos = np.zeros((grid_size, grid_size))
        for __ in range(n):
            temp = np.zeros((grid_size, grid_size))
            temp[rand.randrange(grid_size), rand.randrange(grid_size)] = 1
            while np.array_equal(temp, agent_pos) or np.array_equal(temp, goal_pos) or np.amax(obstacle_pos + temp) > 1:
                temp = np.zeros((grid_size, grid_size))
                temp[rand.randrange(grid_size), rand.randrange(grid_size)] = 1
            obstacle_pos += temp

        # arrange input data into 1-dimensional numpy array (1 ,3*grid_size**2)
        data = np.zeros(3*grid_size**2)
        data[0:grid_size**2] = np.reshape(agent_pos, grid_size**2)
        data[grid_size ** 2:2*grid_size ** 2] = np.reshape(goal_pos, grid_size ** 2)
        data[2*grid_size ** 2:3*grid_size ** 2] = np.reshape(obstacle_pos, grid_size ** 2)
        data = data.reshape((1, 3*grid_size ** 2))

        # render(agent_pos, goal_pos, obstacle_pos, grid_size)

        # begin one game
        while not game_over(agent_pos, goal_pos, obstacle_pos):

            # predict next movement
            Q_predict = sess.run(model, feed_dict={x: data})

            b = 0.9 * (1 - iteration/training_iteration) + 0.1
            if rand.random() < b:
                move = rand.randint(0, 3)
            else:
                move = np.argmax(Q_predict)

            # perform next movement
            new_agent_pos, wall_status = perform_move(agent_pos, move, grid_size)
            new_data = np.copy(data)
            new_data[0, 0:grid_size ** 2] = np.reshape(new_agent_pos, grid_size ** 2)

            # get reward
            reward = get_reward(new_agent_pos, goal_pos, obstacle_pos, wall_status, v1, v2, v3)

            if len(buffer) < buffer_size:
                buffer.append((data, move, reward, new_data))
            else:
                if h < buffer_size-1:
                    h += 1
                else:
                    h = 0
                buffer[h] = (data, move, reward, new_data)

                if iteration < training_bound:
                    batch = [(data, move, reward, new_data)]
                    gamma = gamma_1
                else:
                    batch = rand.sample(buffer, batch_size)
                    gamma = gamma_2

                x_train = []
                y_train = []

                for memory in batch:
                    data, move, reward, new_data = memory
                    score += reward/len(batch)

                    # predict old movement
                    Q_predict = sess.run(model, feed_dict={x: data})

                    # predict new movement
                    Q_target = sess.run(model, feed_dict={x: new_data})

                    # calculate target value of Q function
                    target = np.copy(Q_predict)
                    if reward == v2:  # non-terminal state
                        target[0, move] = gamma * np.amax(Q_target) + reward
                    else:  # terminal state
                        target[0, move] = reward

                    # update batch training data
                    x_train.append(data.reshape(3*grid_size**2,))
                    y_train.append(target.reshape(4,))

                x_train = np.array(x_train)
                y_train = np.array(y_train)

                # Fit training using batch data
                sess.run(optimizer, feed_dict={x: x_train, y: y_train})

                # Write logs for each iteration
                summary_str = sess.run(merged_summary_op, feed_dict={x: x_train, y: y_train})
                summary_writer.add_summary(summary_str, iteration)

                # Update agent position
                agent_pos = new_agent_pos
                data[0, 0:grid_size ** 2] = np.reshape(agent_pos, grid_size ** 2)

        # Display logs per iteration step
        if np.array_equal(agent_pos, goal_pos):
            print("Iteration:", "{:05d}".format(iteration), "   +  average game score: ", "{:.2f}".format(score))
        else:
            print("Iteration:", "{:05d}".format(iteration), "   -  average game score: ", "{:.2f}".format(score))

    elapsed_time = time.time() - start_time
    print("Training completed! (" + str(time.strftime("%H:%M:%S", time.gmtime(elapsed_time))) + ")")

    # ########## TEST

    test_iteration = 500
    avg_score = 0  # average testing score
    acc = 0  # number of goals reached

    # Testing cycle
    for iteration in range(test_iteration):

        score = 0  # game score

        # define agent starting position
        agent_pos = np.zeros((grid_size, grid_size))
        agent_pos[rand.randrange(grid_size), rand.randrange(grid_size)] = 1

        # define goal position
        goal_pos = np.zeros((grid_size, grid_size))
        goal_pos[rand.randrange(grid_size), rand.randrange(grid_size)] = 1
        while np.array_equal(agent_pos, goal_pos):
            goal_pos = np.zeros((grid_size, grid_size))
            goal_pos[rand.randrange(grid_size), rand.randrange(grid_size)] = 1

        # define obstacle position
        obstacle_pos = np.zeros((grid_size, grid_size))
        for __ in range(n):
            temp = np.zeros((grid_size, grid_size))
            temp[rand.randrange(grid_size), rand.randrange(grid_size)] = 1
            while np.array_equal(temp, agent_pos) or np.array_equal(temp, goal_pos) or np.amax(obstacle_pos + temp) > 1:
                temp = np.zeros((grid_size, grid_size))
                temp[rand.randrange(grid_size), rand.randrange(grid_size)] = 1
            obstacle_pos += temp

        render(agent_pos, goal_pos, obstacle_pos, grid_size)

        # arrange input data into 1-dimensional numpy array (1 ,3*grid_size**2)
        data = np.zeros(3*grid_size**2)
        data[0:grid_size**2] = np.reshape(agent_pos, grid_size**2)
        data[grid_size ** 2:2*grid_size ** 2] = np.reshape(goal_pos, grid_size ** 2)
        data[2*grid_size ** 2:3*grid_size ** 2] = np.reshape(obstacle_pos, grid_size ** 2)
        data = data.reshape((1, 3*grid_size ** 2))

        # begin one game
        while not game_over(agent_pos, goal_pos, obstacle_pos):

            # predict next movement
            Q_predict = sess.run(model, feed_dict={x: data})
            move = np.argmax(Q_predict)

            # perform next movement
            agent_pos, wall_status = perform_move(agent_pos, move, grid_size)
            data[0, 0:grid_size ** 2] = np.reshape(agent_pos, grid_size ** 2)

            # get reward and update overall score for this game
            reward = get_reward(agent_pos, goal_pos, obstacle_pos, wall_status, v1, v2, v3)
            score += reward

            render(agent_pos, goal_pos, obstacle_pos, grid_size)

            if np.array_equal(agent_pos, goal_pos):
                acc += 1
                print("------  CORRECT  ------")

            # prevent loops
            if score <= -5 or np.array_equal(agent_pos, obstacle_pos):
                print("------  ERROR  ------")
                score = 0
                break

        print("___________")

        # calculate average score over all testing iterations
        avg_score += score

    avg_score /= acc
    acc /= test_iteration
    # Display final score
    print("- - - - - - - - - -")
    print("Average test score (of correct games): ", "{:.3f}".format(avg_score))
    print("Accuracy: ", "{:.2f}".format(acc))


# tensorboard --logdir=D:\Studia\Inne\q_learning
