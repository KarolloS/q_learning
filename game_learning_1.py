import random as rand
import tensorflow as tf
import time
import numpy as np


# calculate next agent position based on the next move
def perform_move(a_pos, next_m, g_size):
    loc = np.transpose(np.nonzero(a_pos))
    loc = loc[0]

    new_a_pos = np.copy(a_pos)
    if next_m == 0:
        if loc[0] > 0:
            new_a_pos[loc[0]-1, loc[1]] = 1
            new_a_pos[loc[0], loc[1]] = 0
    elif next_m == 1:
        if loc[0] < g_size-1:
            new_a_pos[loc[0]+1, loc[1]] = 1
            new_a_pos[loc[0], loc[1]] = 0
    elif next_m == 2:
        if loc[1] > 0:
            new_a_pos[loc[0], loc[1]-1] = 1
            new_a_pos[loc[0], loc[1]] = 0
    elif next_m == 3:
        if loc[1] < g_size-1:
            new_a_pos[loc[0], loc[1]+1] = 1
            new_a_pos[loc[0], loc[1]] = 0

    return new_a_pos


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
def get_reward(a_pos, g_pos, o_pos, r1, r2):
    a_loc = np.transpose(np.nonzero(a_pos))
    g_loc = np.transpose(np.nonzero(g_pos))
    o_loc = np.transpose(np.nonzero(o_pos))

    # agent in the obstacle
    for i in range(len(np.transpose(o_loc)[0])):
        if np.array_equal(a_loc[0], o_loc[i]):
            return -r1 + r2

    # agent reached the goal
    if np.array_equal(a_loc[0], g_loc[0]):
        return r1 + r2

    return r2


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
training_iteration = 1000

v1 = 1  # goal/obstacle reward
v2 = -0.05  # movement reward

grid_size = 4  # size of the grid


# TF graph input
x = tf.placeholder("float", [None, 3*grid_size**2], name='input')  # obstacles + goal + start
y = tf.placeholder("float", [None, 4], name='output')  # down, up, left, right

# Create a model
# Set model weights
W1 = tf.Variable(tf.random_normal([3*grid_size**2, 128], stddev=0.1), name='weight_1')
W2 = tf.Variable(tf.random_normal([128, 128], stddev=0.1), name='weight_2')
W3 = tf.Variable(tf.random_normal([128, 4], stddev=0.1), name='weight_3')

with tf.variable_scope("W3W2W1x_b"):
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
# with tf.Session() as sess:  # use GPU
with tf.Session(config=tf.ConfigProto(device_count={'GPU': 0})) as sess:  # don't use GPU
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
        agent_pos[0, 1] = 1

        # define goal position
        goal_pos = np.zeros((grid_size, grid_size))
        goal_pos[3, 3] = 1

        # define obstacle position
        obstacle_pos = np.zeros((grid_size, grid_size))
        obstacle_pos[1, 1] = 1
        obstacle_pos[2, 3] = 1

        # arrange input data into 1-dimensional numpy array (1 ,3*grid_size**2)
        data = np.zeros(3*grid_size**2)
        data[0:grid_size**2] = np.reshape(agent_pos, grid_size**2)
        data[grid_size ** 2:2*grid_size ** 2] = np.reshape(goal_pos, grid_size ** 2)
        data[2*grid_size ** 2:3*grid_size ** 2] = np.reshape(obstacle_pos, grid_size ** 2)
        data = data.reshape((1, 3*grid_size ** 2))

        # render(agent_pos,goal_pos,obstacle_pos,grid_size)

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
            new_agent_pos = perform_move(agent_pos, move, grid_size)
            new_data = np.copy(data)
            new_data[0, 0:grid_size ** 2] = np.reshape(new_agent_pos, grid_size ** 2)

            # get reward and update overall score for this game
            reward = get_reward(new_agent_pos, goal_pos, obstacle_pos, v1, v2)
            score += reward

            # predict the best possible movement for new state
            Q_target = sess.run(model, feed_dict={x: new_data})

            # calculate target value of Q function
            target = np.copy(Q_predict)
            if reward == v2:  # non-terminal state
                target[0, move] = 0.9*np.amax(Q_target) + reward
            else:  # terminal state
                target[0, move] = reward

            # Fit training
            sess.run(optimizer, feed_dict={x: data, y: target})

            # Write logs for each iteration
            summary_str = sess.run(merged_summary_op, feed_dict={x: data, y: target})
            summary_writer.add_summary(summary_str, iteration)

            # Update agent position
            agent_pos = new_agent_pos
            data[0, 0:grid_size ** 2] = np.reshape(agent_pos, grid_size ** 2)

            # prevent loops
            if score <= -5:
                break

        # Display logs per iteration step
        print("Iteration: ", "{:05d}".format(iteration), "   game score: ", "{:.2f}".format(score))

    elapsed_time = time.time() - start_time
    print("Training completed! (" + str(time.strftime("%H:%M:%S", time.gmtime(elapsed_time))) + ")")

    # ########## TEST

    test_iteration = 1
    avg_score = 0  # average testing score

    # Testing cycle
    for iteration in range(test_iteration):

        score = 0  # game score

        # define agent starting position
        agent_pos = np.zeros((grid_size, grid_size))
        agent_pos[0, 1] = 1

        # define goal position
        goal_pos = np.zeros((grid_size, grid_size))
        goal_pos[3, 3] = 1

        # define obstacle position
        obstacle_pos = np.zeros((grid_size, grid_size))
        obstacle_pos[1, 1] = 1
        obstacle_pos[2, 3] = 1

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
            agent_pos = perform_move(agent_pos, move, grid_size)
            data[0, 0:grid_size ** 2] = np.reshape(agent_pos, grid_size ** 2)

            # get reward and update overall score for this game
            reward = get_reward(agent_pos, goal_pos, obstacle_pos, v1, v2)
            score += reward

            render(agent_pos, goal_pos, obstacle_pos, grid_size)

            # prevent loops
            if score <= -10:
                break

        # calculate average score over all testing iterations
        avg_score += score/test_iteration

    # Display final average score
    print("- - - - - - - - - -")
    print("Average test game score: ", "{:.2f}".format(avg_score))


# tensorboard --logdir=D:\Studia\Inne\q_learning
