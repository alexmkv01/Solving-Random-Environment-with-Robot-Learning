# Import some installed modules.
import time
import numpy as np
import pyglet

# Import some modules from this exercise.
from environment import Environment
from robot import Robot
import graphics
from model import Model

# Set the numpy random seed 
random_seed = int(time.time())
np.random.seed(random_seed)

# Create a pyglet window.
window = pyglet.window.Window(width=graphics.window_size + 100, height=graphics.window_size)
# Set the background colour to black.
pyglet.gl.glClearColor(0, 0, 0, 1)

# Create an environment.
environment = Environment(graphics.window_size, random_seed)
# Create a model.
robots_model = Model(environment)
# Create a robot.
robot = Robot(robots_model, environment.max_action, environment.goal_state)

# Set some flags which determine what is displayed on the window.
draw_terrain = True
draw_paths = False
draw_model = False

# Set the number of physical steps taken, and the number of resets, so far.
# Note that we assume that there will not be any resets during testing, i.e. testing is just one episode.
training_num_physical_steps = 0
training_num_resets = 0
testing_num_physical_steps = 0

# Create global variable for the testing start time.
testing_start_time = 0

# Set whether we are in training or testing mode.
is_training = True

# Define the time limits
TRAINING_TIME_LIMIT = 600
TESTING_TIME_LIMIT = 200

# Set the minimum distance to the goal during testing
testing_best_distance = np.inf


# Define what happens when the button is pressed
@window.event
def on_mouse_press(x, y, button, modifiers):
    global draw_terrain, draw_paths, draw_model
    # Check if the GUI buttons have been pressed
    if button == pyglet.window.mouse.LEFT:
        if graphics.button_x < x < graphics.button_x + graphics.button_width:
            if graphics.terrain_button_y < y < graphics.terrain_button_y + graphics.button_height:
                if draw_terrain:
                    draw_terrain = False
                else:
                    draw_terrain = True
            if graphics.paths_button_y < y < graphics.paths_button_y + graphics.button_height:
                if draw_paths:
                    draw_paths = False
                else:
                    draw_paths = True
            elif graphics.model_button_y < y < graphics.model_button_y + graphics.button_height:
                if draw_model:
                    draw_model = False
                else:
                    draw_model = True


# Define what happens when the rendering is called.
@window.event
def on_draw():
    # Clear the window by filling with the background colour (black).
    window.clear()
    # Draw the environment.
    graphics.draw_environment(environment, draw_terrain)
    # Draw the stored paths.
    if draw_paths:
        graphics.draw_paths(paths=robot.paths_to_draw)
    # Draw the model.
    if draw_model:
        graphics.draw_model(model=robot.model, action=np.array([environment.max_action, environment.max_action]))
    # Draw the buttons
    graphics.draw_buttons(draw_paths, draw_model, draw_terrain)
    # Optionally, save an image of the current window.
    # This may be helpful for the coursework.
    if 0:
        graphics.save_image()


# Define what happens on each timestep.
def step(dt):
    global is_training, training_num_physical_steps, testing_num_physical_steps, training_num_resets, testing_best_distance, training_start_time, testing_start_time
    if is_training:
        # Trigger the robot to calculate the next action.
        action, reset = robot.next_action_training(environment.state)
        # Reset?
        if reset:
            environment.reset()
            training_num_resets += 1
        else:
            # Execute this action in the environment.
            state, next_state = environment.step(action)
            training_num_physical_steps += 1
            # Send this information back to the robot.
            robot.process_transition(state, action)
        # Compute the time remaining
        time_now = time.time()
        cpu_time = time_now - training_start_time
        action_time = training_num_physical_steps + 20 * training_num_resets
        time_elapsed = cpu_time + action_time
        time_remaining = TRAINING_TIME_LIMIT - time_elapsed
        print(f'Training: Time elapsed = {int(time_elapsed)} / {TRAINING_TIME_LIMIT}')
        if time_remaining <= 0:
            print('---- Finished training, starting testing ----')
            is_training = False
            environment.reset()
            testing_start_time = time.time()
    else:
        # Trigger the robot to calculate the next action.
        action = robot.next_action_testing(environment.state)
        # Execute this action in the environment.
        environment.step(action)
        testing_num_physical_steps += 1
        # Calculate the time elapsed
        time_now = time.time()
        cpu_time = time_now - testing_start_time
        action_time = testing_num_physical_steps
        time_elapsed = cpu_time + action_time
        print(f'Testing: Time elapsed = {int(time_elapsed)} / {TESTING_TIME_LIMIT}')
        # See if this is the best so far
        distance = np.linalg.norm(environment.state - environment.goal_state)
        if distance < testing_best_distance:
            testing_best_distance = distance
        # See if the robot has reached the goal
        if distance < 0.03:
            print(f'---- Reached goal in {time_elapsed} seconds ----')
            pyglet.app.exit()
        time_remaining = TESTING_TIME_LIMIT - time_elapsed
        if time_remaining <= 0:
            print(f'---- Did not reach goal in time. Closest distance = {testing_best_distance} ----')
            pyglet.app.exit()


# Set how frequently the update function is called.
pyglet.clock.schedule_interval(step, 0.001)

# Create global variable for the training start time, and start the timer.
training_start_time = time.time()

# Finally, call the main pyglet event loop.
# This will continually update and render the environment in a loop.
pyglet.app.run()
