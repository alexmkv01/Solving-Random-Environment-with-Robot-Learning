# Import some installed modules.
import pyglet
import os
import numpy as np


# Define the window width and height (a square) in screen pixels.
# You may wish to modify this to fit your screen size.
window_size = 500

# Define the properties of the buttons
button_width = 80
button_height = 50
button_x = window_size + 10
terrain_button_y = window_size - 10 - button_height
paths_button_y = terrain_button_y - 10 - button_height
model_button_y = paths_button_y - 10 - button_height


# Class to store a path which will then be drawn to the window.
class Path:

    # Initialisation function.
    def __init__(self, points, colour, width, skip=0):
        self.points = points
        self.colour = colour
        self.width = width
        self.skip = skip


# Function to draw the environment on the window.
def draw_environment(environment, draw_terrain):
    if draw_terrain:
        # Draw the background sprite (i.e. the image showing the terrain).
        environment.background_sprite.draw()
    # Draw the initial state as a blue square.
    init_width = init_height = 0.07 * window_size
    init_x = environment.init_state[0] * window_size - 0.5 * init_width
    init_y = environment.init_state[1] * window_size - 0.5 * init_width
    init_colour = (30, 30, 255)
    blue_square = pyglet.shapes.Rectangle(x=init_x, y=init_y, width=init_width, height=init_height, color=init_colour)
    blue_square.draw()
    # Draw the goal state as a green star.
    goal_x = environment.goal_state[0] * window_size
    goal_y = environment.goal_state[1] * window_size
    inner_radius = 0.025 * window_size
    outer_radius = 0.05 * window_size
    goal_colour = (0, 200, 0)
    green_star = pyglet.shapes.Star(x=goal_x, y=goal_y, outer_radius=outer_radius, inner_radius=inner_radius, num_spikes=5, color=goal_colour)
    green_star.draw()
    # Draw the robot as a red circle.
    robot_x = environment.state[0] * window_size
    robot_y = environment.state[1] * window_size
    robot_radius = 0.025 * window_size
    robot_colour = (200, 0, 0)
    red_circle = pyglet.shapes.Circle(x=robot_x, y=robot_y, radius=robot_radius, color=robot_colour)
    red_circle.draw()


# Function to draw some paths on the window.
# The skip argument allows you to speed up drawing by only drawing some of the points on the path.
def draw_paths(paths, skip=0):
    batch = pyglet.graphics.Batch()
    batch_lines = []
    for path in paths:
        for i in range(len(path.points) - 1 - skip):
            point_1 = window_size * path.points[i]
            point_2 = window_size * path.points[i + 1 + skip]
            line_segment = pyglet.shapes.Line(point_1[0], point_1[1], point_2[0], point_2[1], width=path.width, color=path.colour, batch=batch)
            batch_lines.append(line_segment)
            i += skip
    batch.draw()


# Function to draw the lines representing the model.
def draw_model(model, action):
    batch = pyglet.graphics.Batch()
    batch_shapes = []
    cell_line_width = 5
    cell_line_colour = (50, 50, 50)
    # Draw the vertical cell lines
    for cell_x in range(10):
        point_1_y = 0
        point_2_y = window_size
        point_1_x = cell_x * 0.1 * window_size
        point_2_x = cell_x * 0.1 * window_size
        line = pyglet.shapes.Line(point_1_x, point_1_y, point_2_x, point_2_y, width=cell_line_width, color=cell_line_colour, batch=batch)
        batch_shapes.append(line)
    # Draw the horizontal cell lines
    for cell_y in range(10):
        point_1_x = 0
        point_2_x = window_size
        point_1_y = cell_y * 0.1 * window_size
        point_2_y = cell_y * 0.1 * window_size
        line = pyglet.shapes.Line(point_1_x, point_1_y, point_2_x, point_2_y, width=cell_line_width, color=cell_line_colour, batch=batch)
        batch_shapes.append(line)
    # Loop through each cell and draw the predicted next state
    action_line_colour = (150, 150, 150)
    action_line_width = 3
    current_state_colour = (255, 255, 255)
    current_state_width = 10
    next_state_colour = (255, 255, 255)
    next_state_radius = 5
    for cell_x in range(10):
        for cell_y in range(10):
            current_state = np.array([cell_x * 0.1 + 0.05, cell_y * 0.1 + 0.05], dtype=np.float32)
            next_state, _ = model.predict(current_state, action)
            current_state_point = window_size * current_state
            next_state_point = window_size * next_state
            action_line = pyglet.shapes.Line(current_state_point[0], current_state_point[1], next_state_point[0], next_state_point[1], width=action_line_width, color=action_line_colour, batch=batch)
            batch_shapes.append(action_line)
            current_state_square = pyglet.shapes.Rectangle(current_state_point[0] - current_state_width * 0.5, current_state_point[1] - current_state_width * 0.5, width=current_state_width, height=current_state_width, color=current_state_colour, batch=batch)
            batch_shapes.append(current_state_square)
            next_state_circle = pyglet.shapes.Circle(next_state_point[0], next_state_point[1], next_state_radius, color=next_state_colour, batch=batch)
            batch_shapes.append(next_state_circle)
    batch.draw()


# Function to draw the buttons on the top-right of the window.
def draw_buttons(draw_paths, draw_model, draw_terrain):
    # Draw the dividing line
    dividing_line = pyglet.shapes.Line(x=window_size, y=0, x2=window_size, y2=window_size, width=3, color=(255, 255, 255, 255))
    dividing_line.draw()
    # Create and draw some rectangles
    if draw_terrain:
        terrain_button_colour = (100, 200, 100, 255)
    else:
        terrain_button_colour = (200, 100, 100, 255)
    paths_button = pyglet.shapes.Rectangle(x=button_x, y=terrain_button_y, width=button_width, height=button_height, color=terrain_button_colour)
    paths_button.draw()
    if draw_paths:
        paths_button_colour = (100, 200, 100, 255)
    else:
        paths_button_colour = (200, 100, 100, 255)
    paths_button = pyglet.shapes.Rectangle(x=button_x, y=paths_button_y, width=button_width, height=button_height, color=paths_button_colour)
    paths_button.draw()
    if draw_model:
        model_button_colour = (100, 200, 100, 255)
    else:
        model_button_colour = (200, 100, 100, 255)
    model_button = pyglet.shapes.Rectangle(x=button_x, y=model_button_y, width=button_width, height=button_height, color=model_button_colour)
    model_button.draw()
    # Create and draw some text
    terrain_text = pyglet.text.Label("Show Terrain", color=(0, 0, 0, 255), font_name='Arial', font_size=9, x=button_x + 5, y=terrain_button_y + 20)
    terrain_text.draw()
    paths_text = pyglet.text.Label("Show Paths", color=(0, 0, 0, 255), font_name='Arial', font_size=10, x=button_x + 5, y=paths_button_y + 20)
    paths_text.draw()
    model_text = pyglet.text.Label("Show Model", color=(0, 0, 0, 255), font_name='Arial', font_size=10, x=button_x + 5, y=model_button_y + 20)
    model_text.draw()


# Function to save an image of the current window.
def save_image():
    # If the program is quit during the saving process, the image file will be corrupted.
    # Therefore, we save it with a temporary filename.
    # Then when we are sure that saving has completed, we rename it to the desired filename.
    # With this, if the program quits during the saving process, the previous image with this filename will persist.
    filename = 'robot-learning.png'
    path = os.path.join(os.getcwd(), filename)
    path_temp = os.path.join(os.getcwd(), 'temp.png')
    pyglet.image.get_buffer_manager().get_color_buffer().save(path_temp)
    os.rename(path_temp, path)
