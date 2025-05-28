from PIL import Image
import cv2
import numpy as np
from tensorflow.python.framework import config as tf_config
import tensorflow as tf
import random
import os


ACTION_INDEX_TO_STRING = {
    0: "UP",
    1: "DOWN",
    2: "RIGHT",
    3: "LEFT",
    4: "STAY",
    5: "INTERACT"
}


BASE_DIR = "evaluations"


def set_seed(seed=42):
    # Set seeds for reproducibility
    np.random.seed(seed)
    tf.random.set_seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    tf_config.set_soft_device_placement(True)


def show_window(img_path, actions=None, q_values=None, timeout=500, save_images=False, img_idx=0):
    img = Image.open(img_path)
    image = np.array(img)
    image_cv = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    if save_images:
        out_path = os.path.join(BASE_DIR, f"states/{img_idx}.png")
        cv2.imwrite(out_path, image_cv)
    else:
        cv2.namedWindow("State", cv2.WINDOW_AUTOSIZE)
        cv2.moveWindow("State", 100, 100)
        cv2.imshow("State", image_cv)
        cv2.waitKey(timeout)
        cv2.destroyAllWindows()

    os.remove(img_path)

    if actions:
        show_action_text(actions, timeout)


def show_action_text(actions, timeout):
    action1, action2 = actions
    action1 = ACTION_INDEX_TO_STRING[action1]
    action2 = ACTION_INDEX_TO_STRING[action2]

    width, height = 200, 200
    background_color = (255, 255, 255)  # white background
    text_color = (0, 0, 0)  # black text
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    thickness = 2

    # Create a blank white image
    image = np.full((height, width, 3), background_color, dtype=np.uint8)
    # Put the first text
    cv2.putText(image, action1, (50, 70), font, font_scale, text_color, thickness, cv2.LINE_AA)
    # Put the second text
    cv2.putText(image, action2, (50, 140), font, font_scale, text_color, thickness, cv2.LINE_AA)

    # Show the image
    window_name = "Actions"
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
    cv2.moveWindow(window_name, 500, 100)  # adjust window position
    cv2.imshow(window_name, image)
    cv2.waitKey(timeout)


def get_action_values_plt(q_values, agent_idx):
    q_values = np.array(q_values)
    assert q_values.shape == (2, 6), "values should have shape (2, 6)"

    # New dimensions for a smaller canvas
    height, width = 300, 500
    margin = 40  # Reduced margin
    bar_width = 15  # Reduced bar width
    spacing = 60  # Reduced spacing between actions

    num_actions = q_values.shape[1]
    num_agents = q_values.shape[0]

    # Colors in BGR (OpenCV format)
    if agent_idx == 0:
        agent_colors = [(204, 102, 0), (102, 204, 0)]
    else:
        agent_colors = [(102, 204, 0), (204, 102, 0)]

    # Create white canvas
    img = np.ones((height, width, 3), dtype=np.uint8) * 255

    # Normalize Q-values for plotting height
    max_q = np.max(q_values)
    scale = (height - 2 * margin) / max_q if max_q > 0 else 1

    # X positions for each action, ensure it is centered properly
    start_x = (width - (num_actions * spacing)) // 2  # Center the actions
    positions = [start_x + i * spacing for i in range(num_actions)]

    base_y = height - margin

    # Draw bars
    for action_idx in range(num_actions):
        for agent_idx in range(num_agents):
            q = q_values[agent_idx][action_idx]
            bar_height = int(q * scale)

            x1 = int(positions[action_idx] + (agent_idx - 0.5) * bar_width)
            y1 = base_y
            x2 = x1 + bar_width
            y2 = base_y - bar_height

            cv2.rectangle(
                img,
                (x1, y1),
                (x2, y2),
                agent_colors[agent_idx],
                thickness=-1
            )

    # Determine max val per agent
    max_indices = [np.argmax(q_values[agent_idx]) for agent_idx in range(num_agents)]

    # Draw action labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    for i, pos in enumerate(positions):
        label = ACTION_INDEX_TO_STRING[i]

        # Determine label color
        label_color = (0, 0, 0)  # default: black
        for agent_idx, max_idx in enumerate(max_indices):
            if i == max_idx:
                label_color = agent_colors[agent_idx]

        text_size = cv2.getTextSize(label, font, 0.5, 1)[0]
        text_x = int(pos - text_size[0] / 2)
        text_y = base_y + 30
        cv2.putText(img, label, (text_x, text_y), font, 0.5, label_color, 1, cv2.LINE_AA)

    # Add title and axis labels
    cv2.putText(img, "Action-Values for Each Agent", (int(width / 2 - 150), 40), font, 0.8, (0, 0, 0), 2)
    cv2.putText(img, "Actions", (int(width / 2 - 30), height - 20), font, 0.6, (0, 0, 0), 1)
    cv2.putText(img, "Q", (10, int(height / 2)), font, 0.6, (0, 0, 0), 1)

    """# Draw Y-axis ticks and labels (Q-values)
    num_ticks = 5  # Number of ticks for the Q-value axis
    tick_step = max_q / num_ticks  # Step size for ticks
    for i in range(num_ticks + 1):
        tick_value = int(i * tick_step)
        tick_x = margin - 10
        tick_y = int(base_y - i * (height - 2 * margin) / num_ticks)
        cv2.putText(img, str(tick_value), (tick_x, tick_y + 5), font, 0.5, (0, 0, 0), 1, cv2.LINE_AA)  # Q-value labels
    """
    return img


def load_and_delete_img(img_path):
    img = Image.open(img_path)
    image = np.array(img)
    image_cv = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    os.remove(img_path)
    return image_cv


def save_video_from_images(imgs, output_video_path, fps=30):
    # Get the shape of the first image to define the frame size (height, width, channels)
    height, width, channels = imgs[0].shape

    # Define the codec and create a VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'XVID')  # You can use other codecs like 'MP4V', 'MJPG'
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    for img in imgs:
        # Ensure the image is in the correct format (BGR for OpenCV)
        if img.shape[2] == 1:  # Convert grayscale to BGR if necessary
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        out.write(img)  # Write each image as a frame in the video

    out.release()  # Finalize the video writing process


def count_params(model):
    total_params = sum(p.numel() for p in model.parameters())
    return f"{total_params:,}"


def save_img_list(img_list, output_folder):
    # Create the folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Save each image
    for idx, img_array in enumerate(img_list):
        img_rgb = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
        # Convert the ndarray to a PIL Image
        img = Image.fromarray(img_rgb)
        # Define the file path to save the image
        img_path = os.path.join(output_folder, f'image_{idx}.png')
        # Save the image
        img.save(img_path)


def to_value(x):
    """
    Converts a tensor to a raw float value.
    """
    try:
        return float(x.numpy())  # TensorFlow tensor
    except:
        return float(x)  # Already a NumPy or float value


def to_tuple(tensor_tuple):
    """
    Converts a tuple of tensors to a tuple of raw float values.
    """
    return tuple(to_value(x) for x in tensor_tuple)


def interpret_state(state_vector, agent_idx):
    if agent_idx == 1:
        print("\033[92m------------------------------------------------------------\033[0m")
    else:
        print("\033[94m------------------------------------------------------------\033[0m")

    num_pots = 2
    num_players = 2
    # Lengths of each component based on the problem description
    player_i_feature_length = num_pots * 10 + 28
    other_player_feature_length = (num_players - 1) * (num_pots * 10 + 24)
    player_i_dist_to_other_players = (num_players - 1)*2
    dist_to_other_players_length = (num_players - 1) * 2

    assert player_i_feature_length + other_player_feature_length + player_i_dist_to_other_players + dist_to_other_players_length == 96

    # Extract player i's features
    player_i_features = state_vector[:player_i_feature_length]
    # Extract player i's distances to other players
    player_i_dist_to_other_players = state_vector[-4:-2]
    # Extract player i's position
    player_i_position = state_vector[-2:]
    # Start interpreting player i's features
    #print("Player features, origin of axis is UP-LEFT:")
    idx = 0
    # Orientation (length 4 one-hot encoding)
    orientation_mapping = ['UP', 'DOWN', 'RIGHT', 'LEFT']
    pi_orientation = to_tuple(player_i_features[idx:idx + 4])
    orientation = orientation_mapping[pi_orientation.index(1)]  # Get the index of the '1'
    print(f"Orientation: {orientation}")
    idx += 4

    # Object held (length 4 one-hot encoding)
    object_mapping = ['Onion', 'Soup', 'Dish', '0']
    pi_obj = to_tuple(player_i_features[idx:idx + 4])
    # Check if there's any '1' in the one-hot encoding, meaning something is held
    if 1 in pi_obj:
        object_held = object_mapping[pi_obj.index(1)]  # Get the index of the '1'
    else:
        object_held = "No object held"  # If all values are 0, it means no object is hel
    print(f"Object held: {object_held}")
    idx += 4

    # pi_closest_{onion|tomato|dish|soup|serving|empty_counter}: (dx, dy)
    pi_closest_onion = tuple(player_i_features[idx:idx + 2])
    print(f"Closest onion (dx, dy): {to_tuple(pi_closest_onion)}")
    idx += 2

    pi_closest_tomato = tuple(player_i_features[idx:idx + 2])
    print(f"Closest tomato (dx, dy): {to_tuple(pi_closest_tomato)}")
    idx += 2

    pi_closest_dish = tuple(player_i_features[idx:idx + 2])
    print(f"Closest dish (dx, dy): {to_tuple(pi_closest_dish)}")
    idx += 2

    pi_closest_soup = tuple(player_i_features[idx:idx + 2])
    print(f"Closest soup (dx, dy): {to_tuple(pi_closest_soup)}")
    idx += 2

    pi_closest_serving = tuple(player_i_features[idx:idx + 2])
    print(f"Closest empty counter (dx, dy): {to_tuple(pi_closest_serving)}")
    idx += 2
    if to_tuple(pi_closest_serving) != (0, 0):
        print("XXX")
    if to_tuple(pi_closest_serving) == (3, 0):
        print("YYY")

    pi_closest_empty_counter = tuple(player_i_features[idx:idx + 2])
    print(f"Closest serving (dx, dy): {to_tuple(pi_closest_empty_counter)}")
    idx += 2

    # pi_cloest_soup_n_{onions|tomatoes}
    pi_closest_soup_n_onions = int(player_i_features[idx])
    print(f"Number of onions in closest soup: {pi_closest_soup_n_onions}")
    idx += 1

    pi_closest_soup_n_tomatoes = int(player_i_features[idx])
    print(f"Number of tomatoes in closest soup: {pi_closest_soup_n_tomatoes}")
    idx += 1

    # pi_closest_pot_{j}_exists, is_empty, is_full, is_cooking, is_ready, num_onions, num_tomatoes, cook_time
    num_pots_features = 10  # 10 features per pot
    for j in range(num_pots):
        pot_features = player_i_features[idx:idx + num_pots_features]
        print(f"Pot {j} features:")
        interpret_pot_features(pot_features)
        idx += num_pots_features

    # pi_wall (length 4 boolean)
    pi_wall = player_i_features[idx:idx + 4]
    print(f"Walls in each direction (N, E, S, W): {to_tuple(pi_wall)}")
    idx += 4

    # Interpret distances to other players
    for j in range(num_players - 1):
        dist = tuple(player_i_dist_to_other_players[j * 2:j * 2 + 2])
        print(f"Distance to other player: {to_tuple(dist)}")

    if agent_idx == 1:
        print("\033[92m------------------------------------------------------------\033[0m", end="\n\n")
    else:
        print("\033[94m------------------------------------------------------------\033[0m", end="\n\n")


def interpret_pot_features(pot_features):
    """
    Interprets the features for a given pot.
    """
    # Pot exists: 0 or 1
    pot_exists = int(pot_features[0])
    print(f"  Pot exists: {'Yes' if pot_exists else 'No'}")

    # Pot state (empty, full, cooking, ready)
    is_empty = bool(pot_features[1])
    is_full = bool(pot_features[2])
    is_cooking = bool(pot_features[3])
    is_ready = bool(pot_features[4])

    print(f"  Pot state: {'Empty' if is_empty else 'Not Empty'}, {'Full' if is_full else 'Not Full'}, "
          f"{'Cooking' if is_cooking else 'Not Cooking'}, {'Ready' if is_ready else 'Not Ready'}")

    # Pot contents: number of onions and tomatoes
    num_onions = int(pot_features[5])
    num_tomatoes = int(pot_features[6])
    print(f"  Number of onions in pot: {num_onions}")
    print(f"  Number of tomatoes in pot: {num_tomatoes}")

    # Remaining cooking time (if pot is cooking, otherwise -1)
    cook_time = pot_features[7]  # -1 if no soup is cooking
    if cook_time != -1:
        print(f"  Remaining cooking time: {cook_time} seconds")
    else:
        print("  No soup is cooking.")

    pot_position = pot_features[8:10]
    print(f"  Distance from player (dx, dy): {to_tuple(pot_position)}")
