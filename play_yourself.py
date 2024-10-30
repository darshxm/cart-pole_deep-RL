import gymnasium as gym
import time
import keyboard  # Make sure to install this package

# Initialize the environment
env = gym.make('CartPole-v1', render_mode="human")
state, _ = env.reset()
done = False
total_reward = 0
game_started = False  # Start flag

print("Press 'S' to start the game. Use arrow keys to control the CartPole: 'left' for left force, 'right' for right force.")

# Wait for user to press 'S' to start
while not game_started:
    if keyboard.is_pressed("s"):
        game_started = True
        print("Game started! Control the cart with the arrow keys.")
    time.sleep(0.1)  # Prevent CPU overuse while waiting

# Game loop
while not done:
    env.render()  # Display the environment
    
    # Slow down the simulation
    time.sleep(0.1)  # Adjust this delay as needed for desired speed
    
    # Apply force based on keyboard input
    force = 0  # Neutral force
    if keyboard.is_pressed("left"):
        force = -1  # Apply force to the left
    elif keyboard.is_pressed("right"):
        force = 1  # Apply force to the right

    # Convert continuous force to action (0 or 1)
    action = 0 if force < 0 else 1

    # Take the chosen action in the environment
    state, reward, done, _, _ = env.step(action)
    total_reward += reward

print(f"Total Reward from the Episode: {total_reward}")
env.close()
