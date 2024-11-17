import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from collections import deque
import pyttsx3

# Initialize pyttsx3 engine
def speak_text(text):
    engine = pyttsx3.init()
    engine.setProperty('rate', 125)  # Speaking rate
    engine.setProperty('volume', 1.0)  # Maximum volume
    voices = engine.getProperty('voices')
    engine.setProperty('voice', voices[1].id)  # Set voice
    engine.say(text)
    engine.runAndWait()
    engine.stop()

# Parameters
WINDOW_SIZE = 5  # Rolling window for anomaly detection
THRESHOLD = 1.1  # Z-score threshold for anomalies
MAX_POINTS = 100  # Maximum points to display on the graph

# Flags for speaking
flag_speed = True
flag_horn = True

# Simulated data storage
timestamps = deque(maxlen=MAX_POINTS)
speed_data = deque(maxlen=MAX_POINTS)
horn_usage_data = deque(maxlen=MAX_POINTS)
speed_anomalies = deque(maxlen=MAX_POINTS)
horn_anomalies = deque(maxlen=MAX_POINTS)

# Initialize figure and subplots
fig, axs = plt.subplots(2, 1, figsize=(10, 8))
plt.subplots_adjust(hspace=0.4)

# Add text for anomaly counts
speed_anomaly_text = fig.text(0.8, 0.8, "Speed Anomalies: 0", fontsize=12, color="blue")
horn_anomaly_text = fig.text(0.8, 0.6, "Horn Anomalies: 0", fontsize=12, color="green")

# Function to simulate new data points
def generate_data():
    speed = np.random.randint(15, 35) + np.random.choice([0, 20], p=[0.95, 0.05])  # Spike 5% chance
    horn_usage = np.random.randint(0, 4) + np.random.choice([0, 10], p=[0.98, 0.02])  # Spike 2% chance
    return speed, horn_usage

# Function to detect anomalies
def detect_anomaly(data):
    if len(data) < WINDOW_SIZE:
        return False
    rolling_mean = pd.Series(data).rolling(WINDOW_SIZE).mean().iloc[-1]
    rolling_std = pd.Series(data).rolling(WINDOW_SIZE).std().iloc[-1]
    if rolling_std == 0:  # Avoid division by zero
        return False
    z_score = (data[-1] - rolling_mean) / rolling_std
    return abs(z_score) > THRESHOLD

# Function to update the plot
def update(frame):
    global flag_speed, flag_horn  # Use global flags to manage state

    # Simulate new data
    speed, horn_usage = generate_data()
    timestamp = pd.Timestamp.now()

    # Append new data
    timestamps.append(timestamp)
    speed_data.append(speed)
    horn_usage_data.append(horn_usage)
    speed_anomalies.append(detect_anomaly(list(speed_data)))
    horn_anomalies.append(detect_anomaly(list(horn_usage_data)))

    # Count anomalies
    speed_anomaly_count = sum(speed_anomalies)
    horn_anomaly_count = sum(horn_anomalies)

    # Update anomaly text
    speed_anomaly_text.set_text(f"Speed Anomalies: {speed_anomaly_count}")
    horn_anomaly_text.set_text(f"Horn Anomalies: {horn_anomaly_count}")

    # Trigger speaking if thresholds exceeded and flag is set
    if speed_anomaly_count > 6 and flag_speed:
        speak_text("Rash driving detected. Calling the authorities.")
        flag_speed = False  # Disable further speech for speed

    if horn_anomaly_count > 6 and flag_horn:
        speak_text("Excessive horn usage detected. Please reduce noise.")
        flag_horn = False  # Disable further speech for horn usage

    # Clear and redraw Speed subplot
    axs[0].clear()
    axs[0].plot(timestamps, speed_data, label="Speed", color="blue")
    axs[0].scatter(
        [timestamps[i] for i in range(len(speed_anomalies)) if speed_anomalies[i]],
        [speed_data[i] for i in range(len(speed_anomalies)) if speed_anomalies[i]],
        color="red",
        label="Anomalies",
    )
    axs[0].set_title("Real-time Speed with Anomalies")
    axs[0].set_ylabel("Speed (km/h)")
    axs[0].grid()
    axs[0].legend()

    # Clear and redraw Horn Usage subplot
    axs[1].clear()
    axs[1].plot(timestamps, horn_usage_data, label="Horn Usage", color="green")
    axs[1].scatter(
        [timestamps[i] for i in range(len(horn_anomalies)) if horn_anomalies[i]],
        [horn_usage_data[i] for i in range(len(horn_anomalies)) if horn_anomalies[i]],
        color="red",
        label="Anomalies",
    )
    axs[1].set_title("Real-time Horn Usage with Anomalies")
    axs[1].set_ylabel("Horn Usage (Count)")
    axs[1].grid()
    axs[1].legend()

    # Set common x-axis labels
    axs[1].set_xlabel("Time")

# Initialize animation
ani = FuncAnimation(fig, update, interval=1000)  # Update every 1 second

# Show plot
plt.show()
