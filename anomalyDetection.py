import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from collections import deque

# ----------------------------
# Data Stream Simulation Module
# ----------------------------

def data_stream_simulation(stream_size=1500, 
                           seasonality_period=120, 
                           noise_level=0.7, 
                           drift_rate=0.001,
                           anomaly_count=15,
                           anomaly_min=3,
                           anomaly_max=6):
    """
    Generates a synthetic data stream with combined seasonality, exponential drift, noise, and anomalies.

    Parameters:
    - stream_size (int): Total number of data points.
    - seasonality_period (int): Period of the seasonal component.
    - noise_level (float): Standard deviation of the Gaussian noise.
    - drift_rate (float): Rate of exponential drift.
    - anomaly_count (int): Number of anomalies to introduce.
    - anomaly_min (float): Minimum magnitude of anomalies.
    - anomaly_max (float): Maximum magnitude of anomalies.

    Returns:
    - np.array: Simulated data stream.
    
    Raises:
    - ValueError: If input parameters are out of expected ranges.
    """
    try:
        # Validate input parameters
        if stream_size <= 0:
            raise ValueError("stream_size must be positive.")
        if seasonality_period <= 0:
            raise ValueError("seasonality_period must be positive.")
        if noise_level < 0:
            raise ValueError("noise_level cannot be negative.")
        if drift_rate < 0:
            raise ValueError("drift_rate cannot be negative.")
        if not (0 <= anomaly_count < stream_size):
            raise ValueError("anomaly_count must be non-negative and less than stream_size.")
        if anomaly_min <= 0 or anomaly_max <= anomaly_min:
            raise ValueError("anomaly_min must be positive and less than anomaly_max.")
        
        time = np.arange(stream_size)
        
        # Combined seasonality using sine and cosine for complexity
        seasonal_component = 0.5 * np.sin(2 * np.pi * time / seasonality_period) + \
                             0.3 * np.cos(4 * np.pi * time / seasonality_period)
        
        # Exponential drift
        drift = np.exp(drift_rate * time)
        
        # Gaussian noise
        noise = np.random.normal(0, noise_level, size=stream_size)
        
        # Introduce anomalies
        anomaly_indices = random.sample(range(100, stream_size-100), k=anomaly_count)
        anomaly_magnitude = [random.uniform(anomaly_min, anomaly_max) for _ in range(anomaly_count)]
        
        data = drift + seasonal_component + noise
        for idx, mag in zip(anomaly_indices, anomaly_magnitude):
            data[idx] += mag
        
        return data
    except Exception as e:
        print(f"Error in data_stream_simulation: {e}")
        raise

# ----------------------------
# Anomaly Detection Module
# ----------------------------

class AnomalyDetector:
    """
    Anomaly Detector using the Z-Score method with a sliding window.

    Attributes:
    - window_size (int): Number of data points to consider for calculating statistics.
    - threshold (float): Z-score threshold for detecting anomalies.
    - data_window (deque): Sliding window to store recent data points.
    - mean (float): Current mean of the sliding window.
    - std_dev (float): Current standard deviation of the sliding window.
    """
    def __init__(self, window_size=60, threshold=2.5):
        """
        Initializes the anomaly detector with specified window size and threshold.

        Parameters:
        - window_size (int): Number of data points for the sliding window.
        - threshold (float): Z-score threshold for anomaly detection.
        
        Raises:
        - ValueError: If window_size is not positive or threshold is non-positive.
        """
        if window_size <= 0:
            raise ValueError("window_size must be positive.")
        if threshold <= 0:
            raise ValueError("threshold must be positive.")
        
        self.window_size = window_size
        self.threshold = threshold
        self.data_window = deque(maxlen=window_size)
        self.mean = 0
        self.std_dev = 0

    def update_statistics(self, value):
        """
        Updates the rolling mean and standard deviation with a new data point.

        Parameters:
        - value (float): New data point.
        """
        self.data_window.append(value)
        if len(self.data_window) == self.window_size:
            self.mean = np.mean(self.data_window)
            self.std_dev = np.std(self.data_window)

    def detect_anomaly(self, value):
        """
        Detects whether a new data point is an anomaly based on the Z-Score.

        Parameters:
        - value (float): New data point.

        Returns:
        - bool: True if anomaly, False otherwise.
        """
        if len(self.data_window) < self.window_size:
            return False
        z_score = (value - self.mean) / (self.std_dev + 1e-5)  # Added epsilon to prevent division by zero
        return abs(z_score) > self.threshold

# ----------------------------
# Visualization Module
# ----------------------------

def plot_data_stream(data_stream):
    """
    Plots the simulated data stream with combined seasonality and anomalies using enhanced aesthetics.

    Parameters:
    - data_stream (np.array): The simulated data stream to plot.
    """
    try:
        plt.style.use('ggplot')  # Using 'ggplot' style for a clean look
        plt.figure(figsize=(12, 7))
        
        # Plot the data stream
        plt.plot(data_stream, label='Data Stream', color='#1f77b4', linewidth=2)
        plt.title('Simulated Data Stream with Complex Seasonality and Anomalies', fontsize=18)
        plt.xlabel('Time', fontsize=14)
        plt.ylabel('Value', fontsize=14)
        plt.legend(fontsize=12)
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        plt.tight_layout()
        plt.show()
    except Exception as e:
        print(f"Error in plot_data_stream: {e}")
        raise

def real_time_visualization(data_stream, detector):
    """
    Visualizes the data stream in real-time with anomaly detection, using enhanced colors and markers.

    Parameters:
    - data_stream (np.array): The simulated data stream.
    - detector (AnomalyDetector): The anomaly detector instance.
    """
    try:
        plt.style.use('ggplot')  # Consistent style with the static plot
        fig, ax = plt.subplots(figsize=(12, 7))
        
        x_data, y_data = [], []
        anomaly_x, anomaly_y = [], []
        
        ax.set_xlim(0, len(data_stream))
        ax.set_ylim(np.min(data_stream) - 5, np.max(data_stream) + 5)
        
        # Enhanced plot lines and scatter
        line, = ax.plot([], [], color='#ff7f0e', label='Data Stream', linewidth=2)
        anomaly_scatter = ax.scatter([], [], color='#d62728', marker='D', label='Anomalies', edgecolors='k', s=100)
        
        ax.set_title('Real-Time Data Stream with Enhanced Anomaly Detection', fontsize=18)
        ax.set_xlabel('Time', fontsize=14)
        ax.set_ylabel('Value', fontsize=14)
        ax.legend(loc='upper left', fontsize=12)
        
        def init():
            line.set_data([], [])
            anomaly_scatter.set_offsets(np.empty((0, 2)))
            return line, anomaly_scatter
        
        def update(frame):
            x_data.append(frame)
            y_data.append(data_stream[frame])
            
            detector.update_statistics(data_stream[frame])
            is_anomaly = detector.detect_anomaly(data_stream[frame])
            
            if is_anomaly:
                print(f"Anomaly detected at index {frame}, value: {data_stream[frame]:.2f}")
                anomaly_x.append(frame)
                anomaly_y.append(data_stream[frame])
            
            line.set_data(x_data, y_data)
            
            if anomaly_x and anomaly_y:
                anomaly_scatter.set_offsets(np.c_[anomaly_x, anomaly_y])
            
            # Dynamically adjust the x-axis to keep the plot moving
            if frame > ax.get_xlim()[1] - 200:
                ax.set_xlim(0, frame + 500)
                ax.figure.canvas.draw()
            
            return line, anomaly_scatter
        
        ani = FuncAnimation(fig, update, frames=range(len(data_stream)), init_func=init,
                            blit=True, interval=50, repeat=False)
        
        plt.tight_layout()
        plt.show()
    except Exception as e:
        print(f"Error in real_time_visualization: {e}")
        raise

# ----------------------------
# Main Execution Module
# ----------------------------

def get_user_parameters():
    """
    Prompts the user to input parameters for the anomaly detector.

    Returns:
    - tuple: (window_size (int), threshold (float))
    """
    try:
        window_size_input = 60
        threshold_input = 2.5
        
        window_size = int(window_size_input) if window_size_input else 60
        threshold = float(threshold_input) if threshold_input else 2.5
        
        if window_size <= 0 or threshold <= 0:
            raise ValueError("Window size and threshold must be positive numbers.")
        
        return window_size, threshold
    except ValueError as ve:
        print(f"Invalid input: {ve}. Using default parameters.")
        return 60, 2.5
    except Exception as e:
        print(f"Unexpected error: {e}. Using default parameters.")
        return 60, 2.5

def main():
    """
    Main function to execute the anomaly detection workflow.
    """
    try:
        # Step 1: Generate simulated data stream with updated parameters
        data_stream = data_stream_simulation(
            stream_size=1500, 
            seasonality_period=120, 
            noise_level=0.7, 
            drift_rate=0.001,
            anomaly_count=15,
            anomaly_min=3,
            anomaly_max=6
        )
        
        # Step 2: Plot the initial data stream with enhanced aesthetics
        plot_data_stream(data_stream)
        
        # Step 3: Initialize anomaly detector with user inputs
        window_size, threshold = get_user_parameters()
        detector = AnomalyDetector(window_size=window_size, threshold=threshold)
        
        # Step 4: Visualize real-time data with anomaly detection
        real_time_visualization(data_stream, detector)
        
    except Exception as e:
        print(f"An error occurred in the main execution: {e}")

if __name__ == "__main__":
    main()
