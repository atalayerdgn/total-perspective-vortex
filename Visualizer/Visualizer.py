import matplotlib.pyplot as plt

class SignalVisualizer:
    """
    Visualize EEG signals.
    Args:
        signal (np.ndarray): The EEG signal to visualize.
    Attributes:
        signal (np.ndarray): The EEG signal to visualize.
    Methods:
        plot(title='vis'): Plot the EEG signal.
    Usage:
        vis = SignalVisualizer(signal)
        vis.plot(title='EEG Signal Visualization')
    Example:
        vis = SignalVisualizer(np.random.randn(1000))  # Example signal
        vis.plot(title='Random EEG Signal')
    """
    def __init__(self, signal):
        self.signal = signal

    def plot(self, title='vis'):
        plt.figure(figsize=(10, 4))
        plt.plot(self.signal, color='blue')
        plt.title(title)
        plt.show()
