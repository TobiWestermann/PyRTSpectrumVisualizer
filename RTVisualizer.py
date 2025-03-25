import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.io import wavfile
from scipy import signal
import argparse
import sys
import time
import sounddevice as sd
import threading

class WavFrequencyVisualizer:
    def __init__(self, wav_file, bin_count=64, smooth_factor=0.5, update_interval=50, playback_speed=1.0, audio_latency=0.0, start_paused=True):

        self.bin_count = bin_count
        self.smooth_factor = smooth_factor
        self.update_interval = update_interval
        self.playback_speed = playback_speed
        self.audio_latency = audio_latency

        self.paused = start_paused
        self.pause_start_time = None
        self.total_pause_time = 0

        try:
            self.sample_rate, self.audio_data = wavfile.read(wav_file)
            print(f"Loaded WAV file: {wav_file}")
            print(f"Sample rate: {self.sample_rate} Hz")
            print(f"Duration: {len(self.audio_data) / self.sample_rate:.2f} seconds")
        except Exception as e:
            print(f"Error loading WAV file: {e}")
            sys.exit(1)

        if len(self.audio_data.shape) > 1:
            self.audio_data = self.audio_data.mean(axis=1)

        self.audio_data = self.audio_data.astype(np.float32)
        self.audio_data /= np.max(np.abs(self.audio_data))

        self.audio_playing = False
        self.stream = None

        self.chunk_size = 2048
        self.overlap = self.chunk_size // 2
        self.start_time = None
        self.position = 0
        self.window = signal.windows.hann(self.chunk_size)

        nyquist = self.sample_rate // 2
        min_freq = 20

        use_octave_bins = True
        
        if use_octave_bins:
            octaves = np.log2(nyquist / min_freq)
            bins_per_octave = int(np.ceil(self.bin_count / octaves))
            
            self.freq_bins = []
            current_freq = min_freq
            
            while current_freq <= nyquist:
                self.freq_bins.append(current_freq)
                current_freq *= 2**(1/bins_per_octave)
            
            if len(self.freq_bins) < self.bin_count + 1:
                self.freq_bins.append(nyquist)
            elif len(self.freq_bins) > self.bin_count + 1:
                self.freq_bins = self.freq_bins[:self.bin_count + 1]
            
            self.freq_bins = np.array(self.freq_bins)
        else:
            self.freq_bins = np.logspace(
                np.log10(min_freq),
                np.log10(nyquist),
                self.bin_count + 1
            )
    

        self.prev_spectrum = np.zeros(self.bin_count)

        self.setup_plot()

        self.position = 0

        self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)
        
        self.pause_text = self.ax.text(
            0.5, 0.5, 'PAUSED\nPress SPACE to play', 
            transform=self.ax.transAxes,
            fontsize=24,
            color='white',
            ha='center',
            va='center',
            alpha=0.7,
            visible=self.paused
        )
    
    def start_audio(self):
        if self.stream is not None:
            self.stop_audio()

        latency_samples = int(self.audio_latency * self.sample_rate)
        adjusted_position = max(0, self.position - latency_samples)
        
        start_time_sec = adjusted_position / self.sample_rate
        
        try:
            self.stream = sd.play(
                self.audio_data[adjusted_position:], 
                self.sample_rate * self.playback_speed,
                blocking=False
            )
            self.audio_playing = True
            print(f"Starting audio playback at position {adjusted_position} ({start_time_sec:.2f}s)")
        except Exception as e:
            print(f"Error starting audio playback: {e}")

    def stop_audio(self):
        """Stop audio playback"""
        if self.audio_playing:
            sd.stop()
            self.audio_playing = False
            print("Stopped audio playback")

    def setup_plot(self):
        """Set up the matplotlib figure and axes"""
        plt.style.use('dark_background')
        self.fig, self.ax = plt.subplots(figsize=(12, 6))
        self.fig.canvas.manager.set_window_title('WAV Frequency Visualizer')

        self.bars = self.ax.bar(
            range(self.bin_count), 
            np.zeros(self.bin_count),
            width=0.8,
            color='cyan',
            alpha=0.7
        )

        cm = plt.colormaps.get_cmap('viridis')
        for i, bar in enumerate(self.bars):
            bar.set_color(cm(i / self.bin_count))
            
        self.ax.set_ylim(0, 1.1)
        self.ax.set_xlim(-0.5, self.bin_count - 0.5)
        self.ax.set_title('Real-time Frequency Spectrum', fontsize=16)
        self.ax.set_xlabel('Frequency Bins (log)', fontsize=12)
        self.ax.set_ylabel('Magnitude', fontsize=12)

        standard_freqs = [20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000]
    
        tick_positions = []
        tick_labels = []
        
        for freq in standard_freqs:
            if freq <= self.sample_rate / 2:
                bin_index = np.argmin(np.abs(self.freq_bins[:-1] - freq))
                tick_positions.append(bin_index)
                
                if freq >= 1000:
                    tick_labels.append(f"{freq/1000:.1f}k")
                else:
                    tick_labels.append(f"{freq}")
        
        self.ax.set_xticks(tick_positions)
        self.ax.set_xticklabels(tick_labels)

        self.time_text = self.ax.text(
            0.02, 0.95, 'Time: 0.00 s', 
            transform=self.ax.transAxes,
            fontsize=12,
            color='white'
        )

        self.progress_ax = self.fig.add_axes([0.1, 0.02, 0.8, 0.03])
        self.progress_bar = self.progress_ax.barh(
            [0], [0], height=1, color='cyan', alpha=0.7
        )
        self.progress_ax.set_xlim(0, len(self.audio_data))
        self.progress_ax.axis('off')

    def update(self, frame):
        if self.paused:
            return list(self.bars) + list(self.progress_bar) + [self.time_text, self.pause_text]
        
        if self.start_time is None:
            self.start_time = time.time() - self.total_pause_time
            
        elapsed_time = (time.time() - self.start_time - self.total_pause_time) * self.playback_speed
        target_position = int(elapsed_time * self.sample_rate)
        
        if target_position >= len(self.audio_data):
            self.start_time = time.time() - self.total_pause_time
            self.position = 0
            
            self.stop_audio()
            if not self.paused:
                self.start_audio()
        else:
            self.position = target_position

        start_idx = self.position
        end_idx = start_idx + self.chunk_size

        if end_idx >= len(self.audio_data):
            chunk = np.pad(self.audio_data[start_idx:], (0, end_idx - len(self.audio_data))) * self.window
        else:
            chunk = self.audio_data[start_idx:end_idx] * self.window

        fft = np.fft.rfft(chunk)

        magnitude = np.abs(fft) / len(fft)

        freqs = np.fft.rfftfreq(self.chunk_size, 1.0 / self.sample_rate)

        binned_spectrum = self.bin_frequencies(freqs, magnitude)

        binned_spectrum = np.sqrt(binned_spectrum) * 2

        smoothed_spectrum = (self.smooth_factor * self.prev_spectrum + 
                            (1 - self.smooth_factor) * binned_spectrum)

        smoothed_spectrum = np.clip(smoothed_spectrum, 0, 1)

        for i, bar in enumerate(self.bars):
            bar.set_height(smoothed_spectrum[i])

        self.prev_spectrum = smoothed_spectrum

        current_time = start_idx / self.sample_rate
        total_time = len(self.audio_data) / self.sample_rate

        current_minutes, current_seconds = divmod(current_time, 60)
        total_minutes, total_seconds = divmod(total_time, 60)

        self.time_text.set_text(f'Time: {int(current_minutes):02}:{int(current_seconds):02} / {int(total_minutes):02}:{int(total_seconds):02}')

        self.progress_bar[0].set_width(start_idx)

        artists_list = list(self.bars) + list(self.progress_bar) + [self.time_text, self.pause_text]
        return artists_list

    
    def bin_frequencies(self, freqs, spectrum):

        binned_spectrum = np.zeros(self.bin_count)
        
        for i in range(self.bin_count):
            low_idx = np.argmax(freqs >= self.freq_bins[i])
            high_idx = np.argmax(freqs >= self.freq_bins[i+1])

            if high_idx == 0:
                high_idx = len(freqs)

            if low_idx < high_idx:
                binned_spectrum[i] = np.mean(spectrum[low_idx:high_idx])
        
        return binned_spectrum
    
    def run(self):
        """Run the visualizer animation"""
        self.animation = FuncAnimation(
            self.fig, 
            self.update, 
            interval=self.update_interval,
            blit=True,
            cache_frame_data=False
        )
        plt.show()

    def __del__(self):
        self.stop_audio()

    def on_key_press(self, event):
        """Handle keyboard events"""
        if event.key == ' ':
            self.toggle_pause()
        elif event.key == 'right' and self.paused:
            self.seek(5.0)
        elif event.key == 'left' and self.paused:
            self.seek(-5.0)
    
    def toggle_pause(self):
        """Toggle between play and pause states"""
        if self.paused:
            self.paused = False
            
            if self.pause_start_time is not None:
                self.total_pause_time += time.time() - self.pause_start_time
                self.pause_start_time = None
                
            if self.start_time is None:
                self.start_time = time.time() - self.total_pause_time
                
            if not self.audio_playing:
                self.start_audio()
                
            self.pause_text.set_visible(False)
            
            print("Playback resumed")
        else:
            self.paused = True
            self.pause_start_time = time.time()
            
            self.stop_audio()
            
            self.pause_text.set_visible(True)
            
            print("Playback paused")
    
    def seek(self, seconds):
        """Seek forward/backward in the audio file"""
        if not self.start_time:
            self.start_time = time.time() - self.total_pause_time
            
        # Calculate current and new time positions    
        current_time = (time.time() - self.start_time - self.total_pause_time) * self.playback_speed
        new_time = max(0, min(current_time + seconds, len(self.audio_data) / self.sample_rate))
        
        # Stop current audio playback
        self.stop_audio()
        
        # Update position for both audio and visualization
        self.position = int(new_time * self.sample_rate)
        
        # Adjust start_time to ensure elapsed_time calculation in update() is correct
        self.start_time = time.time() - self.total_pause_time - (new_time / self.playback_speed)
        
        print(f"Seeking to {new_time:.2f} seconds (position {self.position})")
        
        # If we're not paused, restart audio from the new position
        if not self.paused:
            self.start_audio()
        
        # Force an update of the visualization - this will use the updated position
        self.update(None)

def main():
    parser = argparse.ArgumentParser(description='Visualize frequencies from a WAV file')
    parser.add_argument('wav_file', help='Path to the WAV file')
    parser.add_argument('--bins', type=int, default=64, help='Number of frequency bins')
    parser.add_argument('--smooth', type=float, default=0.5, help='Smoothing factor (0.0 to 1.0)')
    parser.add_argument('--interval', type=int, default=50, help='Update interval in milliseconds')
    
    args = parser.parse_args()
    
    visualizer = WavFrequencyVisualizer(
        args.wav_file,
        bin_count=args.bins,
        smooth_factor=args.smooth,
        update_interval=args.interval
    )
    visualizer.run()

def test_visualizer():
    wav_file_path = 'Rock-01.wav'
    visualizer = WavFrequencyVisualizer(
        wav_file=wav_file_path,
        bin_count=64,
        smooth_factor=0.5,
        update_interval=50,
        playback_speed=1.0,
        # Negative latency: audio starts earlier than visualization
        # Positive latency: audio starts later than visualization
        audio_latency=-0.15,
        start_paused=True
    )
    
    visualizer.run()

if __name__ == "__main__":
    # main()
    test_visualizer()