import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.io import wavfile
import os

def quantize(data, levels):
    min_val, max_val = np.min(data), np.max(data)
    step = (max_val - min_val) / levels
    quantized = np.round((data - min_val) / step) * step + min_val
    return quantized

def run_length_encode(data):
    flat_data = data.flatten()
    encoded = []
    count = 1
    for i in range(1, len(flat_data)):
        if flat_data[i] == flat_data[i-1]:
            count += 1
        else:
            encoded.append((flat_data[i-1], count))
            count = 1
    encoded.append((flat_data[-1], count))
    return encoded

def main():
    # Setup standard relative output directory
    output_dir = 'outputs'
    os.makedirs(output_dir, exist_ok=True)

    print("1. Generating Audio Input...")
    fs = 44100  # Sampling rate
    t = np.linspace(0, 1, fs)  # 1 second duration
    clean_signal = np.sin(2 * np.pi * 440 * t)
    noise = np.random.normal(0, 0.5, clean_signal.shape)
    silence = np.zeros(fs // 2)  # 0.5 seconds of silence

    noisy_signal = np.concatenate((clean_signal + noise, silence))
    clean_with_silence = np.concatenate((clean_signal, silence))
    time_axis = np.linspace(0, 1.5, len(noisy_signal))

    # Plotting
    plt.figure(figsize=(12, 6))
    plt.subplot(2, 1, 1)
    plt.title("Original Clean Signal (with silence)")
    plt.plot(time_axis, clean_with_silence)
    plt.subplot(2, 1, 2)
    plt.title("Signal with Noise and Silence")
    plt.plot(time_axis, noisy_signal, color='orange')
    plt.tight_layout()
    
    # Save plot using os.path.join for OS compatibility
    plot_path = os.path.join(output_dir, 'audio_plots.png')
    plt.savefig(plot_path)
    print(f"-> Plots saved to {plot_path}")
    
    print("2. Applying STFT (Transform to Frequency Domain)...")
    f, t_stft, Zxx = signal.stft(noisy_signal, fs, nperseg=1024)
    magnitudes = np.abs(Zxx)
    phases = np.angle(Zxx)

    print("3. Quantizing Magnitude Data...")
    quantized_magnitudes = quantize(magnitudes, 16)

    print("4. Encoding with Run-Length Encoding...")
    compressed_audio = run_length_encode(quantized_magnitudes)
    print(f"-> Compressed array length: {len(compressed_audio)} pairs")

    print("5. Decoding and Evaluation...")
    # Reconstruct the complex STFT matrix using the quantized magnitudes and original phases
    reconstructed_Zxx = quantized_magnitudes * np.exp(1j * phases)
    _, decoded_audio = signal.istft(reconstructed_Zxx, fs)
    
    # Truncate to original length in case ISTFT added padding
    decoded_audio = decoded_audio[:len(noisy_signal)]

    # Save the reconstructed audio
    wav_path = os.path.join(output_dir, 'decompressed_audio.wav')
    wavfile.write(wav_path, fs, decoded_audio.astype(np.float32))
    print(f"-> Decompressed audio saved to {wav_path}")

    # Calculate SNR (Signal-to-Noise Ratio)
    power_signal = np.sum(noisy_signal**2)
    power_noise = np.sum((noisy_signal - decoded_audio)**2)
    
    # Handle edge case where power_noise is 0 to avoid log10 errors
    if power_noise == 0:
        print("-> Evaluation: Audio SNR is Infinite (Perfect reconstruction)")
    else:
        snr = 10 * np.log10(power_signal / power_noise)
        print(f"-> Evaluation: Audio SNR is {snr:.2f} dB")

if __name__ == "__main__":
    main()