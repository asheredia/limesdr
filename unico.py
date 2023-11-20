import SoapySDR
from SoapySDR import * 
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftshift
import time
import wave
import struct
# Configuración SDR
sdr = SoapySDR.Device(dict(driver="rtlsdr"))
sdr.setSampleRate(SOAPY_SDR_RX, 0, 1e6)
sdr.setFrequency(SOAPY_SDR_RX, 0, 915e6)
#SoapySDRDevice_setAntenna(SOAPY_SDR_RX, 0)
#SoapySDRDevice_setGain(SOAPY_SDR_RX, 1)
#SoapySDRDevice_setBandwidth(SOAPY_SDR_RX, 250e3)

# Streaming 
rx_stream = sdr.setupStream(SOAPY_SDR_RX, SOAPY_SDR_CF32)
sdr.activateStream(rx_stream)

# Parámetros de captura
sample_rate = 1e6
record_seconds = 5
num_samples = sample_rate * record_seconds

# Archivo de captura
wav_file = wave.open("samplest.wav", "wb") 
wav_file.setnchannels(2)
wav_file.setsampwidth(2)
wav_file.setframerate(sample_rate)

# Buffer de samples
samples = np.array([0]*1024, np.complex64) 

# Captura de x segundos
print("Grabando...")
start_time = time.time()

# Read samples from the SDR
sr = sdr.readStream(rx_stream, [samples], len(samples))

    # Write the samples to the WAV file
#for i in range(len(samples)):
       # Convert the complex sample to two 16-bit integers
    # Pack the two integers into a single 4-byte string

        # Write the data to the WAV file

while (time.time() - start_time) < record_seconds:
#  sr = sdr.readStream(rx_stream, [samples], len(samples))
#  #wav_file.writeframes(samples.astype(np.complex64).tobytes())
#  wav_file.writeframes(samples.tobytes())
#  print(samples) # Imprimir muestras  
    left = int(np.real(samples) * 32767)
    right = int(np.imag(samples) * 32767)
    data = struct.pack('<hh', left, right)
    wav_file.writeframes(data)

print("Grabación finalizada")
print(num_samples)
wav_file.close()
# Detener streaming
sdr.deactivateStream(rx_stream)
sdr.closeStream(rx_stream)

# FFT para ver en frecuencia
fft_signal = fftshift(fft(samples))
fft_freq = np.linspace(-sample_rate/2, sample_rate/2, len(fft_signal))

# Gráficas
fig, (ax1, ax2) = plt.subplots(2)
ax1.plot(np.abs(samples))
ax1.set_ylabel('Amplitud')
ax2.plot(fft_freq, 20*np.log10(np.abs(fft_signal)))
ax2.set_ylabel('Magnitud (dB)')
ax2.set_xlabel('Frecuencia (Hz)')
fig.tight_layout()
plt.show()
