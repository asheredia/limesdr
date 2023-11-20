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
#SoapySDRDevice_setGain(SOAPY_SDR_RX, 0)
#SoapySDRDevice_setBandwidth(SOAPY_SDR_RX, 250e3)
prev_second = -1
# Streaming 
rx_stream = sdr.setupStream(SOAPY_SDR_RX, SOAPY_SDR_CF32)
sdr.activateStream(rx_stream)

# Parámetros de captura
sample_rate = 1e6
record_seconds = 15
num_samples = sample_rate * record_seconds

# Archivo de captura
wav_file = wave.open("samples.wav", "wb") 
wav_file.setnchannels(2)
wav_file.setsampwidth(2)
wav_file.setframerate(sample_rate)

# Buffer de samples
samples = np.array([0]*1024, np.complex64) 

# Captura de x segundos

print("Grabando...")
start_time = time.time()

while (time.time() - start_time) < record_seconds:
#    current_time = time.time()
 #   elapsed_time = current_time - start_time
  #  print(f"Tiempo transcurrido: {elapsed_time} segundos")
    elapsed_time = round(time.time() - start_time)
    print(f"Segundo: {elapsed_time}")
    sample_count = 0
    sample_count += 1
    print(f"Muestras en segundo {elapsed_time}: {sample_count}")
    if elapsed_time > prev_second:
        sample_count = 0
        prev_second = elapsed_time
    # Leer muestras del SDR
    sr = sdr.readStream(rx_stream, [samples], len(samples))

#
    # Escribir las muestras en el archivo WAV
    for i in range(len(samples)):
        # Obtener los valores I y Q de la muestra compleja
        I = int(np.real(samples[i]) * 32767)
        Q = int(np.imag(samples[i]) * 32767)
        # Empaquetar los valores I y Q en un dato de 4 bytes
        data = struct.pack('<hh', I, Q)
        # Escribir el dato en el archivo WAV
        wav_file.writeframes(data)

print("Grabación finalizada")
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
