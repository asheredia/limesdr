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

# Streaming 
rx_stream = sdr.setupStream(SOAPY_SDR_RX, SOAPY_SDR_CF32)
sdr.activateStream(rx_stream)

# Parámetros de captura
sample_rate = 1e6
record_seconds = 15
num_samples = sample_rate * record_seconds

# Archivo de captura
wav_file = wave.open("muestras.wav", "wb") 
wav_file.setnchannels(2)
wav_file.setsampwidth(2)
wav_file.setframerate(sample_rate)

# Buffer de samples
samples = np.array([0]*1024, np.complex64) 

# Captura de x segundos
print("Grabando...")
start_time = time.time()

while (time.time() - start_time) < record_seconds:
    # Leer muestras del SDR
    sr = sdr.readStream(rx_stream, [samples], len(samples))

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
