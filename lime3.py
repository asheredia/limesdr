import SoapySDR
from SoapySDR import * 
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftshift
import time

# Configuración SDR
sdr = SoapySDR.Device(dict(driver="lime"))
sdr.setSampleRate(SOAPY_SDR_RX, 0, 500e3)
sdr.setFrequency(SOAPY_SDR_RX, 0, 915e6)

# Streaming 
rx_stream = sdr.setupStream(SOAPY_SDR_RX, SOAPY_SDR_CF32)
sdr.activateStream(rx_stream)

# Parámetros de captura
sample_rate = 500e3 
record_seconds = 5
num_samples = sample_rate * record_seconds

# Buffer de samples
samples = np.array([0]*1024, np.complex64) 

# Captura de x segundos
print("Grabando...")
start_time = time.time()
while (time.time() - start_time) < record_seconds:
  sr = sdr.readStream(rx_stream, [samples], len(samples))
  print(samples) # Imprimir muestras  

print("Grabación finalizada")
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
