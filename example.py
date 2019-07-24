import numpy as np
import matplotlib.pyplot as plt
from scipy.io.wavfile import write
import matplotlib2tikz as tikz
from scipy import signal

# data = np.random.uniform(-1, 1, 44100)  # 44100 random samples between -1 and 1
# scaled = np.int16(data/np.max(np.abs(data)) * 32767)
# write('test.wav', 44100, scaled)


def create_sine(freq):
    sampling_rate = 44100
    tmax = 2
    t = np.linspace(0, tmax, sampling_rate*tmax) + np.random.uniform(0, 0.2)
    f = freq*(2*np.pi)
    # sine (omega*t)
    sine = np.sin(t*f)
    return sine


def create_silence():
    tmax = 2
    t = np.linspace(0, tmax, 44100*tmax) + np.random.uniform(0, 0.2)
    return 0*t


def normalize_wav(array):
    return np.int16(array/np.max(np.abs(array)) * 32767)


def write_wave(name, array):
    write(name, 44100, normalize_wav(array))


sine697 = create_sine(697)
write_wave('./sounds/analog_phone/sine697.wav', sine697)

sine770 = create_sine(770)
write_wave('./sounds/analog_phone/sine770.wav', sine770)

sine852 = create_sine(852)
write_wave('./sounds/analog_phone/sine852.wav', sine852)

sine1209 = create_sine(1209)
write_wave('./sounds/analog_phone/sine1209.wav', sine1209)

sine1336 = create_sine(1336)
write_wave('./sounds/analog_phone/sine1336.wav', sine1336)

sine1477 = create_sine(1477)
write_wave('./sounds/analog_phone/sine1477.wav', sine1477)


# 1: 697 and 1209
phone1 = sine697 + sine1209
write_wave('./sounds/analog_phone/phone1.wav', phone1)

# 5: 770 and 1336
phone5 = sine770 + sine1336
write_wave('./sounds/analog_phone/phone5.wav', phone5)

# 9: 852 and 1477
phone9 = sine852 + sine1477
write_wave('./sounds/analog_phone/phone9.wav', phone9)


zeros = create_silence()

seq = np.concatenate([zeros, phone1, zeros, phone5, zeros, phone9, zeros])
write_wave('./sounds/analog_phone/phone_seq.wav', seq)

seq_plot = signal.resample(seq, 30000)
plt.plot(seq)
tikz.save('./plots/seq_plot.tex', standalone=True)
plt.show()

f, t, Zxx = signal.stft(seq, fs=44100, nperseg=44100/8)

plt.imshow(np.abs(Zxx[:300, :]))
plt.xlabel('time')
plt.ylabel('fft bin')
# tikz.save('./plots/stft_full.tex', standalone=True)
plt.savefig('./plots/stft_full.pdf')
plt.show()

# plot at 760
plt.title('STFT during key 1')
plt.plot(f[:300], np.abs(Zxx)[:300, 50])
plt.xlabel('frequency')
plt.ylabel('magnitude')
tikz.save('./plots/stft_key_1.tex', standalone=True)
plt.show()

# plot at 2409
plt.title('STFT during key 2')
plt.plot(f[:300], np.abs(Zxx)[:300, 110])
plt.xlabel('frequency')
plt.ylabel('magnitude')
tikz.save('./plots/stft_key_2.tex', standalone=True)
plt.show()

# plot at 3739
plt.title('STFT during key 3')
plt.plot(f[:300], np.abs(Zxx)[:300, 173])
plt.xlabel('frequency')
plt.ylabel('magnitude')
tikz.save('./plots/stft_key_3.tex', standalone=True)
plt.show()


# full seq fft
fft_seq = np.fft.rfft(seq)
plt.plot(np.abs(fft_seq)[:30000])
plt.xlabel('N/2')
plt.ylabel('magnitude')
tikz.save('./plots/fft.tex', standalone=True)
plt.show()
