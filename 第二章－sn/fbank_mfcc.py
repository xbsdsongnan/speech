import librosa
import numpy as np
import matplotlib.pyplot as plt
import librosa.display
from scipy.fftpack import dct
##from matplotlib.colors import BoundaryNorm
# If you want to see the spectrogram picture
# import matplotlib
# matplotlib.use('Agg')
# import matplotlib.pyplot as plt
# def plot_spectrogram(spec, note,file_name):
#     """Draw the spectrogram picture
#         :param spec: a feature_dim by num_frames array(real)
#         :param note: title of the picture
#         :param file_name: name of the file
#     """ 
#     fig = plt.figure(figsize=(20, 5))
#     heatmap = plt.pcolor(spec)
#     fig.colorbar(mappable=heatmap)
#     plt.xlabel('Time(s)')
#     plt.ylabel(note)
#     plt.tight_layout()
#     plt.savefig(file_name)

# 绘制频谱图
def plot_spectrogram(spec, note):
    fig = plt.figure(figsize=(20, 5))
    heatmap = plt.pcolor(spec)
    fig.colorbar(mappable=heatmap)
    plt.xlabel('Time(s)')
    plt.ylabel(note)
    plt.tight_layout()

#preemphasis config 
alpha = 0.97 #滤波器系数（α）

# Enframe config
frame_len = 400      # 25ms, fs=16kHz
frame_shift = 160    # 10ms, fs=15kHz
fft_len = 512        #参与FFT运算的512个数据

# Mel filter config
num_filter = 23
num_mfcc = 12

# Read wav file
wav, fs = librosa.load('./test.wav', sr=None) #Librosa默认的采样率是22050，如果需要读取原始采样率，需要设定参数sr=None
##plt.plot(wav)
##plt.show()
#print(fs)
plt.figure()
librosa.display.waveplot(wav, fs)
plt.show()
# Pre-Emphasis
def preemphasis(signal, coeff=alpha):
    """perform preemphasis on the input signal.

        :param signal: The signal to filter.
        :param coeff: The preemphasis coefficient. 0 is no filter, default is 0.97.
        :returns: the filtered signal.
    """
    return np.append(signal[0], signal[1:] - coeff * signal[:-1]) #预加重
# Enframe with Hamming window function
def enframe(signal, frame_len=frame_len, frame_shift=frame_shift, win=np.hamming(frame_len)):
    """Enframe with Hamming widow function.

        :param signal: The signal be enframed
        :param win: window function, default Hamming
        :returns: the enframed signal, num_frames by frame_len array
    """
    
    num_samples = signal.size
    num_frames = np.floor((num_samples - frame_len) / frame_shift)+1
    frames = np.zeros((int(num_frames),frame_len))
    for i in range(int(num_frames)):
        frames[i,:] = signal[i*frame_shift:i*frame_shift + frame_len] 
        frames[i,:] = frames[i,:] * win

    return frames
    

def get_spectrum(frames, fft_len=fft_len):
    """Get spectrum using fft
        :param frames: the enframed signal, num_frames by frame_len array
        :param fft_len: FFT length, default 512
        :returns: spectrum, a num_frames by fft_len/2+1 array (real)
    """
    cFFT = np.fft.fft(frames, n=fft_len)
    valid_len = int(fft_len / 2 ) + 1
    spectrum = np.abs(cFFT[:,0:valid_len])
    ##print(spectrum.shape)
    return spectrum
def fbank(spectrum, num_filter = num_filter):
    """Get mel filter bank feature from spectrum
        :param spectrum: a num_frames by fft_len/2+1 array(real)
        :param num_filter: mel filters number, default 23
        :returns: fbank feature, a num_frames by num_filter array 
        DON'T FORGET LOG OPRETION AFTER MEL FILTER!
    """
    low_freq_mel = 0
    high_freq_mel = 2595 * np.log10(1 + (fs / 2) / 700)
    #print(low_freq_mel, high_freq_mel)
    mel_points = np.linspace(low_freq_mel, high_freq_mel, num_filter + 2)  # 所有的mel中心点，为了方便后面计算mel滤波器组，左右两边各补一个中心点
    hz_points = 700 * (10 ** (mel_points / 2595) - 1)

    #fbank_feats = librosa.feature.melspectrogram(fbank_feats=feats)  # 使用stft频谱求Mel频谱
    feats = np.zeros(( num_filter, int(fft_len/2+1))) # 各个mel滤波器在能量谱对应点的取值
    bin = (hz_points/(fs/2)) * (fft_len/2)  # 各个mel滤波器中心点对应FFT的区域编码，找到有值的位置
    #print(bin)
    #print(bin.shape)
    for i in range(1, num_filter + 1):
        left = int(bin[i-1])
        center = int(bin[i])
        right = int(bin[i+1])
        for j in range(left, center):
            feats[i-1, j+1] = (j+1 - bin[i-1]) / (bin[i] - bin[i-1])
        #print(feats.shape)
        for j in range(center, right):
            feats[i-1, j+1] = (bin[i+1] - (j+1)) / (bin[i+1] - bin[i])
    #print(feats.shape)
    fbank = np.dot(spectrum, feats.T)
    #print(fbank.shape)
    fbank = np.where(fbank == 0, np.finfo(float).eps, fbank)
    fbank = 20 * np.log10(fbank)  # dB
    #print(fbank.shape)
    plot_spectrogram(fbank.T, 'Fbank')
    plt.show()
    return fbank
    
def mfcc(fbank, num_mfcc = num_mfcc):
    """Get mfcc feature from fbank feature
        :param fbank: a num_frames by  num_filter array(real)
        :param num_mfcc: mfcc number, default 12
        :returns: mfcc feature, a num_frames by num_mfcc array 
    """
    feats = np.zeros((fbank.shape[0],num_mfcc))
    ##feats = dct(fbank, type=2, axis=1, norm='ortho')[:,:num_mfcc]
    feats = dct(fbank, type=2, axis=1, norm='ortho')[:, 1:(num_mfcc+1)]
    #print(feats)
    #print(feats.shape)
    plot_spectrogram(feats.T, 'MFCC')
    plt.show()
    return feats

def write_file(feats, file_name):
    """Write the feature to file
        :param feats: a num_frames by feature_dim array(real)
        :param file_name: name of the file
    """
    f=open(file_name,'w')
    (row,col) = feats.shape
    for i in range(row):
        f.write('[')
        for j in range(col):
            f.write(str(feats[i,j])+' ')
        f.write(']\n')
    f.close()


def main():
    wav, fs = librosa.load('./test.wav', sr=None)
    signal = preemphasis(wav)
    frames = enframe(signal)
    spectrum = get_spectrum(frames)
    fbank_feats = fbank(spectrum)
    mfcc_feats = mfcc(fbank_feats)
    #plot_spectrogram(fbank_feats, 'Filter Bank','fbank.png')
    write_file(fbank_feats,'./test.fbank')
    #plot_spectrogram(mfcc_feats.T, 'MFCC','mfcc.png')
    write_file(mfcc_feats,'./test.mfcc')

if __name__ == '__main__':
    main()
