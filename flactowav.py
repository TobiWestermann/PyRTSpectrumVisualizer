import soundfile as sf

def convert_flac_to_wav(input_flac_path, output_wav_path):
    data, samplerate = sf.read(input_flac_path)
    
    sf.write(output_wav_path, data, samplerate)

if __name__ == "__main__":
    input_flac_path = 'AuchWennDuDaBist.flac'
    output_wav_path = 'LordFolter.wav'
    convert_flac_to_wav(input_flac_path, output_wav_path)