import librosa
import numpy as np
from numpyencoder import NumpyEncoder
import pandas as pd

import io

import requests
import soundfile as sf
from flask import Flask, request, Response
import flask
import json

import os
# from urllib.request import urlopen
import tempfile
import shutil

import easyocr
reader = easyocr.Reader(['en']) # this needs to run only once to load the model into memory

app = Flask(__name__)

'''
    function: extract_features
    input: path to mp3 files
    output: csv file containing features extracted
    
    This function reads the content in a directory and for each mp3 file detected
    reads the file and extracts relevant features using librosa library for audio
    signal processing
'''
@app.route('/audioFeatures', methods=['POST', 'OPTIONS'])
def audioFeatures():
    # id = 1  # Song ID

    if (request.method == 'OPTIONS'):
        # print('got options 1')
        response = flask.Response()
        response.headers['Access-Control-Allow-Origin'] = '*'
        response.headers['Access-Control-Allow-Headers'] = '*'
        response.headers['Access-Control-Allow-Methods'] = '*'
        response.headers['Access-Control-Expose-Headers'] = '*'
        response.headers['Cross-Origin-Opener-Policy'] = 'same-origin'
        response.headers['Cross-Origin-Embedder-Policy'] = 'require-corp'
        response.headers['Cross-Origin-Resource-Policy'] = 'cross-origin'
        # print('got options 2')
        return response

    # get the body bytes
    body = request.get_data()

    # feature_set = pd.DataFrame()  # Feature Matrix
    feature_set = {}
    
    # Individual Feature Vectors
    # songname_vector = pd.Series()
    # tempo_vector = pd.Series()
    # total_beats = pd.Series()
    # average_beats = pd.Series()
    # chroma_stft_mean = pd.Series()
    # chroma_stft_std = pd.Series()
    # chroma_stft_var = pd.Series()
    # chroma_cq_mean = pd.Series()
    # chroma_cq_std = pd.Series()
    # chroma_cq_var = pd.Series()
    # chroma_cens_mean = pd.Series()
    # chroma_cens_std = pd.Series()
    # chroma_cens_var = pd.Series()
    # mel_mean = pd.Series()
    # mel_std = pd.Series()
    # mel_var = pd.Series()
    # mfcc_mean = pd.Series()
    # mfcc_std = pd.Series()
    # mfcc_var = pd.Series()
    # mfcc_delta_mean = pd.Series()
    # mfcc_delta_std = pd.Series()
    # mfcc_delta_var = pd.Series()
    # rms_mean = pd.Series()
    # rms_std = pd.Series()
    # rms_var = pd.Series()
    # cent_mean = pd.Series()
    # cent_std = pd.Series()
    # cent_var = pd.Series()
    # spec_bw_mean = pd.Series()
    # spec_bw_std = pd.Series()
    # spec_bw_var = pd.Series()
    # contrast_mean = pd.Series()
    # contrast_std = pd.Series()
    # contrast_var = pd.Series()
    # rolloff_mean = pd.Series()
    # rolloff_std = pd.Series()
    # rolloff_var = pd.Series()
    # poly_mean = pd.Series()
    # poly_std = pd.Series()
    # poly_var = pd.Series()
    # tonnetz_mean = pd.Series()
    # tonnetz_std = pd.Series()
    # tonnetz_var = pd.Series()
    # zcr_mean = pd.Series()
    # zcr_std = pd.Series()
    # zcr_var = pd.Series()
    # harm_mean = pd.Series()
    # harm_std = pd.Series()
    # harm_var = pd.Series()
    # perc_mean = pd.Series()
    # perc_std = pd.Series()
    # perc_var = pd.Series()
    # frame_mean = pd.Series()
    # frame_std = pd.Series()
    # frame_var = pd.Series()
    
    # Traversing over each file in path
    # file_data = [f for f in listdir(path) if isfile (join(path, f))]
    # for line in file_data:
    #     if ( line[-1:] == '\n' ):
    #         line = line[:-1]

    # Reading Song
    # songname = path + line
    out_dir = tempfile.mkdtemp()
    songname = os.path.join(out_dir, 'song.mp3')
    # write body to file
    with open(songname, 'wb') as f:
        f.write(body)

    y, sr = librosa.load(songname)
    # y, sr = sf.read(io.BytesIO(body))
    S = np.abs(librosa.stft(y))
    
    # Extracting Features
    tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
    chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
    chroma_cq = librosa.feature.chroma_cqt(y=y, sr=sr)
    chroma_cens = librosa.feature.chroma_cens(y=y, sr=sr)
    melspectrogram = librosa.feature.melspectrogram(y=y, sr=sr)
    rms = librosa.feature.rms(y=y)
    cent = librosa.feature.spectral_centroid(y=y, sr=sr)
    spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    contrast = librosa.feature.spectral_contrast(S=S, sr=sr)
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    poly_features = librosa.feature.poly_features(S=S, sr=sr)
    tonnetz = librosa.feature.tonnetz(y=y, sr=sr)
    zcr = librosa.feature.zero_crossing_rate(y)
    harmonic = librosa.effects.harmonic(y)
    percussive = librosa.effects.percussive(y)
    
    mfcc = librosa.feature.mfcc(y=y, sr=sr)
    mfcc_delta = librosa.feature.delta(mfcc)

    onset_frames = librosa.onset.onset_detect(y=y, sr=sr)
    frames_to_time = librosa.frames_to_time(onset_frames[:20], sr=sr)
    
    # Transforming Features
    # songname_vector.set_value(id, line)  # song name
    # tempo_vector.set_value(id, tempo)  # tempo
    # total_beats.set_value(id, sum(beats))  # beats
    # average_beats.set_value(id, np.average(beats))
    # chroma_stft_mean.set_value(id, np.mean(chroma_stft))  # chroma stft
    # chroma_stft_std.set_value(id, np.std(chroma_stft))
    # chroma_stft_var.set_value(id, np.var(chroma_stft))
    # chroma_cq_mean.set_value(id, np.mean(chroma_cq))  # chroma cq
    # chroma_cq_std.set_value(id, np.std(chroma_cq))
    # chroma_cq_var.set_value(id, np.var(chroma_cq))
    # chroma_cens_mean.set_value(id, np.mean(chroma_cens))  # chroma cens
    # chroma_cens_std.set_value(id, np.std(chroma_cens))
    # chroma_cens_var.set_value(id, np.var(chroma_cens))
    # mel_mean.set_value(id, np.mean(melspectrogram))  # melspectrogram
    # mel_std.set_value(id, np.std(melspectrogram))
    # mel_var.set_value(id, np.var(melspectrogram))
    # mfcc_mean.set_value(id, np.mean(mfcc))  # mfcc
    # mfcc_std.set_value(id, np.std(mfcc))
    # mfcc_var.set_value(id, np.var(mfcc))
    # mfcc_delta_mean.set_value(id, np.mean(mfcc_delta))  # mfcc delta
    # mfcc_delta_std.set_value(id, np.std(mfcc_delta))
    # mfcc_delta_var.set_value(id, np.var(mfcc_delta))
    # rms_mean.set_value(id, np.mean(rms))  # rms
    # rms_std.set_value(id, np.std(rms))
    # rms_var.set_value(id, np.var(rms))
    # cent_mean.set_value(id, np.mean(cent))  # cent
    # cent_std.set_value(id, np.std(cent))
    # cent_var.set_value(id, np.var(cent))
    # spec_bw_mean.set_value(id, np.mean(spec_bw))  # spectral bandwidth
    # spec_bw_std.set_value(id, np.std(spec_bw))
    # spec_bw_var.set_value(id, np.var(spec_bw))
    # contrast_mean.set_value(id, np.mean(contrast))  # contrast
    # contrast_std.set_value(id, np.std(contrast))
    # contrast_var.set_value(id, np.var(contrast))
    # rolloff_mean.set_value(id, np.mean(rolloff))  # rolloff
    # rolloff_std.set_value(id, np.std(rolloff))
    # rolloff_var.set_value(id, np.var(rolloff))
    # poly_mean.set_value(id, np.mean(poly_features))  # poly features
    # poly_std.set_value(id, np.std(poly_features))
    # poly_var.set_value(id, np.var(poly_features))
    # tonnetz_mean.set_value(id, np.mean(tonnetz))  # tonnetz
    # tonnetz_std.set_value(id, np.std(tonnetz))
    # tonnetz_var.set_value(id, np.var(tonnetz))
    # zcr_mean.set_value(id, np.mean(zcr))  # zero crossing rate
    # zcr_std.set_value(id, np.std(zcr))
    # zcr_var.set_value(id, np.var(zcr))
    # harm_mean.set_value(id, np.mean(harmonic))  # harmonic
    # harm_std.set_value(id, np.std(harmonic))
    # harm_var.set_value(id, np.var(harmonic))
    # perc_mean.set_value(id, np.mean(percussive))  # percussive
    # perc_std.set_value(id, np.std(percussive))
    # perc_var.set_value(id, np.var(percussive))
    # frame_mean.set_value(id, np.mean(frames_to_time))  # frames
    # frame_std.set_value(id, np.std(frames_to_time))
    # frame_var.set_value(id, np.var(frames_to_time))  
    
    # Concatenating Features into one csv and json format
    # feature_set['song_name'] = line  # song name
    feature_set['tempo'] = tempo  # tempo 
    feature_set['total_beats'] = sum(beats)  # beats
    feature_set['average_beats'] = np.average(beats)
    feature_set['chroma_stft_mean'] = np.mean(chroma_stft)  # chroma stft
    feature_set['chroma_stft_std'] = np.std(chroma_stft)
    feature_set['chroma_stft_var'] = np.var(chroma_stft)
    feature_set['chroma_cq_mean'] = np.mean(chroma_cq)  # chroma cq
    feature_set['chroma_cq_std'] = np.std(chroma_cq)
    feature_set['chroma_cq_var'] = np.var(chroma_cq)
    feature_set['chroma_cens_mean'] = np.mean(chroma_cens)  # chroma cens
    feature_set['chroma_cens_std'] = np.std(chroma_cens)
    feature_set['chroma_cens_var'] = np.var(chroma_cens)
    feature_set['melspectrogram_mean'] = np.mean(melspectrogram)  # melspectrogram
    feature_set['melspectrogram_std'] = np.std(melspectrogram)
    feature_set['melspectrogram_var'] = np.var(melspectrogram)
    feature_set['mfcc_mean'] = np.mean(mfcc)  # mfcc
    feature_set['mfcc_std'] = np.std(mfcc)
    feature_set['mfcc_var'] = np.var(mfcc)
    feature_set['mfcc_delta_mean'] = np.mean(mfcc_delta)  # mfcc delta
    feature_set['mfcc_delta_std'] = np.std(mfcc_delta)
    feature_set['mfcc_delta_var'] = np.var(mfcc_delta)
    feature_set['rms_mean'] = np.mean(rms)  # rms
    feature_set['rms_std'] = np.std(rms)
    feature_set['rms_var'] = np.var(rms)
    feature_set['cent_mean'] = np.mean(cent)  # cent
    feature_set['cent_std'] = np.std(cent)
    feature_set['cent_var'] = np.var(cent)
    feature_set['spec_bw_mean'] = np.mean(spec_bw)  # spectral bandwidth
    feature_set['spec_bw_std'] = np.std(spec_bw)
    feature_set['spec_bw_var'] = np.var(spec_bw)
    feature_set['contrast_mean'] = np.mean(contrast)  # contrast
    feature_set['contrast_std'] = np.std(contrast)
    feature_set['contrast_var'] = np.var(contrast)
    feature_set['rolloff_mean'] = np.mean(rolloff)  # rolloff
    feature_set['rolloff_std'] = np.std(rolloff)
    feature_set['rolloff_var'] = np.var(rolloff)
    feature_set['poly_mean'] = np.mean(poly_features)  # poly features
    feature_set['poly_std'] = np.std(poly_features)
    feature_set['poly_var'] = np.var(poly_features)
    feature_set['tonnetz_mean'] = np.mean(tonnetz)  # tonnetz
    feature_set['tonnetz_std'] = np.std(tonnetz)
    feature_set['tonnetz_var'] = np.var(tonnetz)
    feature_set['zcr_mean'] = np.mean(zcr)  # zero crossing rate
    feature_set['zcr_std'] = np.std(zcr)
    feature_set['zcr_var'] = np.var(zcr)
    feature_set['harm_mean'] = np.mean(harmonic)  # harmonic
    feature_set['harm_std'] = np.std(harmonic)
    feature_set['harm_var'] = np.var(harmonic)
    feature_set['perc_mean'] = np.mean(percussive)  # percussive
    feature_set['perc_std'] = np.std(percussive)
    feature_set['perc_var'] = np.var(percussive)
    feature_set['frame_mean'] = np.mean(frames_to_time)  # frames
    feature_set['frame_std'] = np.std(frames_to_time)
    feature_set['frame_var'] = np.var(frames_to_time)

    # json_string = json.dumps(str(feature_set))
    json_string = json.dumps(feature_set, cls=NumpyEncoder)

    shutil.rmtree(out_dir)

    response = flask.Response(json_string)
    response.headers['Content-Type'] = 'application/json'
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Headers'] = '*'
    response.headers['Access-Control-Allow-Methods'] = '*'
    response.headers['Access-Control-Expose-Headers'] = '*'
    response.headers['Cross-Origin-Opener-Policy'] = 'same-origin'
    response.headers['Cross-Origin-Embedder-Policy'] = 'require-corp'
    response.headers['Cross-Origin-Resource-Policy'] = 'cross-origin'
    # print('got options 2')
    return response

@app.route("/ddc", methods=["POST", "OPTIONS"])
def ddc():
    if (flask.request.method == "OPTIONS"):
        # print("got options 1")
        response = flask.Response()
        response.headers["Access-Control-Allow-Origin"] = "*"
        response.headers["Access-Control-Allow-Headers"] = "*"
        response.headers["Access-Control-Allow-Methods"] = "*"
        response.headers["Access-Control-Expose-Headers"] = "*"
        response.headers["Cross-Origin-Opener-Policy"] = "same-origin"
        response.headers["Cross-Origin-Embedder-Policy"] = "require-corp"
        response.headers["Cross-Origin-Resource-Policy"] = "cross-origin"
        # print("got options 2")
        return response

    # get the body bytes
    body = request.get_data()

    proxyHeaders = {}
    proxyHeaders['Content-Type'] = request.headers['Content-Type']
    proxyRequest = requests.post(
        "http://127.0.0.1:3456/choreograph",
        headers=proxyHeaders,
        data=body
    )
    # proxy the response content back to the client
    response = flask.Response(proxyRequest.content, status=proxyRequest.status_code)
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Headers"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "*"
    response.headers["Access-Control-Expose-Headers"] = "*"
    response.headers["Cross-Origin-Opener-Policy"] = "same-origin"
    response.headers["Cross-Origin-Embedder-Policy"] = "require-corp"
    response.headers["Cross-Origin-Resource-Policy"] = "cross-origin"
    return response

@app.route("/ocr", methods=["POST", "OPTIONS"])
def ocr():
    if (flask.request.method == "OPTIONS"):
        # print("got options 1")
        response = flask.Response()
        response.headers["Access-Control-Allow-Origin"] = "*"
        response.headers["Access-Control-Allow-Headers"] = "*"
        response.headers["Access-Control-Allow-Methods"] = "*"
        response.headers["Access-Control-Expose-Headers"] = "*"
        response.headers["Cross-Origin-Opener-Policy"] = "same-origin"
        response.headers["Cross-Origin-Embedder-Policy"] = "require-corp"
        response.headers["Cross-Origin-Resource-Policy"] = "cross-origin"
        # print("got options 2")
        return response

    # get the body bytes
    body = request.get_data()

    text = reader.readtext(body)
    print(f"got result: {text}")
    text_string = json.dumps(text, cls=NumpyEncoder)

    # proxy the response content back to the client
    response = flask.Response(text_string)
    response.headers["Content-Type"] = "text/plain"
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Headers"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "*"
    response.headers["Access-Control-Expose-Headers"] = "*"
    response.headers["Cross-Origin-Opener-Policy"] = "same-origin"
    response.headers["Cross-Origin-Embedder-Policy"] = "require-corp"
    response.headers["Cross-Origin-Resource-Policy"] = "cross-origin"
    return response

# get the port from env
port = int(os.environ.get('PORT', 80))

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=port, threaded=True)