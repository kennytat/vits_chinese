import os
import glob
import sys
import argparse
import logging
import json
import subprocess
import numpy as np
from scipy.io.wavfile import read
import torch
import shutil
from datetime import datetime
from pydub import AudioSegment
import librosa
import numpy as np
import soundfile as sf
import math
import srt
import hashlib
import re
import docx2txt
from pathlib import Path
MATPLOTLIB_FLAG = False

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logger = logging

class ParaStruct():
    def __init__(self, text, total_duration, start_time):
        self.text = text
        self.total_duration = total_duration
        self.start_time = start_time
        
def encode_filename(filename):
    print("Encoding filename:", filename)
    result = hashlib.md5(filename.encode())
    return result.hexdigest()
  
def new_dir_now():
    now = datetime.now() # current date and time
    date_time = now.strftime("%Y%m%d%H%M")
    return date_time

def pad_zero(s, th):
    num_str = str(s)
    while len(num_str) < th:
        num_str = '0' + num_str
    return num_str
  
def remove_comment(txt_input):
  pattern = "<comment>.*?</comment>"
  txt = re.sub(pattern, "", txt_input, flags=re.MULTILINE|re.DOTALL)
  return txt

def transcript_to_srt(txt_input):
  try:
    srt = ""
    subs = re.sub(r'\n(?!\d{2,3}:\d{2}:\d{2}:\d{2}|\n)', ' ', txt_input).strip()
    subs = re.sub(r'\n+', '\n', subs).strip().splitlines()
    odd = subs[0:][::2]
    even = subs[1:][::2]
    regex = re.compile(r'\d{2,3}:\d{2}:\d{2}:\d{2}\s\-\s\d{2,3}:\d{2}:\d{2}:\d{2}')
    if len(list(filter(regex.match, odd))) == len(odd):
        print('Transcript valid - convert to SRT::')
        for i in range(len(odd)):
          txt = "{index}\n{time}\n{content}\n\n".format(index = (i+1), time = odd[i].replace("-","-->"), content = even[i])
          srt = srt + txt
        # print(srt)
        return srt.strip()
    else:
        print('Transcript not valid - parse normally!!')
        return txt_input
  except:
    print("Transcript not valid - parse normally!!")
    return txt_input
  
def file_to_paragraph(file):
  txt = ''
  file_extension = Path(file).suffix
  if file_extension == '.doc' or file_extension == '.docx':
    txt = docx2txt.process(file)
  if file_extension == '.txt' or file_extension == '.srt':
    txt = open(file, 'r').read()
  return txt_to_paragraph(txt)
  
def txt_to_paragraph(txt_input):
  ## Try parsing SRT, if fail then parse normally
  srt_input = transcript_to_srt(txt_input)
  # print("SRT:: \n", srt)
  try:
    subs = list(srt.parse(srt_input))
    for i, para in enumerate(subs):
      subs[i].duration = (para.end - para.start).total_seconds()
      # subs[i].start_silence = para.start.total_seconds() if i <= 0 else (para.start - subs[i - 1].end).total_seconds()
      subs[i].start_time = para.start.total_seconds()
    return [ParaStruct(para.content, para.duration, para.start_time) for para in subs]
  except:
    print("Input txt is not SRT - parse normally::")
    paras = txt_input.lower()
    paras = remove_comment(paras)
    paras = paras.split("\n")
    # Each new line between paragraphs add more silence duration
    p_list = []
    for p in paras:
      last_el = len(p_list) - 1
      if p == 'sil' and last_el < len(p_list):
          if isinstance(p_list[last_el],int):
              p_list[last_el] = p_list[last_el] + 1
          else:
              p_list.append(1)
      else:
          p_list.append(ParaStruct(p, 0, 0))
    # paras = [x for x in paras if x]
    print("Total paras: {}".format(len(p_list)))
    print(p_list)
    return p_list
  
def combine_wav_segment(wav_list, output_file):
    print("synthesization done, start concatenating:: ")
    if len(wav_list) == 1 and wav_list[0].start_time == 0:
      # move wav_list[0] to output_file
      shutil.move(wav_list[0].wav_path, output_file)
      return (output_file, None)
    else:
      if wav_list[0].start_time > 0:
      ## If wav_list contain time code, concatenate by time code
        # Calculate the total duration of the combined audio tracks
        audio, sample_rate = librosa.load(wav_list[0].wav_path)
        print("last_audio_path:: ", wav_list[len(wav_list) - 1].wav_path)
        last_audio_duration = librosa.get_duration(filename=wav_list[len(wav_list) - 1].wav_path)
        print("last_audio_duration:: ", last_audio_duration)
        start_time = 0
        end_time = wav_list[len(wav_list) - 1].start_time + last_audio_duration
        total_duration = math.ceil(end_time - start_time)
        print("total_duration:: ", total_duration)
        # Calculate the total number of samples needed for the combined audio file
        total_samples = int(total_duration * sample_rate)
        print("total_samples:: ", total_samples)
        # Create blank combined_audio with total_duration
        combined_wav = np.zeros(total_samples)
        print("combined_audio:: ", combined_wav)
        # Add each audio track to the combined audio array and check if any track overlap each other
        wav_overlap = []
        for i in range(len(wav_list)):
            print("Combining wav:: ", i, wav_list[i].wav_path)
            # Calculate the start and end sample indices for the current audio track
            start_sample = int((wav_list[i].start_time - start_time) * sample_rate)
            print("start_sample:: ", start_sample)
            end_sample = start_sample + len(librosa.load(wav_list[i].wav_path)[0])
            print("end_sample:: ", end_sample)
            # Load the current audio track and copy it into the combined audio array
            audio, sample_rate = librosa.load(wav_list[i].wav_path)
            print("Current wav:: ", audio)
            combined_wav[start_sample:end_sample] = audio
            track_end_time = wav_list[i].start_time + librosa.get_duration(y=audio, sr=sample_rate)
            # desired_end_time = wav_list[i].start_time + wav_list[i].total_duration
            if i != len(wav_list) - 1 and track_end_time > wav_list[i+1].start_time:
              wav_list[i].track_end_time = track_end_time
              wav_list[i].line = i + 1
              wav_overlap.append(wav_list[i])
            print("wav_overlap list:: ", len(wav_overlap))
        sf.write(output_file, combined_wav, samplerate=sample_rate)
        log_file = None
        if len(wav_overlap) > 0:
          log_file = os.path.splitext(output_file)[0] + '.log'
          with open(log_file,'w') as errFile:
            errFile.write("These paras got overlap::\n" + "\n".join("Para:: {} | Start_at:: {} | End_at:: {}".format(str(item.line), str(datetime.timedelta(seconds=item.start_time)), str(datetime.timedelta(seconds=item.track_end_time))) for item in wav_overlap))
          print("Combined wav:: ", combined_wav)
        return (output_file, log_file)
      else:
        audio = AudioSegment.from_file(wav_list[0].wav_path, format="wav")
        # Concatenate the remaining audio files
        for file in wav_list[1:]:
            wav = AudioSegment.from_file(file.wav_path, format="wav")
            audio += wav
        # create the output file
        audio.export(output_file, format="wav")
        return (output_file, None)
        
def convert_voice(input_dir, model_dir):
  print("start convert_voice::", input_dir, model_dir)
  model_path = os.path.join(model_dir, "G.pth")
  config_path = os.path.join(model_dir, "config.json")
  output_dir = f'{input_dir}.out'
  os.system(f'svc infer -re -m {model_path} -c {config_path} {input_dir}')
  if os.path.exists(input_dir): shutil.rmtree(input_dir, ignore_errors=True)
  shutil.move(output_dir, input_dir)
  
def load_class(full_class_name):
    cls = None
    if full_class_name in globals():
        cls = globals()[full_class_name]
    else:
        if "." in full_class_name:
            import importlib
            module_name, cls_name = full_class_name.rsplit('.', 1)
            mod = importlib.import_module(module_name)
            cls = (getattr(mod, cls_name))
    return cls

def load_teacher(checkpoint_path, model):
    assert os.path.isfile(checkpoint_path)
    checkpoint_dict = torch.load(checkpoint_path, map_location='cpu')
    saved_state_dict = checkpoint_dict['model']
    if hasattr(model, 'module'):
        state_dict = model.module.state_dict()
    else:
        state_dict = model.state_dict()
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('enc_q') or k.startswith('flow'):
            new_state_dict[k] = saved_state_dict[k]
        else:
            new_state_dict[k] = v
    if hasattr(model, 'module'):
        model.module.load_state_dict(new_state_dict)
    else:
        model.load_state_dict(new_state_dict)
    return model


def load_checkpoint(checkpoint_path, model, optimizer=None):
    assert os.path.isfile(checkpoint_path)
    checkpoint_dict = torch.load(checkpoint_path, map_location="cpu")
    iteration = checkpoint_dict["iteration"]
    learning_rate = checkpoint_dict["learning_rate"]
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint_dict["optimizer"])
    saved_state_dict = checkpoint_dict["model"]
    if hasattr(model, "module"):
        state_dict = model.module.state_dict()
    else:
        state_dict = model.state_dict()
    new_state_dict = {}
    for k, v in state_dict.items():
        try:
            new_state_dict[k] = saved_state_dict[k]
        except:
            logger.info("%s is not in the checkpoint" % k)
            new_state_dict[k] = v
    if hasattr(model, "module"):
        model.module.load_state_dict(new_state_dict)
    else:
        model.load_state_dict(new_state_dict)
    logger.info(
        "Loaded checkpoint '{}' (iteration {})".format(checkpoint_path, iteration)
    )
    return model, optimizer, learning_rate, iteration


def save_checkpoint(model, optimizer, learning_rate, iteration, checkpoint_path):
    logger.info(
        "Saving model and optimizer state at iteration {} to {}".format(
            iteration, checkpoint_path
        )
    )
    if hasattr(model, "module"):
        state_dict = model.module.state_dict()
    else:
        state_dict = model.state_dict()
    torch.save(
        {
            "model": state_dict,
            "iteration": iteration,
            "optimizer": optimizer.state_dict(),
            "learning_rate": learning_rate,
        },
        checkpoint_path,
    )


def load_model(checkpoint_path, model):
    assert os.path.isfile(checkpoint_path)
    checkpoint_dict = torch.load(checkpoint_path, map_location="cpu")
    saved_state_dict = checkpoint_dict["model"]
    if hasattr(model, "module"):
        state_dict = model.module.state_dict()
    else:
        state_dict = model.state_dict()
    new_state_dict = {}
    for k, v in state_dict.items():
        try:
            new_state_dict[k] = saved_state_dict[k]
        except:
            logger.info("%s is not in the checkpoint" % k)
            new_state_dict[k] = v
    if hasattr(model, "module"):
        model.module.load_state_dict(new_state_dict)
    else:
        model.load_state_dict(new_state_dict)
    return model


def save_model(model, checkpoint_path):
    if hasattr(model, 'module'):
        state_dict = model.module.state_dict()
    else:
        state_dict = model.state_dict()
    torch.save({'model': state_dict}, checkpoint_path)


def summarize(
    writer,
    global_step,
    scalars={},
    histograms={},
    images={},
    audios={},
    audio_sampling_rate=22050,
):
    for k, v in scalars.items():
        writer.add_scalar(k, v, global_step)
    for k, v in histograms.items():
        writer.add_histogram(k, v, global_step)
    for k, v in images.items():
        writer.add_image(k, v, global_step, dataformats="HWC")
    for k, v in audios.items():
        writer.add_audio(k, v, global_step, audio_sampling_rate)


def latest_checkpoint_path(dir_path, regex="G_*.pth"):
    f_list = glob.glob(os.path.join(dir_path, regex))
    f_list.sort(key=lambda f: int("".join(filter(str.isdigit, f))))
    x = f_list[-1]
    print(x)
    return x


def plot_spectrogram_to_numpy(spectrogram):
    global MATPLOTLIB_FLAG
    if not MATPLOTLIB_FLAG:
        import matplotlib
        matplotlib.use("Agg")
        MATPLOTLIB_FLAG = True
        mpl_logger = logging.getLogger("matplotlib")
        mpl_logger.setLevel(logging.WARNING)
    import matplotlib.pylab as plt
    import numpy as np

    fig, ax = plt.subplots(figsize=(10, 2))
    im = ax.imshow(spectrogram, aspect="auto", origin="lower", interpolation="none")
    plt.colorbar(im, ax=ax)
    plt.xlabel("Frames")
    plt.ylabel("Channels")
    plt.tight_layout()

    fig.canvas.draw()
    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep="")
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close()
    return data


def plot_alignment_to_numpy(alignment, info=None):
    global MATPLOTLIB_FLAG
    if not MATPLOTLIB_FLAG:
        import matplotlib

        matplotlib.use("Agg")
        MATPLOTLIB_FLAG = True
        mpl_logger = logging.getLogger("matplotlib")
        mpl_logger.setLevel(logging.WARNING)
    import matplotlib.pylab as plt
    import numpy as np

    fig, ax = plt.subplots(figsize=(6, 4))
    im = ax.imshow(
        alignment.transpose(), aspect="auto", origin="lower", interpolation="none"
    )
    fig.colorbar(im, ax=ax)
    xlabel = "Decoder timestep"
    if info is not None:
        xlabel += "\n\n" + info
    plt.xlabel(xlabel)
    plt.ylabel("Encoder timestep")
    plt.tight_layout()

    fig.canvas.draw()
    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep="")
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close()
    return data


def load_wav_to_torch(full_path):
    sampling_rate, data = read(full_path)
    return torch.FloatTensor(data.astype(np.float32)), sampling_rate


def load_filepaths_and_text(filename, split="|"):
    with open(filename, encoding="utf-8") as f:
        filepaths_and_text = []
        for line in f:
            path_text = line.strip().split(split)
            filepaths_and_text.append(path_text)
    return filepaths_and_text


def get_hparams(init=True):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        default="./configs/bert_vits.json",
        help="JSON file for configuration",
    )
    parser.add_argument("-m", "--model", type=str, required=True, help="Model name")

    args = parser.parse_args()
    model_dir = os.path.join("./logs", args.model)

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    config_path = args.config
    config_save_path = os.path.join(model_dir, "config.json")
    if init:
        with open(config_path, "r") as f:
            data = f.read()
        with open(config_save_path, "w") as f:
            f.write(data)
    else:
        with open(config_save_path, "r") as f:
            data = f.read()
    config = json.loads(data)

    hparams = HParams(**config)
    hparams.model_dir = model_dir
    return hparams


def get_hparams_from_dir(model_dir):
    config_save_path = os.path.join(model_dir, "config.json")
    with open(config_save_path, "r") as f:
        data = f.read()
    config = json.loads(data)

    hparams = HParams(**config)
    hparams.model_dir = model_dir
    return hparams


def get_hparams_from_file(config_path):
    with open(config_path, "r") as f:
        data = f.read()
    config = json.loads(data)

    hparams = HParams(**config)
    return hparams


def check_git_hash(model_dir):
    source_dir = os.path.dirname(os.path.realpath(__file__))
    if not os.path.exists(os.path.join(source_dir, ".git")):
        logger.warn(
            "{} is not a git repository, therefore hash value comparison will be ignored.".format(
                source_dir
            )
        )
        return

    cur_hash = subprocess.getoutput("git rev-parse HEAD")

    path = os.path.join(model_dir, "githash")
    if os.path.exists(path):
        saved_hash = open(path).read()
        if saved_hash != cur_hash:
            logger.warn(
                "git hash values are different. {}(saved) != {}(current)".format(
                    saved_hash[:8], cur_hash[:8]
                )
            )
    else:
        open(path, "w").write(cur_hash)


def get_logger(model_dir, filename="train.log"):
    global logger
    logger = logging.getLogger(os.path.basename(model_dir))
    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter("%(asctime)s\t%(name)s\t%(levelname)s\t%(message)s")
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    h = logging.FileHandler(os.path.join(model_dir, filename))
    h.setLevel(logging.DEBUG)
    h.setFormatter(formatter)
    logger.addHandler(h)
    return logger


class HParams:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            if type(v) == dict:
                v = HParams(**v)
            self[k] = v

    def keys(self):
        return self.__dict__.keys()

    def items(self):
        return self.__dict__.items()

    def values(self):
        return self.__dict__.values()

    def __len__(self):
        return len(self.__dict__)

    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        return setattr(self, key, value)

    def __contains__(self, key):
        return key in self.__dict__

    def __repr__(self):
        return self.__dict__.__repr__()
