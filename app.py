from models import SynthesizerTrn
from vits_pinyin import VITS_PinYin
from text import cleaned_text_to_sequence
from text.symbols import symbols
import gradio as gr
import utils
import torch
import argparse
import os
import re
import logging
from config import FLAGS
from pathlib import Path
import shutil
import soundfile as sf
import librosa

logging.getLogger('numba').setLevel(logging.WARNING)
limitation = os.getenv("SYSTEM") == "spaces"

class WavStruct():
    def __init__(self, wav_path, start_time):
        self.wav_path = wav_path
        self.start_time = start_time

def text_to_speech(net_g, tts_front, input_text, speed):
    print("pre text_to_speech::")
    phonemes, char_embeds = tts_front.chinese_to_phonemes(input_text)
    input_ids = cleaned_text_to_sequence(phonemes)
    with torch.no_grad():
        x_tst = torch.LongTensor(input_ids).unsqueeze(0).to(device)
        x_tst_lengths = torch.LongTensor([len(input_ids)]).to(device)
        x_tst_prosody = torch.FloatTensor(char_embeds).unsqueeze(0).to(device)
        audio = net_g.infer(x_tst, x_tst_lengths, x_tst_prosody, noise_scale=0.5, length_scale=speed)[0][0, 0].data.cpu().float().numpy()
    del x_tst, x_tst_lengths, x_tst_prosody
    return audio

def pre_text_to_speech(net_g, tts_front, input_text, output_file, tts_voice_ckpt_dir, speed, desired_duration = 0, start_time = 0):
    print("pre text_to_speech::")
    ## Cannot tts text > 150 words, temporary slice to 150 if exceed
    if limitation:
        text_len = len(re.sub("\[([A-Z]{2})\]", "", input_text))
        max_len = 150
        if text_len > max_len:
            input_text = input_text[:150]
            
    ## Return wav without matching desired_duration 
    # audio = text_to_speech(net_g, tts_front, input_text, speed)
    ## OR
    ## TTS once for getting real duration then calculate speed and run again for adjusting speed
    wav_tmp = text_to_speech(net_g, tts_front, input_text, speed)
    predicted_duration = librosa.get_duration(y=wav_tmp, sr=FLAGS.sample_rate)
    speed = round(desired_duration/ predicted_duration, 2) if predicted_duration > 0 else speed
    print("adjusting speed::", predicted_duration, desired_duration, speed)
    audio = text_to_speech(net_g, tts_front, input_text, speed)
    ## Write final output to file
    sf.write(output_file, audio, samplerate=FLAGS.sample_rate)
    return WavStruct(output_file, start_time)
  
def synthesize(net_g, tts_front, input, is_file, speed, method, tts_voice_ckpt_dir, convert_voice_ckpt_dir):
    print("start synthesizing::", input, is_file, speed, method, tts_voice_ckpt_dir, convert_voice_ckpt_dir)
    filepath = ""
    paragraphs = ""
    file_name_only = ""
    basename, ext = os.path.splitext(os.path.basename(input))
    print(basename, ext)
    if is_file:
      file_name_only = Path(basename)
      filepath = utils.encode_filename(input)
      paragraphs = utils.file_to_paragraph(input)
    else:
      filepath = "{}".format(utils.new_dir_now())
      file_name_only = utils.encode_filename(filepath)
      paragraphs = utils.txt_to_paragraph(input)
    
    output_dir_name = utils.new_dir_now()
    
    output_dir_tmppath = os.path.join(FLAGS.os_tmp, output_dir_name)
    Path(output_dir_tmppath).mkdir(parents=True, exist_ok=True)
    ### Put segments in temp dir for concatnation later
    tmp_dirname = os.path.join(output_dir_tmppath, filepath)
    Path(tmp_dirname).mkdir(parents=True, exist_ok=True)
    # print("filename::", filepath, paragraphs, file_name_only, tmp_dirname)
    final_name = "{}.wav".format(file_name_only)
    final_output = os.path.join(output_dir_tmppath, final_name)
    log_output = None
    print("Output Temp: ", final_output)
    temp_output = ''

    # process_list = []
    # queue_list = Queue()
    # wav_list = []
    results = []
    for (no, para) in enumerate(paragraphs):
        name = "{}.wav".format(utils.pad_zero(no, 5))
        print("Prepare normalized text: ", para.text)
        temp_output = os.path.join(tmp_dirname, name)
        print("paragraph:: ", para.text, temp_output, para.total_duration, para.start_time)
        result = pre_text_to_speech(net_g, tts_front, para.text, temp_output, tts_voice_ckpt_dir, speed, para.total_duration, para.start_time)
        results.append(result)
        # queue_list.put((para.text, temp_output, para.total_duration, para.start_time))
        
    # print("Parallel processing {} tasks".format(len(process_list)))
    # print("Queue list:: ", queue_list.qsize())
    # JOBS = os.environ.get('JOBS', os.cpu_count()/2)
    # print("JOBS: {}".format(JOBS))
    # results = []
    # with joblib.parallel_backend(backend="multiprocessing", n_jobs=int(JOBS)):
    #   results = Parallel(verbose=100)(delayed(text_to_speech)(text, output_file, tts_voice_ckpt_dir, speed_threshold, speed, total_duration, start_silence) for (text, output_file, total_duration, start_silence) in queue_list.queue)
    # results = [text_to_speech(text, output_file, tts_voice_ckpt_dir, speed_threshold, speed) for (text, output_file) in process_list]
    # print("wav_list result::", results)
    print("TTS Done:: Start converting voice")
    if torch.cuda.is_available() and convert_voice_ckpt_dir != "none":
      utils.convert_voice(tmp_dirname, convert_voice_ckpt_dir)
      
    # TTS Done, post-processing
    if method == 'join':
      result_path, log_path = utils.combine_wav_segment(results, final_output)
      print("combine_wav_segment result::", result_path, log_path)
      final_output = result_path
      log_output = log_path
    if method == 'split':
      archive_path = re.sub(r'\.wav$', '', final_output)
      shutil.make_archive(archive_path, 'zip', tmp_dirname)
      final_output = "{}.zip".format(archive_path)
    print("final_output_path::", final_output)
    
    ## Done, remove tmp files
    # if os.path.exists(tmp_dirname): shutil.rmtree(tmp_dirname, ignore_errors=True)

    return (final_output, log_output)


def create_calback():
    def tts_calback(input_files, input_text, tts_voice, convert_voice, speed, method):    
        results_list = []
        logs_list = []
        result_text = FLAGS.empty_wav
        # pinyin
        tts_front = VITS_PinYin("./bert", device)
        # config
        hps = utils.get_hparams_from_file(os.path.join(FLAGS.config_dir, "bert_vits.json"))

        # model
        net_g = SynthesizerTrn(
            len(symbols),
            hps.data.filter_length // 2 + 1,
            hps.train.segment_size // hps.data.hop_length,
            **hps.model)

        tts_voice_ckpt_dir = os.path.join(FLAGS.tts_ckpt_dir, tts_voice)
        print("selected TTS voice:", tts_voice_ckpt_dir)
        convert_voice_ckpt_dir = os.path.join(FLAGS.convert_ckpt_dir, convert_voice) if convert_voice != "none" else "none"
        print("selected Convert voice:", convert_voice_ckpt_dir) 
        
        utils.load_model(os.path.join(tts_voice_ckpt_dir, "vits_bert_model.pth"), net_g)
        net_g.eval()
        net_g.to(device)     
        ## Process input_text first
        if input_text:
          # try:
              print('input_text::', input_text)
              output_temp_file, log_temp_file = synthesize(net_g, tts_front, input_text, False, speed, method, tts_voice_ckpt_dir, convert_voice_ckpt_dir)
              if log_temp_file:
                logs_list.append(log_temp_file)
              if method == 'join':
                result_text = output_temp_file if output_temp_file else None
              if method == 'split':
                results_list.append(output_temp_file)
          # except:
          #     print("Skip error file while synthesizing input_text")
        ## Process input_files     
        if input_files:
          print("got input files::",input_files)
          file_list = [f.name for f in input_files]
          for file_path in file_list:
              try:
                  print('file_path::',file_path)
                  output_temp_file, log_temp_file = synthesize(net_g, tts_front, file_path, True, speed, method, tts_voice_ckpt_dir, convert_voice_ckpt_dir)
                  results_list.append(output_temp_file)
                  if log_temp_file:
                    logs_list.append(log_temp_file)
              except:
                  print("Skip error file while synthesizing doc: {}".format(file_path))
        print("[DONE] {} tasks: {}".format(len(results_list), results_list))
        return results_list, result_text, logs_list
      
      

        # if limitation:
        #     text_len = len(re.sub("\[([A-Z]{2})\]", "", input_text))
        #     max_len = 150
        #     if text_len > max_len:
        #         return "Error: Text is too long", None

        # phonemes, char_embeds = tts_front.chinese_to_phonemes(input_text)
        # input_ids = cleaned_text_to_sequence(phonemes)
        # with torch.no_grad():
        #     x_tst = torch.LongTensor(input_ids).unsqueeze(0).to(device)
        #     x_tst_lengths = torch.LongTensor([len(input_ids)]).to(device)
        #     x_tst_prosody = torch.FloatTensor(char_embeds).unsqueeze(0).to(device)
        #     audio = net_g.infer(x_tst, x_tst_lengths, x_tst_prosody, noise_scale=0.5, length_scale=speed)[0][0, 0].data.cpu().float().numpy()
        # del x_tst, x_tst_lengths, x_tst_prosody
        # return "Success", (16000, audio)
      
    return tts_calback


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--share", action="store_true", default=False, help="share gradio app")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")
    
    # Create temporary directory for files processing
    os.makedirs(FLAGS.os_tmp, exist_ok=True)
    FLAGS.empty_wav.touch(exist_ok=True)
    
    # # pinyin
    # tts_front = VITS_PinYin("./bert", device)

    # # config
    # hps = utils.get_hparams_from_file(os.path.join(FLAGS.config_dir, "bert_vits.json"))

    # # model
    # net_g = SynthesizerTrn(
    #     len(symbols),
    #     hps.data.filter_length // 2 + 1,
    #     hps.train.segment_size // hps.data.hop_length,
    #     **hps.model)

    # model_path = os.path.join(FLAGS.tts_ckpt_dir, "female_a", "vits_bert_model.pth")
    # utils.load_model(model_path, net_g)
    # net_g.eval()
    # net_g.to(device)

    tts_calback = create_calback()
    
    tts_voices = [voice for voice in os.listdir(FLAGS.tts_ckpt_dir) if os.path.isdir(os.path.join(FLAGS.tts_ckpt_dir, voice))]
    convert_voices = ["none"] + [voice for voice in os.listdir(FLAGS.convert_ckpt_dir) if os.path.isdir(os.path.join(FLAGS.convert_ckpt_dir, voice))]
    app = gr.Blocks()
    with app:
        gr.Markdown("# Chinese TTS\n\n" )

        with gr.Tabs():
            with gr.TabItem("TTS"):
                with gr.Row():
                    with gr.Column():
                        input_files = gr.File(label="Upload .docx file(s)", interactive=True, file_count='directory', file_types=[".doc", ".docx", ".txt"])
                        textbox = gr.TextArea(label="Text (Maximum 150 words per paragraph)", placeholder="Type your sentence here (Maximum 150 words)", value="", elem_id=f"tts-input")
                        tts_voice = gr.Radio(label="TTS Voice", value=tts_voices[0], choices=tts_voices)
                        convert_voice = gr.Radio(label="Convert Voice", value="none", choices=convert_voices)
                        duration_slider = gr.Slider(minimum=0.5, maximum=1.5, value=1, step=0.02, label='速度 Speed')
                        method = gr.Radio(label="Method", value="join", choices=["join","split"])
                    with gr.Column():
                        files_output = gr.Files(label="Files Output")
                        audio_output = gr.Audio(label="Text Audio Output", elem_id="tts-audio")
                        logs_output = gr.Files(label="Error Audio Logs")
                        btn = gr.Button("Generate!")
                        btn.click(tts_calback,
                                  inputs=[input_files, textbox, tts_voice, convert_voice, duration_slider, method],
                                  outputs=[files_output, audio_output, logs_output])
            # gr.Examples(
            #     examples=example,
            #     inputs=[textbox, duration_slider],
            #     outputs=[text_output, audio_output],
            #     fn=tts_calback
            # )
    app.queue(concurrency_count=3).launch(
      # auth=("tts", "tts"),
      show_api=False,
      debug=False,
      server_name="0.0.0.0",
      server_port=7901,
      share=args.share)
