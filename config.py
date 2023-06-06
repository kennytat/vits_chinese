import os
from argparse import Namespace
from pathlib import Path
import tempfile

class FLAGS(Namespace):
    """Configurations"""
    sample_rate = 16000
    # ckpt
    os_tmp = Path(os.path.join(tempfile.gettempdir(), "tts-cn"))
    empty_wav = Path(os.path.join(f'{os_tmp}', "test.wav"))
    bert_ckpt_dir = Path(os.path.join(os.getcwd(), "model", "bert"))
    tts_ckpt_dir = Path(os.path.join(os.getcwd(), "model", "tts"))
    convert_ckpt_dir = Path(os.path.join(os.getcwd(), "model", "convert"))
    config_dir = Path(os.path.join(os.getcwd(), "configs"))
    salt = Path(os.path.join(os.getcwd(), "model", "salt.salt"))
    key = "^VGMAI*607#"
    