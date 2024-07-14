from io import BytesIO
import os

import sys
import traceback
from infer.lib import jit
from infer.lib.jit.get_synthesizer import get_synthesizer
from time import time as ttime
import fairseq
import faiss
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio

now_dir = os.getcwd()
sys.path.append(now_dir)
from multiprocessing import Manager as M

from configs.config import Config

mm = M()


def printt(strr, *args):
    if len(args) == 0:
        print(strr)
    else:
        print(strr % args)

class RVC:
    def __init__(
        self,
        key,
        pth_path,
        index_path,
        index_rate,
        n_cpu,
        inp_q,
        opt_q,
        config: Config,
        last_rvc=None,
    ) -> None:
        """
        初始化
        """
        try:
            # global config
            self.config = config
            self.inp_q = inp_q
            self.opt_q = opt_q
            # device="cpu"########强制cpu测试
            self.device = config.device
            self.f0_up_key = key
            self.f0_min = 50
            self.f0_max = 1100
            self.f0_mel_min = 1127 * np.log(1 + self.f0_min / 700)
            self.f0_mel_max = 1127 * np.log(1 + self.f0_max / 700)
            self.n_cpu = n_cpu
            self.use_jit = self.config.use_jit
            self.is_half = config.is_half

            self.pth_path: str = pth_path
            self.index_path = index_path
            self.index_rate = index_rate
            self.cache_pitch: np.ndarray = np.zeros(1024, dtype="int32")
            self.cache_pitchf = np.zeros(1024, dtype="float32")

            # knn-vc related stuff
            self.knn_vc = torch.hub.load('bshall/knn-vc', 'knn_vc', prematched=True, trust_repo=True, pretrained=True)
            self.matching_set = torch.load(pth_path).float().to('cuda')
            self.tgt_sr = 16000

            self.net_g: nn.Module = None

        except:
            printt(traceback.format_exc())

    def change_key(self, new_key):
        self.f0_up_key = new_key

    def change_index_rate(self, new_index_rate):
        self.index_rate = new_index_rate

    def get_f0_post(self, f0):
        return 0, 0

    def get_f0(self, x, f0_up_key, n_cpu, method="harvest"):
        return 0 

    def get_f0_crepe(self, x, f0_up_key):
        return 0

    def get_f0_rmvpe(self, x, f0_up_key):
        return 0

    def get_f0_fcpe(self, x, f0_up_key):
        return 0

    def infer(
        self,
        input_wav: torch.Tensor,
        block_frame_16k, # only needed for pitch detection
        skip_head,
        return_length,
        f0method,
    ) -> np.ndarray:
        t1 = ttime()
        ##### Feature Extraction

        feats = self.knn_vc.get_features(input_wav)
        t2 = ttime()
        ##### no more index search :)
        t3 = ttime()
        ##### no more f0 extraction :)
        t4 = ttime()
        ##### model
        infered_audio = self.knn_vc.match(feats, self.matching_set, topk=4, tgt_loudness_db=None).to('cuda')

        t5 = ttime()
        printt(
            "Spent time: fea = %.3fs, index = %.3fs, f0 = %.3fs, model = %.3fs",
            t2 - t1,
            t3 - t2,
            t4 - t3,
            t5 - t4,
        )
        result = infered_audio.squeeze().float()
        print(f'in: {input_wav.shape}, out: {result.shape}')
        return infered_audio.squeeze().float()
