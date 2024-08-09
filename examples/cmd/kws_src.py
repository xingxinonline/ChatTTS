import os, sys
import torch
import lzma
import numpy as np
import pybase16384 as b14


def compress_and_encode(tensor):
    np_array = tensor.numpy().astype(np.float16)
    compressed = lzma.compress(np_array.tobytes(), format=lzma.FORMAT_RAW,
                               filters=[{"id": lzma.FILTER_LZMA2, "preset": 9 | lzma.PRESET_EXTREME}])
    encoded = b14.encode_to_string(compressed)
    return encoded

if sys.platform == "darwin":
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

now_dir = os.getcwd()
sys.path.append(now_dir)

import argparse
from typing import Optional, List

import numpy as np

import ChatTTS

from tools.audio import pcm_arr_to_mp3_view
from tools.logger import get_logger

logger = get_logger("Command")


def save_mp3_file(wav, index):
    data = pcm_arr_to_mp3_view(wav)
    mp3_filename = f"output_audio_{index}.mp3"
    with open(mp3_filename, "wb") as f:
        f.write(data)
    logger.info(f"Audio saved to {mp3_filename}")


def main(texts: List[str], spk: Optional[str] = None, stream=False):
    logger.info("Text input: %s", str(texts))

    chat = ChatTTS.Chat(get_logger("ChatTTS"))
    logger.info("Initializing ChatTTS...")
    if chat.load():
        logger.info("Models loaded successfully.")
    else:
        logger.error("Models load failed.")
        sys.exit(1)
    spk = torch.load("g:/workspace/kws/chattts_pt/seed_2160_restored_emb.pt",
                     map_location=torch.device('cpu')).detach()
    spk_emb_str = compress_and_encode(spk)
    if spk is None:
        spk = chat.sample_random_speaker()
    logger.info("Use speaker:")
    print(spk)

    logger.info("Start inference.")
    texts = chat.infer(
        texts,
        skip_refine_text=False,
        refine_text_only=True,
        params_refine_text=ChatTTS.Chat.RefineTextParams(
            prompt='[oral_0][laugh_0][break_5]',
        )
    )
    logger.info("Text output: %s", str(texts)) 
    wavs = chat.infer(
        texts,
        stream,
        skip_refine_text=True,
        # refine_text_only=True,
        params_infer_code=ChatTTS.Chat.InferCodeParams(
            prompt='[speed_9]',
            temperature=0.0003,  # using custom temperature
            top_P=0.7,  # top P decode
            top_K=20,  # top K decode
            spk_emb=spk_emb_str,
        ),
    )
    logger.info("Inference completed.")
    # Save each generated wav file to a local file
    if stream:
        wavs_list = []
    for index, wav in enumerate(wavs):
        if stream:
            for i, w in enumerate(wav):
                save_mp3_file(w, (i + 1) * 1000 + index)
            wavs_list.append(wav)
        else:
            save_mp3_file(wav, index)
    if stream:
        for index, wav in enumerate(np.concatenate(wavs_list, axis=1)):
            save_mp3_file(wav, index)
    logger.info("Audio generation successful.")


if __name__ == "__main__":
    logger.info("Starting ChatTTS commandline demo...")
    parser = argparse.ArgumentParser(
        description="ChatTTS Command",
        usage='[--spk xxx] [--stream] "Your text 1." " Your text 2."',
    )
    parser.add_argument(
        "--spk",
        help="Speaker (empty to sample a random one)",
        type=Optional[str],
        default=None,
    )
    parser.add_argument(
        "--stream",
        help="Use stream mode",
        action="store_true",
    )
    parser.add_argument(
        "texts",
        help="Original text",
        default=["YOUR TEXT HERE"],
        nargs=argparse.REMAINDER,
    )
    args = parser.parse_args()
    main(args.texts, args.spk, args.stream)
    logger.info("ChatTTS process finished.")
