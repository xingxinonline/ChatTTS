import os, sys
import torch
import lzma
import random
import numpy as np
import pybase16384 as b14
import sounddevice as sd

# 指定文件夹路径
folder_path = "g:/workspace/kws/chattts_pt/"
# 获取文件夹中所有 .pt 文件的列表
pt_files = [f for f in os.listdir(folder_path) if f.endswith('.pt')]

# 如果文件列表不为空，则随机选择一个文件
if pt_files:
    selected_file = random.choice(pt_files)
    selected_file_path = os.path.join(folder_path, selected_file)
    
    # 加载随机选择的文件
    # spk = torch.load(selected_file_path, map_location=torch.device('cpu')).detach()
    print(f"Loaded file: {selected_file_path}")
else:
    print("No .pt files found in the specified directory.")

# pt_files = [f for f in os.listdir(folder_path) if f.endswith('.pt')]
# selected_file = random.choice(pt_files)
# print(f"Selected file: {selected_file}")



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

import re

# 清理文件名，去掉特殊标记和空格
def clean_filename(filename):
    # 使用正则表达式去除方括号及其内容
    cleaned_name = re.sub(r'\[.*?\]', '', filename)
    # 删除空格
    cleaned_name = cleaned_name.replace(' ', '')
    return cleaned_name


logger = get_logger("Command")


def save_mp3_file(wav, index):
    data = pcm_arr_to_mp3_view(wav)
    mp3_filename = f"output_audio_{index}.mp3"
    with open(mp3_filename, "wb") as f:
        f.write(data)
    logger.info(f"Audio saved to {mp3_filename}")

import scipy

def main(texts: List[str], spk: Optional[str] = None, stream=False):
    logger.info("Text input: %s", str(texts))

    chat = ChatTTS.Chat(get_logger("ChatTTS"))
    logger.info("Initializing ChatTTS...")
    if chat.load():
        logger.info("Models loaded successfully.")
    else:
        logger.error("Models load failed.")
        sys.exit(1)
    # spk = torch.load(selected_file_path,
                    #  map_location=torch.device('cpu')).detach()
    # 如果文件列表不为空，则遍历所有文件
    if pt_files:
        for selected_file in pt_files:
            # 生成 0 到 7 之间的随机整数
            random_break = random.randint(0, 7)
            print(f"Random break: {random_break}")
            # 生成 0 到 9 之间的随机整数
            random_speed = random.randint(0, 8)
            print(f"Random speed: {random_speed}")
            selected_file_path = os.path.join(folder_path, selected_file)
            
            # 加载每个文件并进行操作
            spk = torch.load(selected_file_path, map_location=torch.device('cpu')).detach()
            print(f"Loaded file: {selected_file_path}")
            # 在这里对 `spk` 进行你需要的操作
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
                    prompt=f'[oral_0][laugh_0][break_{random_break}]',
                )
            )
            logger.info("Text output: %s", str(texts)) 
            # 创建与 .pt 文件同名的文件夹
            output_dir = os.path.join(now_dir, os.path.splitext(selected_file)[0])
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            wavs = chat.infer(
                texts,
                stream,
                skip_refine_text=True,
                # refine_text_only=True,
                params_infer_code=ChatTTS.Chat.InferCodeParams(
                    prompt=f'[speed_{random_speed}]',
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
                    # 假设 `texts[index]` 是原始文件名的一部分
                    original_filename = f"{texts[index]}.wav"
                    cleaned_filename = clean_filename(original_filename)
                    # 构建保存路径
                    output_wav_path = os.path.join(output_dir, cleaned_filename)

                    # 播放音频
                    sd.play(wav, samplerate=24000)
                    sd.wait()  # 等待播放结束

                    # 将音频保存为 WAV 文件
                    scipy.io.wavfile.write(filename=output_wav_path, rate=24000, data=wav.T)
            if stream:
                for index, wav in enumerate(np.concatenate(wavs_list, axis=1)):
                    save_mp3_file(wav, index)
            logger.info("Audio generation successful.")
    else:
        print("No .pt files found in the specified directory.")
    


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
