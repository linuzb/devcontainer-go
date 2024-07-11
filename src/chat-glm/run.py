import argparse
import torch
from transformers import pipeline, AutoTokenizer, AutoModel
import scipy
import numpy as np

# 1. 语音转文本（Speech to Text）
def speech_to_text(audio_file):
    # https://huggingface.co/openai/whisper-base
    transcriber = pipeline(model="openai/whisper-base")
    result = transcriber(audio_file)
    transcribed_text = result['text']
    return transcribed_text

# 2. 对话生成（Chat）
def generate_response(input_txt, device):
    # https://huggingface.co/THUDM/chatglm3-6b
    tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm3-6b", trust_remote_code=True)
    model = AutoModel.from_pretrained("THUDM/chatglm3-6b", trust_remote_code=True)

    model.to(device)
    model.eval()

    response, history = model.chat(tokenizer, input_txt, history=[])
    return response

# 3. 文本转语音（Text to Speech）
def text_to_speech(response_text):
    # https://huggingface.co/suno/bark-small
    synthesiser = pipeline("text-to-speech", "suno/bark-small")
    speech = synthesiser(response_text, forward_params={"do_sample": True})

    # 调试信息
    print(f"Generated speech: sampling_rate={speech['sampling_rate']}, audio_shape={speech['audio'].shape}, audio_dtype={speech['audio'].dtype}")

    return speech

# 4. 保存语音到指定目录
def save_audio(audio, save_path):
    sampling_rate = audio["sampling_rate"]
    audio_data = audio["audio"]

    # 检查采样率是否在有效范围内
    if not (0 < sampling_rate <= 192000):
        raise ValueError(f"Invalid sampling rate: {sampling_rate}")

    # 检查音频数据是否在有效范围内，并转换为一维数组
    if audio_data.ndim == 2 and audio_data.shape[0] == 1:
        audio_data = audio_data.flatten()

    # 将数据类型转换为 int16
    if audio_data.dtype != np.int16:
        # 假设音频数据在 [-1, 1] 范围内，进行缩放和转换
        audio_data = (audio_data * 32767).astype(np.int16)

    # 保存生成的音频文件
    scipy.io.wavfile.write(save_path, rate=sampling_rate, data=audio_data)
    print(f"Saved generated audio to {save_path}")


# 主函数，整合上述功能
def main(audio_file, save_path):
    # 1. 语音转文本
    input_txt = speech_to_text(audio_file)
    print("User input (speech to text):", input_txt)

    # 2. 对话生成
    
    # fake chat
    # response_text = input_txt

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    response_text = generate_response(input_txt, device)
    print("Assistant response:", response_text)

    # 3. 文本转语音
    audio = text_to_speech(response_text)

    # 4. 保存语音到指定目录
    save_audio(audio, save_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Speech to Speech Assistant")
    parser.add_argument("--audio_file", type=str, help="Path to input speech audio file")
    parser.add_argument("--save_path", type=str, help="Path to save output speech audio file")
    args = parser.parse_args()

    main(args.audio_file, args.save_path)
