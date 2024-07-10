import argparse
import torch
from transformers import pipeline, AutoTokenizer, AutoModel
import scipy

# 1. 语音转文本（Speech to Text）
def speech_to_text(audio_file):
    transcriber = pipeline(model="openai/whisper-base")
    result = transcriber(audio_file)
    transcribed_text = result['text']
    return transcribed_text

# 2. 对话生成（Chat）
def generate_response(input_txt, device):
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

    return speech["audio"], speech["sampling_rate"]

# 4. 保存语音到指定目录
def save_audio(audio, sampling_rate, save_path):
    # 保存生成的音频文件
    scipy.io.wavfile.write(save_path, rate=audio["sampling_rate"], data=audio["audio"])
    print(f"Saved generated audio to {save_path}")

# 主函数，整合上述功能
def main(audio_file, save_path):
    # 1. 语音转文本
    input_txt = speech_to_text(audio_file)
    print("User input (speech to text):", input_txt)

    # 2. 对话生成
    response_text = input_txt

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # response_text = generate_response(input_txt, device)
    # print("Assistant response:", response_text)

    # 3. 文本转语音
    audio, sampling_rate = text_to_speech(response_text)

    # 4. 保存语音到指定目录
    save_audio(audio, sampling_rate, save_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Speech to Speech Assistant")
    parser.add_argument("--audio_file", type=str, help="Path to input speech audio file")
    parser.add_argument("--save_path", type=str, help="Path to save output speech audio file")
    args = parser.parse_args()

    main(args.audio_file, args.save_path)
