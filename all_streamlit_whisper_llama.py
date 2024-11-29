from faster_whisper import WhisperModel
from transformers import pipeline
import fugashi
import torch
import pyaudio
import numpy as np
import wave
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import streamlit as st
import subprocess
import sys
import time
from llama_cpp import Llama
import threading
from datetime import datetime

st.set_page_config(layout="wide")
st.title("音声認識と議事録作成システム")

# Whisper code

# 文字をテキストに挿入する関数
def insert_char_to_text(i, char, text):
    l = list(text)
    l.insert(i, char)
    inserted_text = "".join(l)
    return inserted_text

# 話し言葉の特有のパターン
speech_patterns = ["えっと", "あの", "それで", "だから", "まぁ", "えー"]

# 話し言葉の特有パターンを抽出する関数
def extract_speech_patterns(text):
    tagger = fugashi.Tagger()
    words = tagger(text)
    patterns = [word.surface for word in words if word.surface in speech_patterns]
    return patterns

# 音声データ内の休止や沈黙を検出する関数
def detect_pauses(text):
    pauses = []
    words = text.split()
    for i, word in enumerate(words):
        if word == "":
            pauses.append(i)
    return pauses

model_path = "C:/Users/zemi/Downloads/Qwen2.5-7B-Instruct.Q4_K_M.gguf"
THRESHOLD = 600  # 無音判定用の閾値
SILTIME = 0.8  # 無音判定用の秒


# PyAudioによるリアルタイム音声入力と録音の設定
CHUNK = int(16000 * SILTIME)  # 音声データのチャンクサイズ
FORMAT = pyaudio.paInt16
CHANNELS = 1  # モノラルに設定
RATE = 16000  # Whisperは16kHzを使用
WAVE_OUTPUT_FILENAME = "output.wav"


class AudioWorker(threading.Thread):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.text = ""
        self.should_stop = threading.Event()

    def run(self):
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = WhisperModel("large-v3", device=str(device), compute_type="float16")

        # 音声入力ストリームを作成
        p = pyaudio.PyAudio()
        stream = p.open(format=FORMAT,
                        channels=CHANNELS,
                        rate=RATE,
                        input=True,
                        input_device_index=1,  # 適切なデバイスIDに設定
                        frames_per_buffer=CHUNK)

        print("リアルタイム音声認識と録音を開始します。")

        audio_float = []

        while not self.should_stop.wait(0):
            try:

                audio_data = np.frombuffer(stream.read(CHUNK), dtype=np.int16)
                print(np.max(audio_data))

                # バッファに音声データを追加
                if np.max(audio_data) >= THRESHOLD:
                    audio_float.append(audio_data)

                # Whisperを用いて音声をテキストに変換
                if np.max(audio_data) < THRESHOLD and len(audio_float) > 0:
                    segments,_ = model.transcribe(np.concatenate(audio_float), beam_size=10, language = "ja", vad_filter=True, without_timestamps=True, initial_prompt="次の文章の適切な位置に、「、」や「。」を付けてください。文章が区切れる箇所に、「、」を挿入してください。\n\n")

                    for segment in segments:
                        if not "ご視聴ありがとうございました" in segment.text and not "文章が区切れる箇所に" in segment.text and not "最後までご視聴いただきありがとうございます" in segment.text and not "本日はご覧いただきありがとうございます" in segment.text and not "最後までご視聴いただき、ありがとうございました。" in segment.text and not "「、」" in segment.text and not "「、 、" in segment.text and not "「、、" in segment.text:
                            print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))
                            self.text += segment.text + "\n"

                    audio_float = []

                time.sleep(0.1)
            
            except Exception as e:
                print(e)

        # ストリームを閉じる
        stream.stop_stream()
        stream.close()
        p.terminate()
    



# Llama code

class LLMWorker(threading.Thread):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model = Llama(model_path=model_path, n_gpu_layers=-1,tensor_split = [0.0, 1.0], n_ctx=8192)
        self.system_text = "あなたはとても優秀なアシスタントです。"
        self.text = ""
        self.summary = ""
        self.advice = ""
        self.mode = "summary"
        self.should_stop = threading.Event()
        
    def run(self):
        while not self.should_stop.wait(0):
            if self.text == "":
                time.sleep(0.5)
                continue

            try:
                if self.mode == "summary":
                    messages = [
                        {"role" : "system", "content" : self.system_text},
                        {"role" : "user", "content" : "以下の文章を新たな議事録にしてください。簡潔にまとめずに、できるだけ詳しくまとめてください。\n\n" + self.text}
                    ]
                    self.text = ""
                    result = self.model.create_chat_completion(messages, max_tokens=1024)["choices"][0]["message"]["content"]
                    print(result)

                    while result.startswith("\n"):
                        result = result.lstrip()
                    self.summary += result
                    self.summary += "    \n\n"

                elif self.mode == "advice":
                    messages = [
                        {"role" : "system", "content" : self.system_text},
                        {"role" : "user", "content" : "以下の文章の内容に対する助言もお願いします。\n\n" + self.text}
                    ]
                    self.text = ""
                    result = self.model.create_chat_completion(messages, max_tokens=1024)["choices"][0]["message"]["content"]
                    print(result)

                    self.advice = result
                    self.advice += "    \n\n"

            except Exception as e:
                print(e)



# Streamlit code

def main():

    # セッション状態の初期化

    if "audio_worker" not in st.session_state:
        st.session_state.audio_worker = None
        st.session_state.llm_worker = None

    # file_name をセッション状態に保存
    if "file_name" not in st.session_state:
        st.session_state.file_name = ""
        st.session_state.file_name_summary = ""
        st.session_state.file_name_comment = ""
        st.session_state.file_name_advice = ""
        st.session_state.error_message = ""
        
    if "show_message" not in st.session_state:
        st.session_state.show_message = False

    if "counter" not in st.session_state:
        st.session_state.counter = 0
    
    if "start_time" not in st.session_state:
        st.session_state.start_time = 0
    
    if "start_time_save" not in st.session_state:
        st.session_state.start_time_save = 0  
        st.session_state.start_time_save2 = 0 # 開始時間を初期化
        st.session_state.start_time_save3 = 0
    
    if "timer_running" not in st.session_state:
        st.session_state.timer_running = False  # タイマーが動いているかどうかを管理する
    
    if "elapsed_time_save" not in st.session_state:
        st.session_state.elapsed_time_save = time.time() - 10
        st.session_state.elapsed_time_save2 = time.time() - 10
        st.session_state.elapsed_time_save3 = time.time() - 10
    
    if "message" not in st.session_state:
        st.session_state.message = ""
        st.session_state.message_talk = ""  # メッセージ内容の初期化
        st.session_state.message_advice = ""


    file_name = st.text_input("保存する議事録のファイル名を入力してください（拡張子なし）:",value=st.session_state.file_name)  # セッションステートの値を初期値として使用

    # 入力値をセッションステートに同期
    if file_name and file_name != st.session_state.file_name:
        st.session_state.file_name = file_name
        st.session_state.error_message = ""

    # ファイルパスの更新
        st.session_state.file_name_summary = f"C:/Users/zemi/Desktop/whisper/download/{st.session_state.file_name}.txt"
        st.session_state.file_name_comment = f"C:/Users/zemi/Desktop/whisper/download/{st.session_state.file_name}_での発言.txt"
        st.session_state.file_name_advice = f"C:/Users/zemi/Desktop/whisper/download/{st.session_state.file_name}_ファイルのアドバイス.txt"

        if os.path.exists(st.session_state.file_name_summary) or \
            os.path.exists(st.session_state.file_name_comment) or \
            os.path.exists(st.session_state.file_name_advice):
            st.session_state.error_message = "同じ名前のファイルが既に存在します。別の名前を入力してください。"
            st.session_state.file_name = ""

    button_css = f"""
    <style>
      div.stButton > button {{
        font-weight  : bold                ;/* 文字：太字                   */
        border-radius: 10px 10px 10px 10px ;/* 枠線：半径10ピクセルの角丸     */
        background   : #333333                ;/* 背景色：濃い灰色            */
      }}
    </style>
    """
    st.markdown(button_css, unsafe_allow_html=True)

     # エラーがある場合、エラーメッセージを表示
    if st.session_state.error_message:
        st.error(st.session_state.error_message)
    else:
        # worker を制御（起動/停止）する部分
        with st.sidebar:
            if st.session_state.file_name:
                if st.button("音声認識を開始する", disabled=st.session_state.audio_worker is not None):
                    st.write(f"音声認識を開始します。結果はファイル: {st.session_state.file_name_summary} に保存されます。")
                    st.write("起動中です...")
                    st.session_state.start_time = time.time()
                    st.session_state.counter = 0
                    st.session_state.audio_worker = AudioWorker(daemon=True)
                    st.session_state.audio_worker.start()
                    st.session_state.llm_worker = LLMWorker(daemon=True)
                    st.session_state.llm_worker.start()
                    st.session_state.timer_running = True  # タイマーを開始

                    print("初期化されました")
                    st.rerun()

                if st.button("音声認識を終了する", disabled=st.session_state.audio_worker is None):
                    try:
                        st.write("リアルタイム音声認識と録音を終了します。")

                        st.session_state.timer_running = False  # タイマーを停止

                        # AudioWorkerの停止
                        if st.session_state.audio_worker:
                            st.session_state.audio_worker.should_stop.set()
                            st.write("音声認識ワーカーを停止中...")
                            st.session_state.audio_worker.join(timeout=3)  # 最大3秒待機                       
                            st.write("音声認識ワーカーが終了しました。")
                            st.session_state.audio_worker = None

                        # LLMWorkerの停止
                        if st.session_state.llm_worker:
                            st.session_state.llm_worker.should_stop.set()
                            st.write("LLMワーカーを停止中...")
                            st.session_state.llm_worker.join(timeout=3)  # 最大3秒待機                           
                            st.write("LLMワーカーが終了しました。")
                            st.session_state.llm_worker = None

                        # セッションステートを初期化（再度ファイル名を入力させる）
                        st.session_state.file_name = ""
                        st.session_state.file_name_summary = ""
                        st.session_state.file_name_comment = ""
                        st.session_state.file_name_advice = ""
                        st.session_state.error_message = ""  # エラーメッセージも初期化

                        # 最初の画面に戻すためにrerunを呼び出し
                        st.write('')
                        st.write('')
                        st.write('')
                        st.write('')
                        time.sleep(1.0)
                        st.rerun()

                    except Exception as e:
                        st.error(f"終了処理中にエラーが発生しました: {e}")

                if st.session_state.timer_running:
                    # 経過時間を計算
                    elapsed_time = time.time() - st.session_state.start_time
                    st.session_state.counter = elapsed_time  # 秒単位でカウントアップ

                    # 時間を分:秒形式（00:00）で表示
                    minutes = int(st.session_state.counter // 60)
                    seconds = int(st.session_state.counter % 60)
                    formatted_time = f"{minutes:02}:{seconds:02}"

                    # 会議時間と現在時刻の表示
                    st.write(f"会議時間: {formatted_time} \n")
                    current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')  # 現在時刻を取得
                    st.write(f"現在の時刻: {current_time}")  # プレースホルダーに時刻を表示
                     
    # worker の状態を表示する部分
    message_placeholder = st.empty()

    if st.session_state.audio_worker is None:
        st.markdown("ファイル名を入力し、Enterキーを押すと左に音声認識の開始ボタンと終了ボタンが表示されます")
        
        # 現在時刻の表示
        current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')  # 現在時刻を取得
        message_placeholder.write(f"現在の時刻: {current_time}")  # プレースホルダーに時刻を表示
        st.write('')
        time.sleep(1.0)
        st.rerun()      
    else:
        message_placeholder.empty()
        col1, col2 = st.columns(2)

    # 要約とアドバイス生成
        with col1:
            summary = st.text_area("議事録", key="area1", height=34*10, value=st.session_state.llm_worker.summary)
            
            if st.button("議事録にする", key="btn1"):
                st.session_state.llm_worker.summary = ""
                st.session_state.llm_worker.mode = "summary"
                st.session_state.llm_worker.text = "" + st.session_state.audio_worker.text
                st.rerun()

            if st.button("ファイルに保存", key="btn2"):
                st.session_state.show_message = True
                st.session_state.start_time_save = time.time()  # 現在時刻を記録

                with open(st.session_state.file_name_summary , "w", encoding="utf-8") as f:
                    f.write("今回の議事録\n\n" + summary + "\n")

                    # メッセージの表示
                    st.session_state.message = "議事録の要約が保存されました。保存場所 : " + st.session_state.file_name_summary

            # 経過時間を計算
            st.session_state.elapsed_time_save = time.time() - st.session_state.start_time_save
            if st.session_state.elapsed_time_save > 5:
                st.session_state.message = ''
            st.write(st.session_state.message)

            advice = st.text_area("会議のアドバイス", key="area3", height=34*10, value=st.session_state.llm_worker.advice)

            if st.button("アドバイス", key="btn3"):
                st.session_state.llm_worker.advice = ""
                st.session_state.llm_worker.mode = "advice"
                st.session_state.llm_worker.text = "" + st.session_state.llm_worker.summary
                st.rerun()

            if st.button("ファイルに保存", key="btn4"):
                st.session_state.show_message = True
                st.session_state.start_time_save2 = time.time()  # 現在時刻を記録

                with open(st.session_state.file_name_advice, "w", encoding="utf-8") as f:
                    f.write("今回の会議のアドバイス\n\n" + advice + "\n")

                    # メッセージの表示             
                    st.session_state.message_advice = "今回の会議のアドバイスが保存されました。保存場所 : " + st.session_state.file_name_advice

            # 経過時間を計算
            st.session_state.elapsed_time_save2 = time.time() - st.session_state.start_time_save2
            if st.session_state.elapsed_time_save2 > 5:
                st.session_state.message_advice = ''
            st.write(st.session_state.message_advice)
            
        with col2:
            record = st.text_area("今までの発言", key="area2", height=34*10, value=st.session_state.audio_worker.text)

            if st.button("ファイルに保存", key="btn5"):
                st.session_state.show_message = True
                st.session_state.start_time_save3 = time.time()  # 現在の時間を記録

                # ファイルへの保存処理
                with open(st.session_state.file_name_comment, "w", encoding="utf-8") as f:
                    f.write("今回話したこと\n\n" + record + "\n")

                    # メッセージの表示
                    st.session_state.message_talk = "今までの発言が保存されました。保存場所 : " + st.session_state.file_name_comment

            # 経過時間を計算
            st.session_state.elapsed_time_save3 = time.time() - st.session_state.start_time_save3
            if st.session_state.elapsed_time_save3 > 5:
                st.session_state.message_talk = ''
            st.write(st.session_state.message_talk)

        if st.session_state.audio_worker.is_alive():
            time.sleep(1.0)
            st.write('')
            st.rerun()

if __name__ == "__main__":
    main()
