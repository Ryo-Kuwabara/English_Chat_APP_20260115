import streamlit as st
import os
import time
from pathlib import Path
import wave
import pyaudio
from pydub import AudioSegment
from audio_recorder_streamlit import audio_recorder
import numpy as np
from scipy.io.wavfile import write
from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
)
from langchain.schema import SystemMessage
from langchain.memory import ConversationSummaryBufferMemory
from langchain_openai import ChatOpenAI
from langchain.chains import ConversationChain
import constants as ct

def record_audio_simple(key_suffix=""):
    """
    シンプルな音声録音機能
    Args:
        key_suffix: キーの接尾辞（重複を避けるため）
    Returns:
        audio_data: 録音された音声データ（BytesIOオブジェクト）、またはNone
    """
    
    # シンプルなキー管理（重複回避）
    recorder_key = f"main_recorder_{key_suffix}" if key_suffix else "main_recorder"
    
    # グローバルマイクアクセス許可の管理
    if "global_microphone_permission" not in st.session_state:
        st.session_state["global_microphone_permission"] = False
    
    # 処理中かどうかに応じてヒントテキストを変更
    if st.session_state.get("current_step", "waiting") == "processing":
        button_text = "⏳ 処理中..."
        is_disabled = True
    else:
        if not st.session_state["global_microphone_permission"]:
            button_text = "🎤 マイクアクセス許可 (初回のみ)"
        else:
            button_text = "🎤 録音開始 / 🛑 録音停止"
        is_disabled = False
    
    # 録音コンポーネントを表示（常に同じキーを使用）
    if not is_disabled:
        audio_data = audio_recorder(
            text=button_text,
            recording_color="#e8b62c",
            neutral_color="#6aa36f", 
            icon_name="microphone-lines",
            icon_size="2x",
            key=recorder_key,  # シンプルなキー管理
            energy_threshold=(-1.0, 1.0),
            pause_threshold=300.0,  # 5分間（実質的に自動停止を無効化）
            sample_rate=41_000
        )
    else:
        # 処理中は無効化されたコンポーネントを表示
        st.info("⏳ 音声処理中です... 完了までお待ちください")
        audio_data = None
    
    # マイクアクセス許可の状態管理
    if audio_data is not None and not st.session_state["global_microphone_permission"]:
        st.session_state["global_microphone_permission"] = True
        st.success("✅ マイクアクセスが許可されました！2回目以降は許可不要です。")
    
    return audio_data

def save_audio_to_file(audio_data, file_path):
    """
    音声データをファイルに保存
    Args:
        audio_data: audio_recorderから取得した音声データ
        file_path: 保存先ファイルパス
    Returns:
        bool: 保存成功の場合True、失敗の場合False
    """
    try:
        if audio_data is not None and len(audio_data) > 0:
            # audio_recorderはbytesオブジェクトを返すので、BytesIOに変換
            from io import BytesIO
            from pydub import AudioSegment
            
            # bytesデータをBytesIOオブジェクトに変換
            audio_bytes = BytesIO(audio_data)
            
            # AudioSegmentで読み込み
            audio_segment = AudioSegment.from_file(audio_bytes)
            
            # 音声の長さをチェック（0.1秒未満の場合は拒否）
            duration_ms = len(audio_segment)
            duration_seconds = duration_ms / 1000.0
            
            if duration_seconds < 0.1:
                st.error(f"録音時間が短すぎます（{duration_seconds:.2f}秒）。最低0.1秒以上録音してください。")
                return False
            
            # WAVファイルとして保存
            audio_segment.export(file_path, format="wav")
            
            return True
        else:
            st.error("録音データが空です。もう一度録音してください。")
            return False
    except Exception as e:
        st.error(f"音声ファイル保存エラー: {e}")
        return False

def transcribe_audio(audio_input_file_path):
    """
    音声入力ファイルから文字起こしテキストを取得
    Args:
        audio_input_file_path: 音声入力ファイルのパス
    """
    try:
        with open(audio_input_file_path, 'rb') as audio_input_file:
            transcript = st.session_state.openai_obj.audio.transcriptions.create(
                model="whisper-1",
                file=audio_input_file,
                language="en"
            )
        
        return transcript
    except Exception as e:
        st.error(f"音声認識エラー: {e}")
        raise e
    finally:
        # ファイルが存在する場合のみ削除
        if os.path.exists(audio_input_file_path):
            os.remove(audio_input_file_path)

def save_to_wav(llm_response_audio, audio_output_file_path):
    """
    一旦mp3形式で音声ファイル作成後、wav形式に変換
    Args:
        llm_response_audio: LLMからの回答の音声データ
        audio_output_file_path: 出力先のファイルパス
    """

    temp_audio_output_filename = f"{ct.AUDIO_OUTPUT_DIR}/temp_audio_output_{int(time.time())}.mp3"
    with open(temp_audio_output_filename, "wb") as temp_audio_output_file:
        temp_audio_output_file.write(llm_response_audio)
    
    audio_mp3 = AudioSegment.from_file(temp_audio_output_filename, format="mp3")
    audio_mp3.export(audio_output_file_path, format="wav")

    # 音声出力用に一時的に作ったmp3ファイルを削除
    os.remove(temp_audio_output_filename)

def play_wav(audio_output_file_path, speed=1.0):
    """
    音声ファイルの読み上げ
    Args:
        audio_output_file_path: 音声ファイルのパス
        speed: 再生速度（1.0が通常速度、0.5で半分の速さ、2.0で倍速など）
    """

    # 音声ファイルの読み込み
    audio = AudioSegment.from_wav(audio_output_file_path)
    
    # 速度を変更
    if speed != 1.0:
        # frame_rateを変更することで速度を調整
        modified_audio = audio._spawn(
            audio.raw_data, 
            overrides={"frame_rate": int(audio.frame_rate * speed)}
        )
        # 元のframe_rateに戻すことで正常再生させる（ピッチを保持したまま速度だけ変更）
        modified_audio = modified_audio.set_frame_rate(audio.frame_rate)

        modified_audio.export(audio_output_file_path, format="wav")

    # PyAudioで再生
    with wave.open(audio_output_file_path, 'rb') as play_target_file:
        p = pyaudio.PyAudio()
        stream = p.open(
            format=p.get_format_from_width(play_target_file.getsampwidth()),
            channels=play_target_file.getnchannels(),
            rate=play_target_file.getframerate(),
            output=True
        )

        data = play_target_file.readframes(1024)
        while data:
            stream.write(data)
            data = play_target_file.readframes(1024)

        stream.stop_stream()
        stream.close()
        p.terminate()
    
    # LLMからの回答の音声ファイルを削除
    os.remove(audio_output_file_path)

def create_chain(system_template):
    """
    LLMによる回答生成用のChain作成
    """

    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content=system_template),
        MessagesPlaceholder(variable_name="history"),
        HumanMessagePromptTemplate.from_template("{input}")
    ])
    chain = ConversationChain(
        llm=st.session_state.llm,
        memory=st.session_state.memory,
        prompt=prompt
    )

    return chain

def create_problem_and_play_audio():
    """
    問題生成と音声ファイルの再生
    Args:
        chain: 問題文生成用のChain
        speed: 再生速度（1.0が通常速度、0.5で半分の速さ、2.0で倍速など）
        openai_obj: OpenAIのオブジェクト
    """

    # 問題文を生成するChainを実行し、問題文を取得
    problem = st.session_state.chain_create_problem.predict(input="")

    # LLMからの回答を音声データに変換
    llm_response_audio = st.session_state.openai_obj.audio.speech.create(
        model="tts-1",
        voice="alloy",
        input=problem
    )

    # 音声ファイルの作成
    audio_output_file_path = f"{ct.AUDIO_OUTPUT_DIR}/audio_output_{int(time.time())}.wav"
    save_to_wav(llm_response_audio.content, audio_output_file_path)

    # 音声ファイルの読み上げ
    play_wav(audio_output_file_path, st.session_state.speed)

    return problem, llm_response_audio

def create_evaluation():
    """
    ユーザー入力値の評価生成
    """

    llm_response_evaluation = st.session_state.chain_evaluation.predict(input="")

    return llm_response_evaluation

def play_audio_web_compatible(audio_file_path, speed=1.0):
    """
    Webアプリ対応の音声再生（ブラウザ側再生）
    - localhost/クラウド環境の両方で動作
    - ブラウザの音声コントロールを使用
    """
    try:
        print(f"[WEB] ブラウザ音声再生開始: {audio_file_path}")
        
        # ファイル存在確認
        if not os.path.exists(audio_file_path):
            print(f"[ERROR] 音声ファイルが見つかりません: {audio_file_path}")
            return False
        
        # 速度調整が必要な場合は事前に処理
        playback_file = audio_file_path
        if speed != 1.0:
            from pydub import AudioSegment
            import time
            
            print(f"[WEB] 速度調整処理: {speed}x")
            audio = AudioSegment.from_wav(audio_file_path)
            modified_audio = audio._spawn(
                audio.raw_data, 
                overrides={"frame_rate": int(audio.frame_rate * speed)}
            )
            modified_audio = modified_audio.set_frame_rate(audio.frame_rate)
            
            # 一時ファイル作成
            temp_path = audio_file_path.replace('.wav', f'_web_temp_{int(time.time())}.wav')
            modified_audio.export(temp_path, format="wav")
            playback_file = temp_path
        
        # Streamlitのst.audioでブラウザ再生
        import streamlit as st
        
        # 音声コントロール付きで表示
        st.audio(playback_file, format="audio/wav")
        
        # 一時ファイルがあれば遅延削除
        if speed != 1.0 and playback_file != audio_file_path:
            import threading
            def delayed_cleanup():
                time.sleep(10)  # 10秒後に削除
                if os.path.exists(playback_file):
                    try:
                        os.remove(playback_file)
                        print(f"[WEB] 一時ファイル削除完了: {playback_file}")
                    except:
                        pass
            
            threading.Thread(target=delayed_cleanup, daemon=True).start()
        
        print(f"[WEB] ブラウザ音声再生設定完了")
        return True
        
    except Exception as e:
        print(f"[ERROR] Web音声再生エラー: {e}")
        if 'st' in globals():
            st.error(f"音声再生エラー: {e}")
        return False

def play_audio_direct(audio_file_path, speed=1.0):
    """
    音声ファイルを直接再生（同期的、確実な再生）- macOS対応強化版
    Args:
        audio_file_path: 音声ファイルのパス
        speed: 再生速度
    """
    try:
        import wave
        import pyaudio
        from pydub import AudioSegment
        import subprocess
        import platform
        
        print(f"[DEBUG] 音声再生開始: {audio_file_path}")
        
        # ファイル存在確認
        if not os.path.exists(audio_file_path):
            raise FileNotFoundError(f"音声ファイルが見つかりません: {audio_file_path}")
        
        # 音声ファイルの読み込み
        audio = AudioSegment.from_wav(audio_file_path)
        print(f"[DEBUG] 音声ファイル読み込み完了: 長さ={len(audio)}ms")
        
        # 速度調整
        playback_file = audio_file_path
        temp_path = None
        if speed != 1.0:
            modified_audio = audio._spawn(
                audio.raw_data, 
                overrides={"frame_rate": int(audio.frame_rate * speed)}
            )
            modified_audio = modified_audio.set_frame_rate(audio.frame_rate)
            temp_path = audio_file_path.replace('.wav', f'_temp_speed_{int(time.time())}.wav')
            modified_audio.export(temp_path, format="wav")
            playback_file = temp_path
            print(f"[DEBUG] 速度調整完了: {speed}x")

        # PyAudioで再生を試行
        try:
            print(f"[DEBUG] PyAudio再生開始")
            with wave.open(playback_file, 'rb') as wf:
                p = pyaudio.PyAudio()
                
                # 利用可能な出力デバイスを確認
                device_count = p.get_device_count()
                print(f"[DEBUG] 利用可能デバイス数: {device_count}")
                
                # デフォルト出力デバイスを取得
                try:
                    default_device_info = p.get_default_output_device_info()
                    print(f"[DEBUG] デフォルトデバイス: {default_device_info['name']}")
                except Exception as device_err:
                    print(f"[DEBUG] デフォルトデバイス取得エラー: {device_err}")
                    # 最初の利用可能な出力デバイスを探す
                    output_device_index = None
                    for i in range(device_count):
                        try:
                            device_info = p.get_device_info_by_index(i)
                            if device_info['maxOutputChannels'] > 0:
                                output_device_index = i
                                print(f"[DEBUG] 使用デバイス: {device_info['name']} (index: {i})")
                                break
                        except:
                            continue
                    
                    if output_device_index is None:
                        raise Exception("利用可能な出力デバイスが見つかりません")
                
                # オーディオストリーム作成
                stream_kwargs = {
                    'format': p.get_format_from_width(wf.getsampwidth()),
                    'channels': wf.getnchannels(),
                    'rate': wf.getframerate(),
                    'output': True,
                    'frames_per_buffer': 1024
                }
                
                # デバイスが指定されている場合は追加
                if 'output_device_index' in locals():
                    stream_kwargs['output_device_index'] = output_device_index
                
                stream = p.open(**stream_kwargs)
                
                # 音声データを読み込んで再生
                chunk_size = 1024
                data = wf.readframes(chunk_size)
                
                while data:
                    stream.write(data)
                    data = wf.readframes(chunk_size)
                
                # リソース解放
                stream.stop_stream()
                stream.close()
                p.terminate()
                
            print(f"[DEBUG] PyAudio再生完了")
            success = True
            
        except Exception as pyaudio_error:
            print(f"[WARNING] PyAudio再生失敗: {pyaudio_error}")
            print(f"[DEBUG] PyAudioエラー詳細: {type(pyaudio_error).__name__}: {str(pyaudio_error)}")
            success = False
            
            # macOSの場合、afplayコマンドで代替再生を試行
            if platform.system() == "Darwin":  # macOS
                try:
                    print(f"[DEBUG] afplayで代替再生を試行")
                    result = subprocess.run(['afplay', playback_file], 
                                          capture_output=True, text=True, timeout=30)
                    if result.returncode == 0:
                        print(f"[DEBUG] afplay再生成功")
                        success = True
                    else:
                        print(f"[ERROR] afplay再生失敗: {result.stderr}")
                        print(f"[ERROR] afplay戻り値: {result.returncode}")
                except Exception as afplay_error:
                    print(f"[ERROR] afplay実行エラー: {afplay_error}")
            
            # LinuxやWindowsの場合の代替手段も追加可能
            if not success:
                print(f"[ERROR] 全ての音声再生方法が失敗しました")
                if 'st' in globals():
                    st.error(f"音声再生エラー: {pyaudio_error}")
                    st.error("代替再生方法も失敗しました。ブラウザでの再生をお試しください。")
        
        # テンポラリファイルがあれば削除
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)
            
        return success
            
    except Exception as e:
        print(f"[ERROR] Direct audio play error: {e}")
        # Streamlitエラー表示
        if 'st' in globals():
            st.error(f"音声再生エラー: {e}")
        return False

def encode_audio_to_base64(audio_file_path):
    """
    音声ファイルをBase64エンコードして返す
    Args:
        audio_file_path: 音声ファイルのパス
    Returns:
        str: Base64エンコードされた音声データ
    """
    import base64
    try:
        with open(audio_file_path, 'rb') as audio_file:
            audio_data = audio_file.read()
            base64_audio = base64.b64encode(audio_data).decode('utf-8')
            return base64_audio
    except Exception as e:
        st.error(f"音声エンコードエラー: {e}")
        return ""

def save_for_replay(audio_output_file_path):
    """
    再読み上げ用にファイルを保存
    Args:
        audio_output_file_path: 元の音声ファイルのパス
    """
    try:
        # 再読み上げ用にファイルを保存
        saved_audio_path = audio_output_file_path.replace('.wav', '_saved.wav')
        audio = AudioSegment.from_wav(audio_output_file_path)
        audio.export(saved_audio_path, format="wav")
        
        # 元のファイルを削除
        if os.path.exists(audio_output_file_path):
            os.remove(audio_output_file_path)
            
    except Exception as e:
        st.error(f"音声ファイル保存エラー: {e}")

def play_and_save_wav(audio_output_file_path, speed=1.0):
    """
    音声ファイルの読み上げと再読み上げ用に保存
    Args:
        audio_output_file_path: 音声ファイルのパス
        speed: 再生速度（1.0が通常速度、0.5で半分の速さ、2.0で倍速など）
    """

    # 音声ファイルの読み込み
    audio = AudioSegment.from_wav(audio_output_file_path)
    
    # 速度を変更
    if speed != 1.0:
        # frame_rateを変更することで速度を調整
        modified_audio = audio._spawn(
            audio.raw_data, 
            overrides={"frame_rate": int(audio.frame_rate * speed)}
        )
        # 元のframe_rateに戻すことで正常再生させる（ピッチを保持したまま速度だけ変更）
        modified_audio = modified_audio.set_frame_rate(audio.frame_rate)

        modified_audio.export(audio_output_file_path, format="wav")

    # 再読み上げ用にファイルを保存（元のファイルをコピー）
    saved_audio_path = audio_output_file_path.replace('.wav', '_saved.wav')
    audio_for_save = AudioSegment.from_wav(audio_output_file_path)
    audio_for_save.export(saved_audio_path, format="wav")

    # PyAudioによる再生をStreamlitの音声再生に変更
    try:
        # Streamlitの音声再生機能を使用（非ブロッキング）
        st.audio(audio_output_file_path, format="audio/wav", autoplay=True)
    except Exception as e:
        st.error(f"音声再生エラー: {e}")
    
    # 元の音声ファイルを削除（保存用は残す）
    if os.path.exists(audio_output_file_path):
        try:
            os.remove(audio_output_file_path)
        except:
            pass  # ファイルが使用中の場合はスキップ

def play_saved_audio(saved_audio_path, speed=1.0):
    """
    保存された音声ファイルを再生
    Args:
        saved_audio_path: 保存された音声ファイルのパス
        speed: 再生速度（1.0が通常速度、0.5で半分の速さ、2.0で倍速など）
    """
    try:
        if not os.path.exists(saved_audio_path):
            st.error("音声ファイルが見つかりません")
            return

        # 音声ファイルの読み込み
        audio = AudioSegment.from_wav(saved_audio_path)
        
        # 一時的な再生用ファイルを作成
        temp_play_path = saved_audio_path.replace('_saved.wav', f'_temp_play_{int(time.time())}.wav')
        
        # 速度を変更
        if speed != 1.0:
            # frame_rateを変更することで速度を調整
            modified_audio = audio._spawn(
                audio.raw_data, 
                overrides={"frame_rate": int(audio.frame_rate * speed)}
            )
            # 元のframe_rateに戻すことで正常再生させる（ピッチを保持したまま速度だけ変更）
            modified_audio = modified_audio.set_frame_rate(audio.frame_rate)
            modified_audio.export(temp_play_path, format="wav")
        else:
            # 速度変更がない場合は元のファイルをコピー
            audio.export(temp_play_path, format="wav")

        # Streamlitの音声再生機能を使用
        st.audio(temp_play_path, format="audio/wav", autoplay=True)
        
        # 少し待ってから一時ファイルを削除
        import threading
        def delayed_cleanup():
            time.sleep(2)
            if os.path.exists(temp_play_path):
                try:
                    os.remove(temp_play_path)
                except:
                    pass
        
        threading.Thread(target=delayed_cleanup, daemon=True).start()
        
    except Exception as e:
        st.error(f"音声再生エラー: {e}")