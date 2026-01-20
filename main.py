import streamlit as st
import os
import time
from time import sleep
from pathlib import Path
from streamlit.components.v1 import html
from langchain.memory import ConversationSummaryBufferMemory
from langchain.chains import ConversationChain
from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
)
from langchain.schema import SystemMessage
from openai import OpenAI
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from pydub import AudioSegment
import functions as ft
import constants as ct


# 各種設定
load_dotenv()
st.set_page_config(
    page_title=ct.APP_NAME
)

# タイトル表示
st.markdown(f"## {ct.APP_NAME}")

# 初期処理
if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.mode = ct.MODE_1  # デフォルトモード
    st.session_state.speed = 1.0  # デフォルト速度
    st.session_state.current_step = "waiting"  # waiting, recording, processing
    st.session_state.recorded_audio = None
    
    # 録音コンポーネント用の初期化
    st.session_state.global_microphone_permission = False
    
    st.session_state.openai_obj = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    st.session_state.llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.5)
    st.session_state.memory = ConversationSummaryBufferMemory(
        llm=st.session_state.llm,
        max_token_limit=1000,
        return_messages=True
    )

    # モード「日常英会話」用のChain作成
    st.session_state.chain_basic_conversation = ft.create_chain(ct.SYSTEM_TEMPLATE_BASIC_CONVERSATION)

# UI設定
st.session_state.mode = st.selectbox(
    "モードを選択", 
    options=[ct.MODE_1, ct.MODE_2], 
    index=0,
    help="練習したいモードを選んでください"
)

st.session_state.speed = st.selectbox(
    "再生速度", 
    options=ct.PLAY_SPEED_OPTION, 
    index=3,
    format_func=lambda x: f"{x}x"
)

with st.chat_message("assistant", avatar="images/ai_icon.jpg"):
    st.markdown("こちらは生成AIによる音声英会話の練習アプリです。何度も繰り返し練習し、英語力をアップさせましょう。")
    st.markdown("**【操作説明】**")
    st.info("""
    📱 **使い方**:
    1. モードと再生速度を選択
    2. **マイクボタン1回目のクリック**: 録音開始
    3. **英語で話す** (好きなだけ長時間OK)
    4. **マイクボタン2回目のクリック**: 録音停止
    5. AIが応答を自動音声再生します
    
    💡 **ポイント**: 
    - 録音時間は自分でコントロール可能
    - 初回のみマイクアクセス許可が必要
    """)

# マイクテスト
st.markdown("### 🎤 マイクテスト")
test_audio = ft.record_audio_simple("test")
if test_audio is not None and len(test_audio) > 100:
    st.success("✅ マイクテスト成功！録音機能が正常に動作しています。")
elif test_audio is not None:
    st.warning("⚠️ 録音データが検出されましたが短すぎます。録音開始→話す→録音停止の流れでテストしてください。")
else:
    if st.session_state.get("global_microphone_permission", False):
        st.info("⬆️ 上のマイクボタンで録音テストをしてください。")
    else:
        st.info("⬆️ 上のマイクボタンをクリックしてマイクアクセス許可を行ってください（初回のみ）。")

st.divider()

# メッセージリストの一覧表示（最新の会話のみ表示）
if st.session_state.messages:
    # 最新メッセージのみを表示
    latest_messages = st.session_state.messages[-2:] if len(st.session_state.messages) >= 2 else st.session_state.messages
    for idx, message in enumerate(latest_messages):
        actual_idx = len(st.session_state.messages) - len(latest_messages) + idx
        if message["role"] == "assistant":
            with st.chat_message(message["role"], avatar="images/ai_icon.jpg"):
                st.markdown(message["content"])
                # 日常英会話モードでかつAIメッセージに音声ファイルが関連付けされている場合
                if (st.session_state.mode == ct.MODE_1 and 
                    "audio_path" in message and 
                    message["audio_path"] and 
                    os.path.exists(message["audio_path"])):
                    
                    col_msg_replay1, col_msg_replay2 = st.columns([1, 4])
                    with col_msg_replay1:
                        # 各メッセージ用の一意なキーを生成
                        replay_key = f"replay_latest_{actual_idx}"
                        if st.button("🔊 再読み上げ", key=replay_key, use_container_width=True):
                            success = ft.play_audio_web_compatible(message["audio_path"], st.session_state.speed)
                            if success:
                                st.toast("音声を再生しました", icon="🔊")
                            else:
                                st.toast("音声再生に失敗しました", icon="❌")
        elif message["role"] == "user":
            with st.chat_message(message["role"], avatar="images/user_icon.jpg"):
                st.markdown(message["content"])

# メイン機能
st.markdown("### 🗣️ 音声英会話練習")

# 現在のステップ表示
if st.session_state.current_step == "waiting":
    if st.session_state.get("global_microphone_permission", False):
        st.info("🎤 **録音開始**: マイクボタンをクリック → 話す → **録音停止**: もう一度マイクボタンをクリック")
    else:
        st.warning("📱 マイクアクセス許可が必要です（初回のみ）。下のマイクボタンをクリックしてブラウザで「許可」を選択してください。")
        
        # Safari専用ガイダンスを追加
        with st.expander("🍎 Safari利用の方へ - 毎回許可が求められる場合"):
            st.markdown("""
            **Safari で毎回許可が求められる場合の解決方法:**
            
            1. **サイト設定を確認**:
               - アドレスバー左の「🔒」または「AA」をクリック
               - 「Webサイトの設定」を選択
               - 「マイク」を「許可」に設定
            
            2. **ページを再読み込み**してからご利用ください
            
            3. それでも解決しない場合は **Chrome** または **Edge** の使用をお勧めします
            """)
elif st.session_state.current_step == "recording":
    st.warning("🔴 **録音中...** 話し終わったら **マイクボタンをもう一度クリック** して停止してください")
elif st.session_state.current_step == "processing":
    st.info("⚙️ 音声を処理中... しばらくお待ちください")

# 録音機能（常に表示、ただし処理中は無効化表示）
recorded_audio = ft.record_audio_simple("main")

# 録音データの処理
if recorded_audio is not None and len(recorded_audio) > 50:  # 最小バイト数を緩和（100→50）
    # 新しい録音データかつ、現在処理中でない場合のみ処理開始
    if (st.session_state.recorded_audio != recorded_audio and 
        st.session_state.current_step == "waiting"):
        st.session_state.recorded_audio = recorded_audio
        st.session_state.current_step = "processing"
        st.rerun()

# 音声処理（processing状態の場合のみ）
if st.session_state.current_step == "processing" and st.session_state.recorded_audio:
    # 処理開始前に録音データをクリア（重複処理を防ぐ）
    current_audio = st.session_state.recorded_audio
    st.session_state.recorded_audio = None
    
    # 音声ファイルを保存
    audio_input_file_path = f"{ct.AUDIO_INPUT_DIR}/audio_input_{int(time.time())}.wav"
    
    if ft.save_audio_to_file(current_audio, audio_input_file_path):
        # 音声認識
        with st.spinner('音声をテキストに変換中...'):
            transcript = ft.transcribe_audio(audio_input_file_path)
            audio_input_text = transcript.text

        # ユーザー入力を表示
        with st.chat_message("user", avatar=ct.USER_ICON_PATH):
            st.markdown(audio_input_text)

        # モード別処理
        if st.session_state.mode == ct.MODE_1:  # 日常英会話
            # AI応答生成
            with st.spinner("AI応答を生成中..."):
                llm_response = st.session_state.chain_basic_conversation.predict(input=audio_input_text)
                
                # 音声合成
                llm_response_audio = st.session_state.openai_obj.audio.speech.create(
                    model="tts-1",
                    voice="alloy",
                    input=llm_response
                )

                # 音声ファイル保存・再生
                audio_output_file_path = f"{ct.AUDIO_OUTPUT_DIR}/audio_output_{int(time.time())}.wav"
                ft.save_to_wav(llm_response_audio.content, audio_output_file_path)
                
                # AI応答を表示
                with st.chat_message("assistant", avatar=ct.AI_ICON_PATH):
                    st.markdown(llm_response)
                    st.info("🔊 音声を自動再生中...")
                
                # Web標準のブラウザ音声再生（localhost/クラウド両対応）
                print(f"[MAIN] Web音声再生開始: {audio_output_file_path}")
                
                # ブラウザでの音声再生
                success = ft.play_audio_web_compatible(audio_output_file_path, st.session_state.speed)
                
                if success:
                    st.success("🔊 音声再生完了（ブラウザ再生）")
                else:
                    st.error("❌ 音声再生に失敗しました")
                
                # 再読み上げ用ファイルを保存（一意なファイル名で）
                timestamp = int(time.time())
                saved_audio_path = f"{ct.AUDIO_OUTPUT_DIR}/audio_saved_{timestamp}.wav"
                audio_for_save = AudioSegment.from_wav(audio_output_file_path)
                audio_for_save.export(saved_audio_path, format="wav")
                
                # このメッセージ専用の音声ファイルパスを保存
                current_message_audio_path = saved_audio_path
                
                # 少し遅延してから元のファイルを削除
                import threading
                def delayed_cleanup():
                    time.sleep(3)  # 3秒後に削除
                    if os.path.exists(audio_output_file_path):
                        try:
                            os.remove(audio_output_file_path)
                        except:
                            pass
                
                threading.Thread(target=delayed_cleanup, daemon=True).start()

            # メッセージ履歴に追加（正しい音声ファイルパスで）
            st.session_state.messages.append({"role": "user", "content": audio_input_text})
            st.session_state.messages.append({
                "role": "assistant", 
                "content": llm_response,
                "audio_path": current_message_audio_path
            })

        elif st.session_state.mode == ct.MODE_2:  # シャドーイング
            # シャドーイング用の処理（簡素化）
            st.info("シャドーイングモードは今後実装予定です")

        # 処理完了後の状態リセット
        st.session_state.current_step = "waiting"
        # 録音データもクリア
        st.session_state.recorded_audio = None
        # 成功メッセージを表示
        st.success("✅ 音声処理が完了しました。次の録音をどうぞ！")
        
        # UI更新のために再実行（録音ボタンを再表示）
        st.rerun()
        
    else:
        # 音声ファイル保存に失敗した場合
        st.error("音声ファイルの保存に失敗しました。もう一度録音してください。")
        st.session_state.current_step = "waiting"
        st.session_state.recorded_audio = None
        # UI更新のために再実行
        st.rerun()

st.divider()

# 会話履歴表示（全履歴）
if len(st.session_state.messages) > 2:
    st.markdown("### 📝 会話履歴")
    for idx, message in enumerate(st.session_state.messages[:-2]):  # 最新2件以外を表示
        if message["role"] == "assistant":
            with st.chat_message(message["role"], avatar="images/ai_icon.jpg"):
                st.markdown(message["content"])
                # 再読み上げボタン
                if "audio_path" in message and message["audio_path"] and os.path.exists(message["audio_path"]):
                    if st.button("🔊 再読み上げ", key=f"history_replay_{idx}", use_container_width=True):
                        success = ft.play_audio_web_compatible(message["audio_path"], st.session_state.speed)
                        if success:
                            st.toast("音声を再生しました", icon="🔊")
                        else:
                            st.toast("音声再生に失敗しました", icon="❌")
                else:
                    st.caption("⚠️ 音声ファイルが利用できません")
        elif message["role"] == "user":
            with st.chat_message(message["role"], avatar="images/user_icon.jpg"):
                st.markdown(message["content"])