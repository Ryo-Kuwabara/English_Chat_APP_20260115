# 🗣️ English Conversation Practice App

**AI音声英会話練習アプリ** - Streamlitで構築された音声対話型英語学習ツール

## ✨ 主な機能

- 🎤 **音声録音**: ブラウザマイクを使用した英語音声入力
- 🤖 **AI対話**: OpenAI GPT-4o-miniによる自然な英会話応答  
- 🔊 **音声再生**: OpenAI TTSによる高品質な音声出力
- 📝 **会話履歴**: 過去の対話記録と再読み上げ機能
- 🎵 **再生制御**: 音声速度調整と個別メッセージ再生

## 🚀 最新の改善点 (2026年1月)

### 🔧 音声再生エラー修正
- **PyAudio出力デバイスエラー** `[Errno -9996]` の完全解決
- macOS環境での音声デバイス自動検出機能
- `afplay`コマンドによる代替再生システム
- 音声再生成功/失敗の明確なフィードバック

### 🍎 Safari完全対応
- **Safari環境での音声機能が正常動作**
- マイクロフォン許可問題への実用的な解決ガイド
- ブラウザ互換性の大幅向上

### 🎯 ユーザー体験向上
- 音声再生失敗時の自動フォールバック機能
- エラー状況での適切なユーザー通知
- 詳細なデバッグログによる問題特定の簡素化

## 🖥️ 対応環境

| ブラウザ | 音声録音 | 音声再生 | 推奨度 |
|----------|----------|----------|--------|
| **Safari** | ✅ | ✅ | ⭐⭐⭐ |
| **Chrome** | ✅ | ✅ | ⭐⭐⭐ |
| **Edge** | ✅ | ✅ | ⭐⭐⭐ |
| **Firefox** | ✅ | ✅ | ⭐⭐ |

## ⚙️ セットアップ

### 1. 環境準備
```bash
# リポジトリクローン
git clone https://github.com/Ryo-Kuwabara/English_Chat_APP_20260115.git
cd English_Chat_APP_20260115

# 仮想環境作成・有効化
python -m venv .venv
source .venv/bin/activate  # macOS/Linux
# .venv\Scripts\activate  # Windows

# 依存パッケージインストール  
pip install -r requirements.txt
```

### 2. API設定
```bash
# .envファイル作成
cp .env.example .env

# OpenAI APIキーを設定
OPENAI_API_KEY=your_openai_api_key_here
```

### 3. アプリ起動
```bash
streamlit run main.py
```

ブラウザで http://localhost:8501 にアクセス

## 🎵 使い方

1. **初回設定**: マイクロフォン許可を「許可」に設定
2. **音声録音**: 🎤ボタンをクリックして英語で話す
3. **AI応答**: 自動で音声認識→AI応答→音声再生
4. **再読み上げ**: 各メッセージの🔊ボタンで再生
5. **会話継続**: 自然な英会話を楽しむ

## 🍎 Safari利用者向けガイド

### マイク許可の永続化方法:
1. アドレスバー左の「🔒」または「AA」をクリック
2. 「Webサイトの設定」を選択  
3. 「マイク」を「許可」に設定
4. ページを再読み込み

## 📚 技術スタック

- **フロントエンド**: Streamlit
- **音声録音**: audio-recorder-streamlit
- **音声認識**: OpenAI Whisper API
- **AI対話**: OpenAI GPT-4o-mini
- **音声合成**: OpenAI TTS API
- **音声処理**: PyAudio, pydub
- **対話管理**: LangChain

## 🐛 トラブルシューティング

### 音声再生エラー
- **macOS**: `afplay`による自動代替再生
- **デバイス問題**: 利用可能デバイスの自動検出
- **ブラウザ制限**: st.audioによるフォールバック

### マイク許可問題  
- **Safari**: サイト設定でマイク許可を永続化
- **Chrome/Edge**: より安定した許可管理
- **HTTPS推奨**: 本番環境でのセキュア接続

## 📈 今後の予定

- [ ] HTTPS対応による完全なブラウザ互換性
- [ ] 追加の英会話モード（ビジネス、日常会話など）
- [ ] 音声品質の更なる向上
- [ ] モバイル端末対応の最適化

## 🤝 貢献

プルリクエスト、Issue報告、機能提案を歓迎します！

## 📄 ライセンス

MIT License

---

**🎯 Perfect for English learners who want to practice speaking with AI!**