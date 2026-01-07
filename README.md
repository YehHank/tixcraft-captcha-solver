# tixcraft-captcha-solver

使用 PyTorch 訓練/推論一個多頭 CNN（每個字元一個 head），用來辨識 Tixcraft 類型的 4 碼字母驗證碼。

Repo：<https://github.com/YehHank/tixcraft-captcha-solver>

## 特色
- PyTorch 訓練腳本（自動偵測 CUDA，無則回退 CPU）
- 評估腳本（輸出 full-string accuracy 與每位數字元 accuracy）
- Flask 推論服務：Base64 圖片 -> 預測文字
- 資料集以 CSV（image_id,label）管理，搭配資料夾的 PNG 圖片

## 環境需求
- Python 3.10+（建議 3.10/3.11）
- （選用）NVIDIA GPU：安裝對應驅動後，依 `requirements.txt` 會安裝 PyTorch CUDA wheels
  - 本專案目前使用 `cu130` 來源（見 `requirements.txt`）
  - 沒有 GPU 也能跑：程式會自動使用 CPU（只是比較慢）

## 安裝（Windows PowerShell）
建議使用虛擬環境（此專案範例使用 `venv`）。

```powershell
cd g:\Projects\tixcraft-captcha-solver

python -m venv venv
./venv/Scripts/Activate.ps1

python -m pip install -U pip
pip install -r requirements.txt
```

## 產生訓練資料（captcha_gen）
在生成圖片前，請先放入至少一個字型檔到 `./data/font`（`.ttf` / `.otf`）。

```bash
python captcha_gen.py
```

預設會在以下資料夾產生：
- `./data/4_imitate_train_set/`
- `./data/4_imitate_vali_set/`
- `./data/manual_label/`

注意：字型與模型等大型二進位檔通常不會提交到 git（見 `.gitignore`）。

## 訓練（train_torch）
```bash
python train_torch.py --epochs 50 --batch-size 32 --lr 0.001 --out ./data/model/imitate_4_model_torch.pt
```

訓練過程會印出：
- `Using device: cuda` 或 `Using device: cpu`
- 每個 epoch 的 train/val loss 與 accuracy

## 評估（demo_torch）
```bash
python demo_torch.py --ckpt ./data/model/imitate_4_model_torch.pt
```

## 推論服務（service_torch_base64）
啟動：

```bash
python service_torch_base64.py --ckpt ./data/model/imitate_4_model_torch.pt --host 0.0.0.0 --port 8888
```

API：
- `POST /predict`
- Body（JSON）：`{"image": "<base64 or data-url>"}`
- Response：`{"prediction": "abcd", "config": {...}}`

## 備註
- 舊版 Keras/TensorFlow 腳本已在 PyTorch 遷移後移除；如需參考可從 git history 取得。
