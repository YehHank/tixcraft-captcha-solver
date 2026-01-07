# tixcraft-captcha-solver

## 介紹
這個專案旨在使用卷積神經網路（CNN）來解決 Tixcraft 的驗證碼問題。

## 功能
- 自動識別並解決驗證碼
- 支援多種驗證碼類型

## 安裝
請確保您已經安裝了所需的 Python 套件。您可以使用以下命令來安裝這些套件：

```bash
pip install -r requirements.txt
```

建議使用虛擬環境（本專案預設使用 `.venv`）。

Windows PowerShell：

```powershell
python -m venv venv
./venv/Scripts/Activate.ps1
pip install -r requirements.txt
```

## 使用方法
1. 克隆此專案到本地端：
    ```bash
    git clone https://github.com/yourusername/tixcraft-captcha-solver.git
    ```
2. 進入專案目錄：
    ```bash
    cd tixcraft-captcha-solver
    ```
3. 生成驗證碼
    您可以使用 captcha_gen.py 來生成驗證碼圖像。
    ```bash
    python captcha_gen.py
    ```
4. 訓練模型（PyTorch）
    使用 train_torch.py 來訓練模型（會自動偵測 CUDA，無則回退 CPU）。
    ```bash
    python train_torch.py
    ```
5. 測試模型（PyTorch）
    使用 demo_torch.py 來測試訓練好的模型。
    ```bash
    python demo_torch.py
    ```
6. 啟用服務（PyTorch）
    使用 Flask 建立簡單 Web 服務，處理 Base64 圖像並回傳預測結果。
    ```bash
    python service_torch_base64.py
    ```

7. （舊版）Keras/TensorFlow 腳本
    舊版的 Keras/TensorFlow 腳本已在 PyTorch 遷移時移除；如需參考可從 git history 取得。

7. 日誌
    訓練過程中的日誌數據將會存儲在 logs/ 目錄中，您可以使用 TensorBoard 來可視化這些數據。
