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
4. 訓練模型
    您可以使用 train_cnn_imitate_4.py 來訓練模型。
    ```bash
    python train_cnn_imitate_4.py
    ```
5. 測試模型
    您可以使用 demo_cnn.py 來測試訓練好的模型。
    ```bash
    python demo_cnn.py
    ```
6. 啟用服務
    是一個使用 Flask 框架構建的簡單 Web 服務，該服務用於處理基於 Base64 編碼的圖像，並使用預訓練的 Keras 模型進行預測。
    ```bash
    python service_base64.py
    ```

7. 日誌
    訓練過程中的日誌數據將會存儲在 logs/ 目錄中，您可以使用 TensorBoard 來可視化這些數據。
