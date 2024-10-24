from flask import Flask, request, jsonify
from flask_cors import CORS
from keras.models import load_model
from keras import backend as K
from PIL import Image
import numpy as np
import base64
import io

# 字母表，用于映射预测的one-hot向量
LETTERSTR = "abcdefghjklmnopqrstuvwxyz"

# 加载模型
model = load_model("./data/model/imitate_4_model.keras")

# 初始化Flask
app = Flask(__name__)

# 允许所有来源的跨域请求
CORS(app)

def preprocess_image(image):
    """将图像处理成模型输入格式"""
    img = image.convert('RGB')  # 转换为 RGB 图像
    img = np.array(img) / 255.0  # 归一化到 [0, 1] 范围
    img = np.resize(img, (100, 120, 3))  # 确保图像大小为 100x120，通道数为 3
    img = np.expand_dims(img, axis=0)  # 添加批次维度
    return img

def predict_captcha(image):
    """使用加载的模型预测验证码"""
    K.clear_session()  # 清理Keras会话
    prediction = model.predict(image)
    
    # 将模型的one-hot输出转换为字符
    answer = ''
    for char in range(4):
        answer += LETTERSTR[np.argmax(prediction[char])]
    return answer

@app.route('/predict', methods=['POST'])
def predict():
    """处理POST请求并返回预测结果"""
    data = request.get_json()

    # 檢查是否有圖像數據
    if 'image' not in data:
        return jsonify({"error": "No image provided"}), 400

    # 提取Base64編碼的圖像數據
    try:
        image_data = data['image']
        # 去掉"data:image/png;base64,"這類前綴
        if ',' in image_data:
            image_data = image_data.split(',')[1]
        
        # 解碼Base64數據
        image_bytes = base64.b64decode(image_data)
        img = Image.open(io.BytesIO(image_bytes))

        # 圖像預處理
        processed_img = preprocess_image(img)

        # 預測驗證碼
        prediction = predict_captcha(processed_img)

        return jsonify({"prediction": prediction}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8888)