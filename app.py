import pandas as pd
from flask import Flask, request, jsonify
import mlflow.pyfunc

# Khởi tạo Flask App
app = Flask(__name__)

# CẤU HÌNH QUAN TRỌNG: Tên mô hình trong Registry
# Phải khớp với tên bạn đã đặt ở bước 'register_model.py'
MODEL_NAME = "PhanLoaiProject_Model"

# Đường dẫn động tới mô hình (MLflow 2.9.0+ không còn dùng stages)
# Sử dụng phiên bản mới nhất thay vì stage
model_uri = f"models:/{MODEL_NAME}/latest"

print(f"Đang tải mô hình từ: {model_uri} ...")

# Load mô hình ngay khi khởi động App
# Lưu ý: Nếu bạn chạy MLflow server riêng, cần set_tracking_uri trước dòng này
try:
    model = mlflow.pyfunc.load_model(model_uri)
    print(">>> Đã tải mô hình mới nhất thành công!")
except Exception as e:
    print(f"LỖI: Không thể tải mô hình. Đảm bảo bạn đã chạy bước Register Model. Chi tiết: {e}")
    model = None

@app.route('/', methods=['GET'])
def home():
    """Trang chủ đơn giản để kiểm tra trạng thái"""
    return """
    <h1>MLOps Project - Classification API</h1>
    <p>Mô hình đang sử dụng: <b>Latest Version</b> từ MLflow Registry.</p>
    <p>Gửi POST request tới <code>/predict</code> với dữ liệu JSON để dự đoán.</p>
    """

@app.route('/predict', methods=['POST'])
def predict():
    """API nhận dữ liệu và trả về kết quả dự đoán"""
    if not model:
        return jsonify({"error": "Mô hình chưa được tải."}), 500

    try:
        # Nhận dữ liệu JSON từ người dùng
        # Định dạng mong đợi: {"data": [[feature1, feature2, ..., feature20]]}
        req_data = request.get_json()
        input_data = req_data['data']
        
        # Chuyển đổi sang Pandas DataFrame (cần thiết cho mô hình Sklearn/MLflow)
        df = pd.DataFrame(input_data)
        
        # Thực hiện dự đoán
        prediction = model.predict(df)
        
        # Trả về kết quả
        return jsonify({
            "prediction": prediction.tolist(),
            "model_version": "Latest (Best Model)",
            "status": "success"
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    # Chạy ứng dụng trên cổng 5001 để tránh đụng độ với MLflow (5000)
    app.run(host='0.0.0.0', port=5001, debug=True)