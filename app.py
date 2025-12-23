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
print(f"Model đang sử dụng: {MODEL_NAME}")

print(f"Đang tải mô hình từ: {model_uri} ...")


# Không load model toàn cục nữa, sẽ load trong API GET
model = None

@app.route('/', methods=['GET'])
def home():
    """Trang chủ đơn giản để kiểm tra trạng thái"""
    from flask import render_template_string
    # Load model khi gọi API GET
    import mlflow.pyfunc
    import mlflow.tracking
    global model
    metrics_html = ''
    params_html = ''
    try:
        model = mlflow.pyfunc.load_model(model_uri)
        model_type = type(model).__name__
        # Lấy run_id của model latest từ MLflow Model Registry
        client = mlflow.tracking.MlflowClient()
        mv = client.get_latest_versions(MODEL_NAME, stages=None)[0]
        run_id = mv.run_id
        run = client.get_run(run_id)
        # Lấy metrics
        metrics = run.data.metrics
        if metrics:
            metrics_html = '<ul>' + ''.join([f'<li><b>{k}:</b> {v}</li>' for k, v in metrics.items()]) + '</ul>'
        # Lấy params
        params = run.data.params
        if params:
            params_html = '<ul>' + ''.join([f'<li><b>{k}:</b> {v}</li>' for k, v in params.items()]) + '</ul>'
        print(f"Model loaded in GET: {MODEL_NAME}, type: {model_type}, metrics: {metrics}, params: {params}")
    except Exception as e:
        model_type = 'None'
        metrics_html = f'Lỗi khi lấy metrics: {e}'
        params_html = f'Lỗi khi lấy params: {e}'
        print(metrics_html)

    html = f"""
    <html>
    <head><title>MLOps Project - Classification API</title></head>
    <body>
        <h1>MLOps Project - Classification API</h1>
        <p><b>Thông tin mô hình:</b></p>
        <ul>
            <li><b>Tên model:</b> {MODEL_NAME}</li>
            <li><b>Model URI:</b> {model_uri}</li>
            <li><b>Kiểu model:</b> {model_type}</li>
        </ul>
        <p><b>Metrics:</b></p>
        {metrics_html}
        <p><b>Parameters:</b></p>
        {params_html}
        <p>Gửi POST request tới <code>/predict</code> với dữ liệu JSON để dự đoán.</p>
        <hr>
    </body>
    </html>
    """
    return render_template_string(html)

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