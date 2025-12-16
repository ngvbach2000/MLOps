# Import các thư viện cần thiết
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

def prepare_data(n_samples=1000, n_features=20):
    """
    Hàm sinh dữ liệu giả lập cho bài toán phân loại.
    """
    # Sinh dữ liệu sử dụng make_classification như yêu cầu đề bài [1]
    X, y = make_classification(
        n_samples=n_samples,    # Số lượng mẫu dữ liệu
        n_features=n_features,  # Số lượng đặc trưng (cột)
        n_classes=2,            # Số lớp phân loại (ví dụ: 0 và 1)
        n_informative=15,       # Số đặc trưng mang thông tin hữu ích
        random_state=42         # Giữ cố định để kết quả tái lập được
    )
    
    # Chia dữ liệu thành tập Train (80%) và Test (20%)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"Đã sinh dữ liệu thành công: {n_samples} mẫu với {n_features} đặc trưng.")
    print(f"Kích thước tập Train: {X_train.shape}")
    print(f"Kích thước tập Test: {X_test.shape}")
    
    return X_train, X_test, y_train, y_test

import mlflow
import mlflow.sklearn
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification

# 1. Chuẩn bị dữ liệu (như bước trước)
X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

def train_svm_model(C=1.0, kernel='rbf'):
    """
    Hàm huấn luyện mô hình SVM và log kết quả vào MLflow.
    Tham số:
        C: Tham số Regularization (để tuning sau này)
        kernel: Loại kernel (linear, rbf, poly...)
    """
    # Bắt đầu một Run trong MLflow
    with mlflow.start_run():
        # A. Tạo và huấn luyện mô hình
        print(f"Đang huấn luyện SVM với C={C}, kernel={kernel}...")
        model = SVC(C=C, kernel=kernel)
        model.fit(X_train, y_train)
        
        # B. Dự đoán và đánh giá
        predictions = model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        f1 = f1_score(y_test, predictions)
        
        print(f"  Accuracy: {accuracy}")
        print(f"  F1 Score: {f1}")
        
        # C. Log tham số (Parameters) vào MLflow
        # (Để phục vụ yêu cầu so sánh kết quả các mô hình [1])
        mlflow.log_param("C", C)
        mlflow.log_param("kernel", kernel)
        mlflow.log_param("n_samples", len(X)) # Log thêm thông tin về dữ liệu
        
        # D. Log chỉ số đánh giá (Metrics) vào MLflow
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("f1_score", f1)
        
        # E. Log mô hình (Model) vào MLflow
        # Đây là bước quan trọng để sau này lưu vào Model Registry [1]
        mlflow.sklearn.log_model(model, "model")
        
        print("Đã hoàn thành Run và log vào MLflow.\n")

import mlflow
from mlflow.tracking import MlflowClient

def register_best_model():
    # 1. Kết nối và lấy thông tin Experiment
    client = MlflowClient()
    experiment_name = "MLOps_Project_PhanLoai" # Tên phải khớp với bước train
    experiment = client.get_experiment_by_name(experiment_name)
    
    if experiment is None:
        print("Không tìm thấy Experiment. Hãy chạy file train trước.")
        return

    # 2. Tìm kiếm tất cả các Runs trong Experiment
    # Sắp xếp giảm dần theo 'accuracy' để lấy cái cao nhất
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["metrics.accuracy DESC"] 
    )
    
    if not runs:
        print("Không tìm thấy Run nào.")
        return

    # 3. Lấy Run tốt nhất (đứng đầu danh sách)
    best_run = runs[0]
    best_run_id = best_run.info.run_id
    best_accuracy = best_run.data.metrics['accuracy']
    
    print(f"Run tốt nhất ID: {best_run_id}")
    print(f"Độ chính xác (Accuracy): {best_accuracy}")
    print(f"Tham số: {best_run.data.params}")

    # 4. Đăng ký mô hình vào Model Registry
    model_name = "PhanLoaiProject_Model" # Tên định danh cho mô hình trong Registry
    model_uri = f"runs:/{best_run_id}/model"
    
    print(f"Đang đăng ký mô hình '{model_name}' từ Run ID này...")
    model_version = mlflow.register_model(model_uri, model_name)
    
    # 5. Model Registry không còn sử dụng stages từ MLflow 2.9.0+
    # (transition_model_version_stage đã bị deprecated)
    
    print(f"Đã đăng ký thành công phiên bản {model_version.version}.")

if __name__ == "__main__":
    mlflow.set_experiment("MLOps_Project_PhanLoai")

    # =================================================================
    # LẦN 1: BASELINE (Mô hình cơ sở)
    # =================================================================
    print("--- RUN 1: Baseline ---")
    # Lý do: Thiết lập mốc so sánh ban đầu với các tham số mặc định phổ biến.
    # Kernel RBF thường hoạt động tốt với dữ liệu phi tuyến tính.
    train_svm_model(C=1.0, kernel='rbf') 

    # =================================================================
    # LẦN 2: TUNING THAM SỐ C (Regularization)
    # =================================================================
    print("--- RUN 2: Tăng tham số C lên 10 ---")
    # LÝ GIẢI HỢP LÝ: 
    # Giả thuyết rằng mô hình Baseline có thể đang bị 'Underfitting' (chưa học đủ sâu).
    # Việc tăng C (từ 1.0 lên 10.0) sẽ làm giảm mức độ Regularization, buộc mô hình 
    # phải cố gắng phân loại đúng nhiều điểm dữ liệu train hơn (Hard margin), 
    # kỳ vọng sẽ tăng độ chính xác (Accuracy) trên tập dữ liệu phức tạp.
    train_svm_model(C=10.0, kernel='rbf')

    # =================================================================
    # LẦN 3: TUNING THAM SỐ KERNEL (Thay đổi thuật toán nhân)
    # =================================================================
    print("--- RUN 3: Đổi sang Kernel Linear ---")
    # LÝ GIẢI HỢP LÝ:
    # Kiểm tra xem dữ liệu có phân tách tuyến tính (Linearly Separable) hay không.
    # Nếu dữ liệu đơn giản, Kernel 'linear' sẽ tính toán nhanh hơn nhiều so với 'rbf' 
    # và ít bị Overfitting hơn. Đây là bước kiểm tra cần thiết để tìm mô hình tối ưu về hiệu năng.
    train_svm_model(C=1.0, kernel='linear')

    # =================================================================
    # LẦN 4: THAY ĐỔI DỮ LIỆU (Theo gợi ý nguồn [1])
    # =================================================================
    print("--- RUN 4: Làm giàu dữ liệu (Tăng mẫu lên 2000) ---")
    # LÝ GIẢI HỢP LÝ:
    # Tài liệu [1] cho phép "thay đổi số lượng mẫu dữ liệu" để mô phỏng làm giàu dữ liệu.
    # Việc tăng số lượng mẫu từ 1000 lên 2000 giúp mô hình học được quy luật tổng quát hơn,
    # kỳ vọng tăng độ ổn định và điểm F1-score trên tập Test.
    
    # (Cần chạy lại bước sinh dữ liệu với n_samples=2000 trước khi train)
    X, y = make_classification(n_samples=2000, n_features=20, n_classes=2, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    train_svm_model(C=1.0, kernel='rbf') # Train lại với dữ liệu mới

    register_best_model()