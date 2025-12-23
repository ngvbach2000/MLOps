# Sử dụng Python 3.9 làm môi trường cơ sở (Base Image)
# Dùng bản slim để image nhẹ hơn
FROM python:3.9-slim

# Thiết lập thư mục làm việc bên trong container là /app
WORKDIR /app

# Copy file requirements.txt vào trong container trước
COPY requirements.txt .

# Cài đặt các thư viện cần thiết
RUN pip install --no-cache-dir -r requirements.txt

# Copy toàn bộ mã nguồn dự án (bao gồm app.py, và thư mục mlruns) vào container
COPY . .

# Mở cổng 5001 (Cổng mà Flask app của bạn đang chạy)
EXPOSE 5001

# Lệnh mặc định, có thể bị override bởi docker-compose
CMD ["python", "app.py"]