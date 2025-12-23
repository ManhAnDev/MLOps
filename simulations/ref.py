import requests

# Tên các feature theo thứ tự
FEATURE_ORDER = [
    "fixed_acidity",
    "volatile_acidity",
    "citric_acid",
    "residual_sugar",
    "chlorides",
    "free_sulfur_dioxide",
    "total_sulfur_dioxide",
    "density",
    "pH",
    "sulphates",
    "alcohol",
    "quality",
    "some_extra"  # Nếu model có 13 feature, đổi tên này cho đúng nếu cần
]

# Dữ liệu đầy đủ
rows = [
    [14.34,1.68,2.7,25,98,2.8,1.31,0.53,2.7,13,0.57,1.96,660],
    [12.53,5.51,2.64,25,96,1.79,0.6,0.63,1.1,5,0.82,1.69,515],
    [12.37,1.07,2.1,18.5,88,3.52,3.75,0.24,1.95,4.5,1.04,2.77,660],
    [13.48,1.67,2.64,22.5,89,2.6,1.1,0.52,2.29,11.75,0.57,1.78,620],
    [13.07,1.5,2.1,15.5,98,2.4,2.64,0.28,1.37,3.7,1.18,2.69,1020]
]

# Chuyển sang list of dicts
data = [dict(zip(FEATURE_ORDER, row)) for row in rows]

# Payload chuẩn
payload = {
    "data": data,
    "feature_names": FEATURE_ORDER
}

# Endpoint upload reference (có thể khác tùy cấu hình Evidently)
url = "http://localhost:8001/reference"

# Gửi request
response = requests.post(url, json=payload)

print(response.status_code)
print(response.text)
