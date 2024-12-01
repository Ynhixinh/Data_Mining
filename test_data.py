import pandas as pd

# Dữ liệu mẫu mới
data = {
    "Age": ["Young", "Young", "Young", "Middle", "Middle", "Middle", "Old", "Old", "Old", "Young", "Middle", "Middle", "Old"],
    "Income": ["Low", "Low", "High", "High", "High", "Low", "Low", "High", "High", "High", "Low", "Low", "Low"],
    "Education": ["High", "High", "Medium", "Low", "Low", "Medium", "Low", "High", "Medium", "Medium", "High", "High", "Medium"],
    "Buy": ["No", "No", "Yes", "Yes", "No", "No", "Yes", "Yes", "No", "Yes", "No", "Yes", "No"]
}

# Chuyển dữ liệu thành DataFrame
df = pd.DataFrame(data)

# Lưu DataFrame vào file CSV
df.to_csv('customer_buy_data.csv', index=False)


