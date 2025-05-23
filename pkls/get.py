import pickle

# 假设文件路径为 'file.pickle'
with open('/home/ssy/mmrotate/pkls/result.pkl', 'rb') as f:  # 以二进制方式打开文件
    data = pickle.load(f)  # 读取并反序列化
    print(data)  # 打印读取到的数据
