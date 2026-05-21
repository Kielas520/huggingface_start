from datasets import load_dataset
from renumics import spotlight

# 1. 加载一个经典的图片分类数据集 (beans)
# 我们只取前 100 条数据进行演示，避免载入太慢
print("正在加载数据集...")
ds = load_dataset("beans", split="train[:100]")

# 2. 启动 Spotlight 可视化
# 这一行代码会启动一个本地服务器，并自动在浏览器打开页面
print("正在启动可视化界面，请查看浏览器...")
spotlight.show(ds)