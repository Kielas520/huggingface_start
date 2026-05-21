import cv2
import torch
from transformers import pipeline
from PIL import Image  # 需要导入 PIL

# 1. 初始化模型，强制指定 mps (Mac GPU 加速)
print("正在加载模型，请稍候...")
# 使用更轻量的模型以获得更好的实时性
detector = pipeline("object-detection", model="facebook/detr-resnet-50", device="mps")

# 2. 打开摄像头
cap = cv2.VideoCapture(0)

print("开始实时检测，按 'q' 键退出...")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # OpenCV 读取的是 BGR，转换为 RGB 用于 PIL
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # 【关键修改】将 NumPy 数组转换为 PIL Image 对象
    pil_image = Image.fromarray(rgb_frame)

    # 3. 进行推理
    # 注意：实时检测可能导致屏幕卡顿，这是正常的
    # 传入 PIL Image 而不是 numpy 数组
    results = detector(pil_image)

    # 4. 绘制结果
    for obj in results:
        box = obj["box"]
        label = obj["label"]
        score = obj["score"]
        
        # 跳过低置信度结果
        if score < 0.8:
            continue

        xmin, ymin, xmax, ymax = int(box["xmin"]), int(box["ymin"]), int(box["xmax"]), int(box["ymax"])
        
        # 画框 (在原始的 BGR frame 上绘制)
        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
        # 画标签
        cv2.putText(frame, f"{label} {score:.2f}", (xmin, ymin - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # 5. 显示画面
    cv2.imshow("Real-time Detection", frame)

    # 按 'q' 退出
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()