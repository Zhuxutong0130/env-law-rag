import torch

print("PyTorch 版本:", torch.__version__)
print("CUDA 是否可用:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("显卡型号:", torch.cuda.get_device_name(0))
    print("显存总量: %.1f GB" % (torch.cuda.get_device_properties(0).total_memory / 1024**3))

    # 真刀真枪算一次：两个大矩阵相乘，强制在 GPU 上执行
    a = torch.randn(2000, 2000, device="cuda")
    b = torch.randn(2000, 2000, device="cuda")
    c = a @ b
    torch.cuda.synchronize()
    print("GPU 矩阵乘法测试: 通过，结果形状 =", tuple(c.shape))
else:
    print("!! CUDA 不可用，检查安装命令是否用了 --index-url cu128")

