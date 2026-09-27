FROM python:3.11-slim

WORKDIR /app

# v2：服务端专用依赖 —— CPU 版 torch，镜像从 ~7GB 瘦身到 ~1.5GB
# wheels/ 里放本地 torch CPU 轮子，build 阶段不依赖外网（离线部署模式）
# 注意：COPY 目录/ 只拷内容不拷目录本身，所以这里必须写两行分别指定
COPY requirements-server.txt ./
COPY wheels ./wheels
RUN pip install --no-cache-dir -r requirements-server.txt

COPY . .

EXPOSE 7860

CMD ["python", "app.py"]
