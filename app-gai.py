import warnings
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')
import streamlit as st
from streamlit_option_menu import option_menu
import cv2
import os
import requests
from PIL import Image
from io import BytesIO
import numpy as np
from ultralytics import YOLO
from urllib.parse import urlparse
import tempfile
import urllib.request
from collections import defaultdict
import time
from datetime import datetime
import shutil

if not os.path.exists('history'):
    os.makedirs('history')

# 在这可加入别的权重文件
model_map = {
    'FPCDet': 'runs/detect/train4/weights/best.pt',
    #'yolov8': 'runs/detect/train/weights/best.pt'
}

st.set_page_config(page_title="在线检测平台", page_icon=":desktop_computer:")

st.markdown("<h1 style='text-align: center'>在线检测平台</h1>", unsafe_allow_html=True)
with st.sidebar:
    model_select = st.selectbox("检测模型选择", list(model_map.keys()))
    if model_select not in st.session_state:
        st.session_state[model_select] = YOLO(model_map[model_select])
    selected = option_menu("功能列表", ["首页", "图片检测", "视频检测", "摄像头检测", "历史检测", "计数可视化"],
                           icons=['house-gear', 'image', 'film', 'webcam', 'hourglass-split', 'graph-up'],
                           menu_icon="list-stars", default_index=0)


def predict_image():
    results = st.session_state[model_select].predict(source='predict_img.jpg', **{'save': False, 'show_conf': True},
                                                     stream=True)
    # 绘制检测结果
    for result in results:
        label_dic = result.names
        frame = result.orig_img
        for a, b, c in zip(result.boxes.xyxy.cpu().numpy(), result.boxes.cls.cpu().numpy(), result.boxes.conf):
            # 获取边界框坐标和标签
            x1, y1, x2, y2 = map(int, a)  # 边界框坐标
            label = label_dic[b]
            confidence = c

            # 在图像上绘制边界框和标签
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 5)
            text = f"{label} ({confidence:.2f})"
            font_scale = 1
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_thickness = 2

            text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]
            text_x, text_y = x1, y1 - 10
            text_bg_x1, text_bg_y1 = text_x, text_y - text_size[1]
            text_bg_x2, text_bg_y2 = text_x + text_size[0], text_y

            cv2.rectangle(frame, (text_bg_x1, text_bg_y1), (text_bg_x2, text_bg_y2), (0, 0, 255), -1)
            cv2.putText(frame, text, (text_x, text_y), font, font_scale, (255, 255, 255), font_thickness,
                        lineType=cv2.LINE_AA)
        # 转换为RGB格式
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return frame_rgb, result.boxes.xyxy.cpu().numpy().shape[0]


def predict_video(video_path):
    if 'video_path' in locals():
        with st.spinner("检测中，请稍等..."):
            cap = cv2.VideoCapture(video_path)
            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)

            # 设置输出视频文件
            output_path = "predict_video.mp4"
            fourcc = cv2.VideoWriter_fourcc(*'X264')
            out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

            frame_rate_divider = 1  # 每1帧进行一次检测
            frame_count = 0
            counts = defaultdict(int)
            object_str = ""
            index = 0
            detection_times = []  # 这里应该是实际的检测时间数据
            detection_counts = []  # 这里应该是实际的检测个数数据

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                # 每隔 frame_rate_divider 帧检测一次
                if frame_count % frame_rate_divider == 0:
                    results = st.session_state[model_select].predict(source=frame,
                                                                     **{'save': False, 'show_conf': False}, stream=True)

                    key = f"({index}): "
                    index += 1
                    for result in results:
                        label_dic = result.names
                        for a, b, c in zip(result.boxes.xyxy.cpu().numpy(), result.boxes.cls.cpu().numpy(),
                                           result.boxes.conf):
                            # 获取边界框坐标和标签
                            x1, y1, x2, y2 = map(int, a)  # 边界框坐标
                            label = label_dic[b]
                            confidence = c

                            # 在图像上绘制边界框和标签
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 5)
                            text = f"{label} ({confidence:.2f})"
                            font_scale = 1
                            font = cv2.FONT_HERSHEY_SIMPLEX
                            font_thickness = 2

                            text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]
                            text_x, text_y = x1, y1 - 10
                            text_bg_x1, text_bg_y1 = text_x, text_y - text_size[1]
                            text_bg_x2, text_bg_y2 = text_x + text_size[0], text_y

                            cv2.rectangle(frame, (text_bg_x1, text_bg_y1), (text_bg_x2, text_bg_y2), (0, 0, 255), -1)
                            cv2.putText(frame, text, (text_x, text_y), font, font_scale, (255, 255, 255),
                                        font_thickness, lineType=cv2.LINE_AA)
                    counts = defaultdict(int)  # 重置计数
                    current_time = time.time()
                    detection_times.append(current_time)  # current_time为当前时间
                    detection_counts.append(result.boxes.xyxy.cpu().numpy().shape[0])  # count_num为当前检测个数

                # 写入输出视频
                out.write(frame)
                frame_count += 1

            # 关闭文件
            cap.release()
            out.release()
            video_file = open(output_path, "rb")
            video_bytes = video_file.read()
            # st.video(video_bytes)
    else:
        st.error("视频检测失败", icon='🚨')


if selected == '首页':
    st.subheader("本平台是基于yolo算法对图片、视频和摄像头三种输入实现对柑橘进行检测，图片与视频检测功能支持文件上传或URL输入形式进行检测，每次检测的结果都可在历史检测中查看")

elif selected == '图片检测':
    options = [":material/image:读取图片", ":material/link:读取URL"]
    selection = st.pills("请选择读取方式", options, default=":material/image:读取图片")
    if selection == options[0]:
        uploaded_file = st.file_uploader("请上传需检测的图像", type=["jpg", "jpeg", "png"])
        start_btn = st.button("检测")
        col1, col2 = st.columns([5, 5])
        if start_btn:
            if uploaded_file is not None:
                with open('predict_img.jpg', "wb") as f:
                    f.write(uploaded_file.getbuffer())
                f.close()
                frame_rgb, count_num = predict_image()
                timestamp = int(time.time())
                cv2.imwrite("history/{}_image.jpg".format(timestamp), cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))
                with open("history.txt", mode='a+', encoding='utf8') as f:
                    f.write("{}\t{}\n".format('history/{}_image.jpg'.format(timestamp),
                                              datetime.now().strftime('%Y-%m-%d')))
                f.close()
                with col1:
                    st.markdown("<h4 style='text-align: center'>原始图像</h4>", unsafe_allow_html=True)
                    st.image('predict_img.jpg')
                with col2:
                    st.markdown("<h4 style='text-align: center'>结果图像</h4>", unsafe_allow_html=True)
                    st.image(frame_rgb, channels="RGB")
                    st.markdown("<h4 style='text-align: center'>检测个数: {}</h4>".format(count_num),
                                unsafe_allow_html=True)
                img_pil = Image.fromarray(frame_rgb)
                buf = BytesIO()
                img_pil.save(buf, format="JPEG")
                byte_im = buf.getvalue()

                st.download_button(
                    label="Download result",
                    data=byte_im,
                    file_name="{}_image_file_predict.jpg".format(int(time.time())),
                    mime="image/jpg"
                )
            else:
                st.warning("未上传文件")
                st.stop()
    elif selection == options[1]:
        img_url = st.text_input("请输入图片URL")
        if st.button("检测"):
            try:
                response = requests.get(img_url, stream=True)
                response.raise_for_status()

                # 将图片数据写入本地文件
                with open('predict_img.jpg', "wb") as file:
                    for chunk in response.iter_content(1024):
                        file.write(chunk)
                file.close()
                st.success("成功读取图片", icon='✅')
            except requests.exceptions.RequestException as e:
                st.error("读取图片出错", icon='🚨')
                st.stop()

            frame_rgb, count_num = predict_image()
            timestamp = int(time.time())
            cv2.imwrite("history/{}_image.jpg".format(timestamp), cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))
            with open("history.txt", mode='a+', encoding='utf8') as f:
                f.write("{}\t{}\n".format('history/{}_image.jpg'.format(timestamp),
                                          datetime.now().strftime('%Y-%m-%d')))
            f.close()
            col1, col2 = st.columns([5, 5])
            with col1:
                st.image("predict_img.jpg")
            with col2:
                st.image(frame_rgb, channels="RGB")
                st.markdown("<h4 style='text-align: center'>检测个数: {}</h4>".format(count_num), unsafe_allow_html=True)
            img_pil = Image.fromarray(frame_rgb)
            buf = BytesIO()
            img_pil.save(buf, format="JPEG")
            byte_im = buf.getvalue()

            st.download_button(
                label="Download result",
                data=byte_im,
                file_name="{}_image_url_predict.jpg".format(int(time.time())),
                mime="image/jpg"
            )

elif selected == '视频检测':
    options = [":material/movie:读取视频", ":material/link:读取URL"]
    selection = st.pills("请选择读取方式", options, default=":material/movie:读取视频")
    if selection == options[0]:
        uploaded_file = st.file_uploader("上传视频文件", type=["mp4", "mov"])
        start_btn = st.button("检测")
        col1, col2 = st.columns([5, 5])
        if start_btn:
            if uploaded_file is not None:
                temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
                temp_file.write(uploaded_file.read())
                video_path = temp_file.name
                predict_video(video_path)
                with col1:
                    st.video(video_path, autoplay=True, muted=True)
                with col2:
                    st.video(open("predict_video.mp4", "rb").read(), autoplay=True, muted=True)
                timestamp = int(time.time())
                shutil.copy('./predict_video.mp4', 'history/{}_video.mp4'.format(timestamp))
                with open("history.txt", mode='a+', encoding='utf8') as f:
                    f.write("{}\t{}\n".format('history/{}_video.mp4'.format(timestamp),
                                              datetime.now().strftime('%Y-%m-%d')))
                f.close()
                with open("predict_video.mp4", "rb") as file:
                    btn = st.download_button(
                        label="Download result",
                        data=file,
                        file_name="{}_video_file_predict.mp4".format(int(time.time())),
                        mime="video/mp4",
                    )
            else:
                st.warning("未上传文件")
                st.stop()
    elif selection == options[1]:
        video_url = st.text_input("请输入视频URL")
        if st.button("检测"):
            try:
                # 下载视频到本地临时文件
                temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
                urllib.request.urlretrieve(video_url, temp_file.name)
                video_path = temp_file.name
                st.success("成功读取视频", icon='✅')
            except:
                st.error("读取视频出错", icon='🚨')
                st.stop()
            predict_video(video_path)
            col1, col2 = st.columns([5, 5])
            with col1:
                st.video(video_path, autoplay=True, muted=True)
            with col2:
                st.video(open("predict_video.mp4", "rb").read(), autoplay=True, muted=True)
            timestamp = int(time.time())
            shutil.copy('./predict_video.mp4', 'history/{}_video.mp4'.format(timestamp))
            with open("history.txt", mode='a+', encoding='utf8') as f:
                f.write("{}\t{}\n".format('history/{}_video.mp4'.format(timestamp),
                                          datetime.now().strftime('%Y-%m-%d')))
            f.close()
            with open("predict_video.mp4", "rb") as file:
                btn = st.download_button(
                    label="Download result",
                    data=file,
                    file_name="{}_video_url_predict.mp4".format(int(time.time())),
                    mime="video/mp4",
                )

elif selected == '摄像头检测':
    run = st.toggle("开启摄像头")
    frame_placeholder = st.empty()

    # 捕获摄像头实时流
    cap = cv2.VideoCapture(0)  # 0代表默认摄像头

    # 运行实时检测循环
    while run:
        ret, frame = cap.read()
        if not ret:
            st.error("无法读取摄像头输入", icon='🚨')
            break

        results = st.session_state[model_select].predict(source=frame, **{'save': False, 'show_conf': False},
                                                         stream=True)

        # 绘制检测结果
        for result in results:
            label_dic = result.names
            for a, b, c in zip(result.boxes.xyxy.cpu().numpy(), result.boxes.cls.cpu().numpy(), result.boxes.conf):
                # 获取边界框坐标和标签
                x1, y1, x2, y2 = map(int, a)  # 边界框坐标
                label = label_dic[b]
                confidence = c

                # 在图像上绘制边界框和标签
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 5)
                text = f"{label} ({confidence:.2f})"
                font_scale = 1
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_thickness = 2

                text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]
                text_x, text_y = x1, y1 - 10
                text_bg_x1, text_bg_y1 = text_x, text_y - text_size[1]
                text_bg_x2, text_bg_y2 = text_x + text_size[0], text_y

                cv2.rectangle(frame, (text_bg_x1, text_bg_y1), (text_bg_x2, text_bg_y2), (0, 0, 255), -1)
                cv2.putText(frame, text, (text_x, text_y), font, font_scale, (255, 255, 255), font_thickness,
                            lineType=cv2.LINE_AA)
        # 转换为RGB格式
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        frame_placeholder.image(frame_rgb, channels="RGB")

        # 控制刷新频率，每10ms刷新一次
        cv2.waitKey(10)

    # 停止摄像头
    cap.release()

elif selected == '历史检测':
    with open("history.txt", "r", encoding='utf8') as f:
        res = f.readlines()
    f.close()
    show_info = [_.split('\t') for _ in res]
    col1, col2, col3 = st.columns([3, 3, 3])
    with col1:
        options = [":material/image:图片检测历史", ":material/movie:视频检测历史"]
        selection = st.pills("请选择浏览的历史内容类别", options, default=":material/image:图片检测历史")
    with col2:
        option_year = st.selectbox(
            "年份筛选",
            tuple(set([_[1].split('-')[0] for _ in show_info])),
        )
    with col3:
        option_month = st.selectbox(
            "月份筛选",
            tuple(set([_[1].split('-')[1] for _ in show_info])),
        )
    if selection == ':material/image:图片检测历史':
        col4, col5, col6, col7, col8 = st.columns([2, 2, 2, 2, 2])
        detail = [_ for _ in show_info if 'image' in _[0]]
        numbers = list(range(len(detail)))
        result = [[numbers[i] for i in range(j, len(detail), 5)] for j in range(5)]
        for inx, item in enumerate(detail):
            if inx in result[0]:
                with col4:
                    st.image(item[0])
                    st.markdown("<p style='text-align: center'>检测时间: {}</p>".format(item[1]), unsafe_allow_html=True)
            elif inx in result[1]:
                with col5:
                    st.image(item[0])
                    st.markdown("<p style='text-align: center'>检测时间: {}</p>".format(item[1]), unsafe_allow_html=True)
            elif inx in result[2]:
                with col6:
                    st.image(item[0])
                    st.markdown("<p style='text-align: center'>检测时间: {}</p>".format(item[1]), unsafe_allow_html=True)
            elif inx in result[3]:
                with col7:
                    st.image(item[0])
                    st.markdown("<p style='text-align: center'>检测时间: {}</p>".format(item[1]), unsafe_allow_html=True)
            elif inx in result[4]:
                with col8:
                    st.image(item[0])
                    st.markdown("<p style='text-align: center'>检测时间: {}</p>".format(item[1]), unsafe_allow_html=True)
    elif selection == ':material/movie:视频检测历史':
        col4, col5 = st.columns([5, 5])
        detail = [_ for _ in show_info if 'video' in _[0]]
        for inx, item in enumerate(detail):
            if inx % 2 == 0:
                with col4:
                    st.video(item[0])
                    st.markdown("<p style='text-align: center'>检测时间: {}</p>".format(item[1]), unsafe_allow_html=True)
            else:
                with col5:
                    st.video(item[0])
                    st.markdown("<p style='text-align: center'>检测时间: {}</p>".format(item[1]), unsafe_allow_html=True)

elif selected == '计数可视化':
    st.subheader("计数可视化")
    # 假设我们有一个函数 get_detection_counts() 返回检测时间和检测个数
    # 这里我们需要模拟或从实际检测中获取这些数据
    detection_times = []  # 这里应该是实际的检测时间数据
    detection_counts = []  # 这里应该是实际的检测个数数据

    # 绘制折线图
    fig, ax = plt.subplots()
    ax.plot(detection_times, detection_counts, marker='o')
    ax.set_xlabel('检测时间')
    ax.set_ylabel('检测个数')
    ax.set_title('检测个数随时间变化图')
    st.pyplot(fig)
