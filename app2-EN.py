import warnings

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
import base64
import  matplotlib.pyplot as plt

if not os.path.exists('history'):
    os.makedirs('history')
model_map = {
    'FPH-CDet': 'best.pt',
    'yolov7-tiny': 'runs/detect/train2/weights/best.pt',
}

st.set_page_config(page_title=" Online Testing Platform", page_icon=":desktop_computer:")


def sidebar_bg(bg):
    side_bg_ext = "png"
    main_bg_ext = "png"

    st.markdown(
        f""" 
      <style>
      [data-testid="stSidebar"] > div:first-child {{
          background-size: cover;
          background: url(data:image/{side_bg_ext};base64,{base64.b64encode(open(bg, "rb").read()).decode()});
      }}
      .stApp {{
          background: url(data:image/{main_bg_ext};base64,{base64.b64encode(open(bg, "rb").read()).decode()});
          background-size: cover
      }}
      </style>
      """,
        unsafe_allow_html=True,
    )


# 调用
sidebar_bg('./back.jpg')

st.markdown(
    "<h1 style='text-align: center'><img width='50' src='https://picture.gptkong.com/20241114/1411fce2b0fd9942d582bd85472990b639.jpg'></img> Online Testing Platform</h1>",
    unsafe_allow_html=True)
with st.sidebar:
    col1, col2 = st.columns([3, 7])
    with col1:
        st.image('logo.jpg')
    model_select = st.selectbox("Model selection for testing", list(model_map.keys()))
    if model_select not in st.session_state:
        st.session_state[model_select] = YOLO(model_map[model_select])
    selected = option_menu("Feature list", ["Home", "Image Detection", "Video Detection", "Camera Detection", "Visual Counting", "History Detection"],
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
        with st.spinner("Waitting..."):
            video_count_lis = []
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            if 29 <= fps <= 31:
                fps = 30
            elif 59 <= fps <= 61:
                fps = 60
            else:
                fps = int(fps)
            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            # fps = cap.get(cv2.CAP_PROP_FPS)

            # 设置输出视频文件
            output_path = "predict_video.mp4"
            fourcc = cv2.VideoWriter_fourcc(*'X264')
            out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

            frame_rate_divider = 1  # 每1帧进行一次检测
            frame_count = 0
            counts = defaultdict(int)
            object_str = ""
            index = 0

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

                # 写入输出视频
                out.write(frame)
                frame_count += 1
                video_count_lis.append(result.boxes.xyxy.cpu().numpy().shape[0])

            # 关闭文件
            cap.release()
            out.release()
            # video_file = open(output_path, "rb")
            # video_bytes = video_file.read()
            # st.video(video_bytes)
            return video_count_lis, fps
    else:
        st.error("Video detection failed", icon='🚨')


if selected == 'Home':
    st.subheader("This platform is based on deep learning algorithms to detect citrus in three types of inputs: images, videos, and cameras. The image and video detection functions support file uploads or URL input for detection. The results of each detection can be viewed in the history detection section.")

elif selected == 'Image Detection':
    options = [":material/image:Reading image", ":material/link:Reading URL"]
    selection = st.pills("Select the method to read", options, default=":material/image:Reading image")
    if selection == options[0]:
        uploaded_file = st.file_uploader("Upload the image to be detected", type=["jpg", "jpeg", "png"])
        start_btn = st.button("Detection")
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
                    st.markdown("<h4 style='text-align: center'>Original image</h4>", unsafe_allow_html=True)
                    st.image('predict_img.jpg')
                with col2:
                    st.markdown("<h4 style='text-align: center'>Result image</h4>", unsafe_allow_html=True)
                    st.image(frame_rgb, channels="RGB")
                    st.markdown("<h4 style='text-align: center'>Detection count: {}</h4>".format(count_num),
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
                st.warning("No file uploaded")
                st.stop()
    elif selection == options[1]:
        img_url = st.text_input("Enter the image URL.")
        if st.button("Detection"):
            try:
                response = requests.get(img_url, stream=True)
                response.raise_for_status()

                # 将图片数据写入本地文件
                with open('predict_img.jpg', "wb") as file:
                    for chunk in response.iter_content(1024):
                        file.write(chunk)
                file.close()
                st.success("Image successfully read", icon='✅')
            except requests.exceptions.RequestException as e:
                st.error("Error reading the image", icon='🚨')
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
                st.markdown("<h4 style='text-align: center'>Number of detections: {}</h4>".format(count_num), unsafe_allow_html=True)
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

elif selected == 'Video detection':
    options = [":material/movie:Reading video", ":material/link:Reading URL"]
    selection = st.pills("Select the method to read", options, default=":material/movie:Reading video")
    if selection == options[0]:
        uploaded_file = st.file_uploader("Upload the video file", type=["mp4", "mov"])
        start_btn = st.button("Detection")
        col1, col2 = st.columns([5, 5])
        if start_btn:
            if uploaded_file is not None:
                temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
                temp_file.write(uploaded_file.read())
                video_path = temp_file.name
                count_res, fps= predict_video(video_path)
                with col1:
                    st.video(video_path, autoplay=True, muted=True)
                with col2:
                    st.video(open("predict_video.mp4", "rb").read(), autoplay=True, muted=True)
                timestamp = int(time.time())
                shutil.copy('./predict_video.mp4', 'history/{}_video.mp4'.format(timestamp))
                with open("history.txt", mode='a+', encoding='utf8') as f:
                    f.write("{}\t{}\t{}\t{}\n".format('history/{}_video.mp4'.format(timestamp),
                                              datetime.now().strftime('%Y-%m-%d'), ','.join([str(_) for _ in count_res]), fps))
                f.close()
                with open("predict_video.mp4", "rb") as file:
                    btn = st.download_button(
                        label="Download result",
                        data=file,
                        file_name="{}_video_file_predict.mp4".format(int(time.time())),
                        mime="video/mp4",
                    )
            else:
                st.warning("No file uploaded")
                st.stop()
    elif selection == options[1]:
        video_url = st.text_input("Please enter the image URL")
        if st.button("Detection"):
            try:
                # 下载视频到本地临时文件
                temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
                urllib.request.urlretrieve(video_url, temp_file.name)
                video_path = temp_file.name
                st.success("Video successfully read", icon='✅')
            except:
                st.error("Error reading the video", icon='🚨')
                st.stop()
            count_res, fps = predict_video(video_path)
            col1, col2 = st.columns([5, 5])
            with col1:
                st.video(video_path, autoplay=True, muted=True)
            with col2:
                st.video(open("predict_video.mp4", "rb").read(), autoplay=True, muted=True)
            timestamp = int(time.time())
            shutil.copy('./predict_video.mp4', 'history/{}_video.mp4'.format(timestamp))
            with open("history.txt", mode='a+', encoding='utf8') as f:
                f.write("{}\t{}\t{}\t{}\n".format('history/{}_video.mp4'.format(timestamp),
                                          datetime.now().strftime('%Y-%m-%d'), ','.join([str(_) for _ in count_res]), fps))
            f.close()
            with open("predict_video.mp4", "rb") as file:
                btn = st.download_button(
                    label="Download result",
                    data=file,
                    file_name="{}_video_url_predict.mp4".format(int(time.time())),
                    mime="video/mp4",
                )

elif selected == 'Camera Detection':
    run = st.toggle("Activate camera")
    with open("tmp.txt", mode='a+', encoding='utf-8') as f:
        f.write('\n{}\t'.format(datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
    f.close()
    frame_placeholder = st.empty()
    count_placeholder = st.empty()
    lis = []
    count_num = 0

    # 捕获摄像头实时流
    cap = cv2.VideoCapture(0)  # 0代表默认摄像头

    # 运行实时检测循环
    while run:
        ret, frame = cap.read()
        if not ret:
            st.error("Unable to read camera input", icon='🚨')
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

        count_placeholder.markdown("<h4 style='text-align: center'>Number of detections: {}</h4>".format(result.boxes.xyxy.cpu().numpy().shape[0]), unsafe_allow_html=True)

        count_num += 1
        # if count_num % 10 == 0:
        with open("tmp.txt", mode='a+', encoding='utf-8') as f:
            f.write('{},'.format(result.boxes.xyxy.cpu().numpy().shape[0]))
        f.close()
        # 控制刷新频率，每10ms刷新一次
        cv2.waitKey(10)

    # 停止摄像头
    cap.release()
    with open("tmp.txt", mode='r', encoding='utf-8') as f:
        tmp = f.readlines()
    f.close()
    timestamp = int(time.time())
    with open("history.txt", mode='a+', encoding='utf8') as f:
        for item in tmp:
            if len(item.strip().split('\t')) == 2:
                f.write("{}\t{}".format('{}_cap'.format(timestamp), item))
    f.close()
    with open("tmp.txt", mode='w', encoding='utf-8') as f:
        f.write('')
    f.close()

elif selected == 'History Detection':
    with open("history.txt", "r", encoding='utf8') as f:
        res = f.readlines()
    f.close()
    show_info = [_.split('\t') for _ in res]
    col1, col2, col3 = st.columns([3, 3, 3])
    with col1:
        options = [":material/image:Image detection history", ":material/movie:Video detection history"]
        selection = st.pills("select the category of history content to browse", options, default=":material/image:Image detection history")
    with col2:
        option_year = st.selectbox(
            "Year filter",
            tuple(set([_[1].split('-')[0] for _ in show_info])),
        )
    with col3:
        option_month = st.selectbox(
            "Month filter",
            tuple(set([_[1].split('-')[1] for _ in show_info])),
        )
    if selection == ':material/image:Image detection history':
        col4, col5, col6, col7, col8 = st.columns([2, 2, 2, 2, 2])
        detail = [_ for _ in show_info if 'image' in _[0]]
        numbers = list(range(len(detail)))
        result = [[numbers[i] for i in range(j, len(detail), 5)] for j in range(5)]
        for inx, item in enumerate(detail):
            if inx in result[0]:
                with col4:
                    st.image(item[0])
                    st.markdown("<p style='text-align: center'>Detection time: {}</p>".format(item[1]), unsafe_allow_html=True)
            elif inx in result[1]:
                with col5:
                    st.image(item[0])
                    st.markdown("<p style='text-align: center'>Detection time: {}</p>".format(item[1]), unsafe_allow_html=True)
            elif inx in result[2]:
                with col6:
                    st.image(item[0])
                    st.markdown("<p style='text-align: center'>Detection time: {}</p>".format(item[1]), unsafe_allow_html=True)
            elif inx in result[3]:
                with col7:
                    st.image(item[0])
                    st.markdown("<p style='text-align: center'>Detection time: {}</p>".format(item[1]), unsafe_allow_html=True)
            elif inx in result[4]:
                with col8:
                    st.image(item[0])
                    st.markdown("<p style='text-align: center'>Detection time: {}</p>".format(item[1]), unsafe_allow_html=True)
    elif selection == ':material/movie:Video detection history':
        col4, col5 = st.columns([5, 5])
        detail = [_ for _ in show_info if 'video' in _[0]]
        for inx, item in enumerate(detail):
            if inx % 2 == 0:
                with col4:
                    st.video(item[0])
                    st.markdown("<p style='text-align: center'>Detection time: {}</p>".format(item[1]), unsafe_allow_html=True)
            else:
                with col5:
                    st.video(item[0])
                    st.markdown("<p style='text-align: center'>Detection time: {}</p>".format(item[1]), unsafe_allow_html=True)


elif selected == 'Visual Counting':
    with open("history.txt", "r", encoding='utf8') as f:
        res = f.readlines()
    f.close()
    show_info = [_.split('\t') for _ in res]
    options = [":material/movie:Video visualization", ":material/image:Camera visualization"]
    selection = st.pills("Select the category of visualization to browse", options, default=":material/movie:Video visualization")
    if selection == ':material/movie:Video visualization':
        detail = [_ for _ in show_info if 'video' in _[0]]
        for item in detail:
            col4, col5 = st.columns([3.5, 6.5])
            with col4:
                st.video(item[0])
                st.markdown("<p style='text-align: center'>Detection time: {}</p>".format(item[1]), unsafe_allow_html=True)
            with col5:
                fig, ax = plt.subplots()
                data = [int(_) for _ in item[2].split(',')]
                ax.plot(data)
                ax.set_xlabel('Seconds')
                ax.set_ylabel('Count')

                num_points = len(data)
                seconds = num_points // int(item[3])  # 总秒数
                xticks_positions = range(0, num_points, int(item[3]))  # 每 30 个点一个刻度
                xticks_labels = range(0, seconds + 1)  # 标签从 0 秒开始

                # 确保 xticks 和 labels 长度一致（防止数据点不是 30 的整数倍）
                if len(xticks_positions) > len(xticks_labels):
                    xticks_positions = xticks_positions[:len(xticks_labels)]
                elif len(xticks_positions) < len(xticks_labels):
                    xticks_labels = xticks_labels[:len(xticks_positions)]

                ax.set_xticks(xticks_positions)
                ax.set_xticklabels(xticks_labels)
                st.pyplot(fig)
    elif selection == ':material/image:Camera visualization':
        detail = [_ for _ in show_info if 'cap' in _[0]]
        for item in detail:
            fig, ax = plt.subplots(figsize=(10, 4))
            data = [int(_) for _ in item[2].strip().split(',') if _]
            ax.plot(data)
            ax.set_title(item[1])
            ax.set_xlabel('Seconds')
            ax.set_ylabel('Count')
            # 计算 x 轴刻度（每 10 个点代表 1 秒）
            num_points = len(data)
            seconds = num_points // 10  # 总秒数
            xticks_positions = range(0, num_points, 10)  # 每 10 个点一个刻度
            xticks_labels = range(0, seconds + 1)  # 标签从 0 秒开始

            # 确保 xticks 和 labels 长度一致（防止数据点不是 10 的整数倍）
            if len(xticks_positions) > len(xticks_labels):
                xticks_positions = xticks_positions[:len(xticks_labels)]
            elif len(xticks_positions) < len(xticks_labels):
                xticks_labels = xticks_labels[:len(xticks_positions)]

            ax.set_xticks(xticks_positions)
            ax.set_xticklabels(xticks_labels)
            st.pyplot(fig)
