import cv2
import numpy as np
import argparse
import os
import base64
from io import BytesIO
from multiprocessing import Process
import subprocess
import time
from PIL import Image
import connect_server as cn

FOREGROUND_IMG = 'foreground.jpg'
BACKGROUND_IMG = 'background.jpg'


def blur_color_img(img, kernel_width=5, kernel_height=5, sigma_x=2, sigma_y=2):
    img = np.copy(img)  # we don't modify the original image
    img[:, :, 0] = cv2.GaussianBlur(img[:, :, 0], ksize=(kernel_width, kernel_height), sigmaX=sigma_x, sigmaY=sigma_y)
    img[:, :, 1] = cv2.GaussianBlur(img[:, :, 1], ksize=(kernel_width, kernel_height), sigmaX=sigma_x, sigmaY=sigma_y)
    img[:, :, 2] = cv2.GaussianBlur(img[:, :, 2], ksize=(kernel_width, kernel_height), sigmaX=sigma_x, sigmaY=sigma_y)
    return img


def background_subtraction(fg_img, bg_img, diff_threshold=30):
    fg_img = blur_color_img(fg_img, 5, 5, 4, 4)
    bg_img = blur_color_img(bg_img, 5, 5, 4, 4)
    mask = fg_img - bg_img
    mask = np.abs(mask)
    mask = np.mean(mask, axis=2, keepdims=False)
    mask[mask < diff_threshold] = 0
    mask[mask >= diff_threshold] = 255
    mask = mask.astype(np.uint8)
    mask = cv2.medianBlur(mask, 5)
    return mask

def main(foreground_img, background_img):
    fg_img = cv2.imread(foreground_img)  # [h, w, 3]
    bg_img = cv2.imread(background_img)  # [h, w, 3]
    mask = background_subtraction(fg_img, bg_img)
    unique, counts = np.unique(mask, return_counts=True)
    ls_counts = dict(zip(unique, counts))
    new_fg = np.zeros(
        [fg_img.shape[0], fg_img.shape[1], 4])  # png image --> has 4-dims instead of 3-dims like color image
    new_fg[:, :, :3] = fg_img
    new_fg[:, :, 3] = mask
    cv2.imwrite('mask.jpg', mask)
    cv2.imwrite('captain_america.png', new_fg)

def send_to_server(host, port, token, title, base64_img):
    try:
        cn.send_photo_to_server(host, port, token, title, "data:image/png;base64," + base64_img)
    except:
        try:
            cn.send_photo_to_server(host, port, token, title, "data:image/png;base64," + base64_img)
        except:
            try:
                print("trying save file to local!")
                img_data = base64.b64decode(base64_img)
                file_name = title+".png"
                f = open(file_name, "wb")
                f.write(img_data)
                f.close()
            except:

                print("error! write file error")
                pass
            pass
        pass
    pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('-v', '--video-path', type=str, help='Duong dan video file')
    parser.add_argument('-vo', '--out-video-path', type=str, default='./out_video.avi', help='Duong dan video xuat ra')
    parser.add_argument('-rsv', '--remote-server', type=str, default="127.0.0.1", help='Server quan ly thong tin')
    parser.add_argument('-p', '--port', type=str, default="3000", help='Server quan ly cua camera tren server')
    parser.add_argument('-tk', '--token', type=str, help='Token quan ly cua camera tren server')
    parser.add_argument('-sen', '--sensitive', type=float, default=0.3,
                        help='Do nhay phat hien chuyen dong. Gia Tri khoang (0:1)')
    parser.add_argument('-st', '--steptime', type=float, default=5.0, help= "Khoang thoi gian giua 2 lan gui data len server")

    FLAGS, unparsed = parser.parse_known_args()

    sensitive = FLAGS.sensitive

    # time to set time steps
    root_time = time.time()

    if FLAGS.video_path:
        try:
            os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;udp"
            vid = cv2.VideoCapture(FLAGS.video_path)
            height, width = None, None
            writer = None
        except:
            pass

        finally:
            grabbed, bg_img = vid.read()
            while True:
                grabbed, fg_img = vid.read()
                if not grabbed:
                    break

                if width is None or height is None:
                    height, width = fg_img.shape[:2]

                mask = background_subtraction(fg_img, bg_img)

                if writer is None:
                    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
                    writer = cv2.VideoWriter(FLAGS.video_output_path, fourcc, 30, (mask.shape[1], mask.shape[0]), True)
                    pass

                writer.write(mask)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

                pass

            vid.release()
            pass

        pass
    else:
        count = 0
        vid = cv2.VideoCapture(0)
        _, bg_img = vid.read()
        height, width = bg_img.shape[:2]
        fg_img = bg_img
        mask = None
        while True:
            if (count == 0):
                _, bg_img = vid.read()
                count += 1
                mask = background_subtraction(fg_img, bg_img, 90)
            else:
                _, fg_img = vid.read()
                mask = background_subtraction(fg_img, bg_img, 90)
                bg_img = fg_img
                count = (count + 1) % 30

            try:
                unique, counts = np.unique(mask, return_counts=True)
                ls_counts = dict(zip(unique, counts))
                percent = ls_counts[255] / (ls_counts[0] + ls_counts[255])
                pass
            except:
                percent = 0
                pass
            finally:
                pass

            cv2.putText(fg_img, str(percent), (40, 40), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 0, 255), 2)
            if (percent > sensitive):
                cv2.putText(fg_img, "Chuyen Dong", (140, 10),
                            cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 0, 0), 2)

                # convert numpy arr to Img and convert to base64 and revert color
                out_img = Image.fromarray(cv2.cvtColor(fg_img, cv2.COLOR_BGR2RGB))
                buffered = BytesIO()
                out_img.save(buffered, format="PNG")
                node_time = time.time()
                # if step time send data to server
                if (node_time - root_time > FLAGS.steptime):
                    str_img = base64.b64encode(buffered.getvalue()).decode()
                    proc = Process(target=send_to_server, args=(FLAGS.remote_server, FLAGS.port, FLAGS.token, "IMG_" +
                                                                str(node_time), base64.b64encode(buffered.getvalue()).decode()))
                    proc.start()
                    # cn.send_photo_to_server(FLAGS.remote_server, FLAGS.port, FLAGS.token, "IMG_" + str(node_time),
                    #                         "data:image/png;base64," + base64.b64encode(buffered.getvalue()).decode())
                    root_time = node_time

            cv2.imshow("FastWarning", fg_img)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    vid.release()
    cv2.destroyAllWindows()
