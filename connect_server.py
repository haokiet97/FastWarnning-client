import json
import requests

def send_photo_to_server(host, port, cam_token, title, photo):
    url = "http://" + host + ":" + port + "/api/cameras/" + cam_token + "/photos"
    json_photo = {"photo": {"title": title, "image": photo}}
    headers = {"Content-Type": "application/json"}
    r = requests.post(url=url, data=json.dumps(json_photo), headers=headers)
    return r

def send_video_to_server(host, port, cam_token, title, description, video):
    url = "http://" + host + ":" + port + "/api/cameras/" + cam_token + "/videos"
    json_video = {"video": {"title": title, "description": description, "data": video}}
    headers = {"Content-Type": "application/json"}
    r = requests.post(url=url, data=json.dumps(json_video), headers=headers)
    return r
