import json
import requests

def send_photo_to_server(host, port, cam_token, title, photo):
    url = "http://" + host + ":" + port + "/api/cameras/" + cam_token + "/photos"
    json_photo = {"photo": {"title": title, "image": photo}}
    headers = {"Content-Type": "application/json"}
    r = requests.post(url=url, data=json.dumps(json_photo), headers=headers)
    return r
