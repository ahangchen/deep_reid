import argparse
import json
import os

from sensor.api import ApiConfig
from utils.file_helper import read_lines
from utils.net_util import upload_file, post_json


def arg_parse():
    parser = argparse.ArgumentParser(description='Training')
    parser.add_argument('--box_dir', default='/home/cwh/coding/sysu_bbox', type=str, help='gpu_ids: e.g. 0  0,1,2  0,2')
    opt = parser.parse_args()
    return opt


def upload_detection_info(img_url, boxes, sensorId, capture_time_str):
    svr_conf = ApiConfig()
    detect_json = {
        "captureTime": capture_time_str,
        "fromSensorId": sensorId,
        "imgUrl": img_url,
        "boxes": []
    }
    for box in boxes:
        detect_json["boxes"].append(box)

    post_json(svr_conf.urls['upload_detect_info'], detect_json)


def main():
    opt = arg_parse()
    img_svr = ApiConfig().urls['img_svr'] + '/sysu/'
    for box_rec in sorted(os.listdir(opt.box_dir)):
        if box_rec.startswith('21'):break
        person_imgs = read_lines(os.path.join(opt.box_dir, box_rec))
        sensorId = int(box_rec[4])
        for person_img in person_imgs:
            if person_img.startswith('img_name'):
               continue
            capture_time_str = '2018-05-18 %s:%s:%s' % (person_img[0:2], person_img[2:4], person_img[4:6])
            img_url = img_svr + '18_c%d_' % sensorId + person_img[:12]
            boxes = json.loads(person_img[14:-2])
            upload_detection_info(img_url, [boxes], sensorId, capture_time_str)




if __name__ == '__main__':
    main()