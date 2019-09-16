from onvif import ONVIFCamera, ONVIFService
import datetime
import io
from PIL import Image
import json
import os
import requests
from requests.auth import HTTPDigestAuth
import sys

import object_detector as OD

IP = "169.254.126.176"
size = 600, 339

def getFileName(head):
    #dt = mycam.devicemgmt.GetSystemDateAndTime()
    dt = datetime.datetime.now()
    year = dt.year
    month = dt.month
    date = dt.day
    hour = dt.hour
    minute = dt.minute
    second = dt.second
    microsecond = dt.microsecond
    filename = '%s-%s-%s-%s-%s-%s-%s-%s.jpg' % (year, month, date, hour, minute, second, microsecond, str(head))
    return filename

def getTimestamp():
    dt = datetime.datetime.now()
    year = dt.year
    month = dt.month
    date = dt.day
    hour = dt.hour
    minute = dt.minute
    second = dt.second
    timestamp = '%s:%s:%s %s-%s-%s' % (hour, minute, second, month, date, year)
    return timestamp;

mycam = ONVIFCamera(IP, 80, 'administrator', 'password')
#media = mycam.create_media_service()
#profiles = media.GetProfiles()
#print(profiles)

URIs = ['http://%s/media/cam0/still.jpg?res=max' % IP,
        'http://%s/media/cam1/still.jpg?res=max' % IP,
        'http://%s/media/cam2/still.jpg?res=max' % IP,
        'http://%s/media/cam3/still.jpg?res=max' % IP]

detection_graph = OD.loadModel("drone-detect/frozen_inference_graph.pb")
category_index = OD.loadLabelMap("drone-detect/label_map.pbtxt")

def saved_images():
    dirs = os.listdir("test_images/")
    for file in dirs:
        image = os.path.join("test_images", file)
        print("Checking %s" % file)
        image = Image.open(image)
        OD.detect(image, file, detection_graph, category_index)

def single_image():
    imgs = []        
    for i, uri in enumerate(URIs):
        r = requests.get(uri, auth=HTTPDigestAuth('administrator', 'password'), stream=True)
        filename = getFileName(i)
        im = Image.open(io.BytesIO(r.content))
        im.thumbnail(size)
        imgs.append((im, filename))
    detected = False;
    for i in range(4):
        if OD.detect(imgs[i][0], imgs[i][1], detection_graph, category_index):
            makeGeoJSON(imgs[i][0], imgs[i][1])
            return()

def continuous_images():
    while(1):
        imgs = []        
        for i, uri in enumerate(URIs):
            r = requests.get(uri, auth=HTTPDigestAuth('administrator', 'password'), stream=True)
            filename = getFileName(i)
            im = Image.open(io.BytesIO(r.content))
            im.thumbnail(size)
            imgs.append((im, filename))
        detected = False;
        for i in range(4):
            if OD.detect(imgs[i][0], imgs[i][1], detection_graph, category_index):
                makeGeoJSON(imgs[i][0], imgs[i][1])
                return

def makeGeoJSON(img, filename):  
    timestamp = getTimestamp();
    json_data = json.dumps({"type":"FeatureCollection","features":[{"type":"Feature","geometry":{"type":"Point","coordinates":[-84.40105167,33.77916167]},"properties":{"id":1,"updateWhen":timestamp,"url":'assets/images/drone/' + filename}}]}, separators=(',', ':'))
    with open('../../frontend/src/assets/drone', "w") as file:
        file.write(json_data);

if __name__ == "__main__":
    if len(sys.argv) == 1: #No parameters
        single_image()
    elif len(sys.argv) == 2 and sys.argv[1] == "stream":
        continuous_images()
    elif len(sys.argv) == 2 and sys.argv[1] == "test":
        saved_images()
    else:
        print("Wrong Usage. See README.md")