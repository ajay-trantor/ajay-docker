import numpy as np
import onnxruntime as rt
import cv2 as cv
import json
import time
import sys

def hello():
    begin = time.time()
    kyc_dict = {0:"login_box",1:"field",2:"button"}
    im = cv.imread('1.png')
    im = im.astype(np.float32) / 255.0
    image = np.transpose(im, axes=(2, 0, 1))

    sess = rt.InferenceSession(r'klite.onnx')
    input_name = sess.get_inputs()[0].name
    output_name = sess.get_outputs()[0].name
    results = sess.run(output_names = None, input_feed = {input_name: image})
    print(results)
    class_names = []
    conf = results[0]
    classes = results[1]
    coords = results[2]

    for i,c in zip(classes,conf):
        if c>0.65:
           class_names.append(kyc_dict[i])

    to_json = {}
    for n,field, cord in zip(range(len(classes)),class_names,coords):
        to_json['bbox'+str(n+1)]= {'field_type':field,'coords':{'xmin': int(cord[0]), 'ymin': int(cord[1]), 'xmax': int(cord[2]),'ymax':int(cord[3])}}
    with open('/tmp/output_json.json','w') as f:
        json.dump(to_json,f)
        print('Created json!')
    end = time.time()
    print(begin-end)
    return True
