from pickletools import float8
import numpy as np
from vision.ssd.vgg_ssd import create_vgg_ssd, create_vgg_ssd_predictor
from vision.ssd.mobilenetv1_ssd import create_mobilenetv1_ssd, create_mobilenetv1_ssd_predictor
from vision.ssd.mobilenetv1_ssd_lite import create_mobilenetv1_ssd_lite, create_mobilenetv1_ssd_lite_predictor
from vision.ssd.squeezenet_ssd_lite import create_squeezenet_ssd_lite, create_squeezenet_ssd_lite_predictor
from vision.ssd.mobilenet_v2_ssd_lite import create_mobilenetv2_ssd_lite, create_mobilenetv2_ssd_lite_predictor
from vision.utils.misc import Timer
import cv2
import sys
import argparse

parser = argparse.ArgumentParser(description='SSD Detector With PyTorch')

parser.add_argument("--net-type", type=str, default="")
parser.add_argument('--model', default='mb1-ssd')
parser.add_argument('--label', type=str, default="")
parser.add_argument('--video', type=str, default="")
parser.add_argument('--output', type=str, default="")
parser.add_argument('--threshold', type=float, default=0.5)

args = parser.parse_args()

if len(sys.argv) < 4:
    print('Usage: python run_ssd_example.py <net type> <model path> <label path> [video file] [output file]')
    sys.exit(0)
net_type = args.net_type
model_path = args.model
label_path = args.label

if not args.video == "":
    cap = cv2.VideoCapture(args.video)  # capture from file
    # cap.set(cv2.CAP_PROP_FPS, 30)
else:
    cap = cv2.VideoCapture(0)   # capture from camera
    cap.set(3, 1280)
    cap.set(4, 720)


if not args.output == "":
    output_file = args.output
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    out = cv2.VideoWriter(output_file, fourcc, 10.0 if args.video == "" else cap.get(cv2.cv.CV_CAP_PROP_FPS), (frame_width,frame_height))
    # print(output_file.split("/")[-1])

class_names = [name.strip() for name in open(label_path).readlines()]
num_classes = len(class_names)
# print(output_file)

if net_type == 'vgg16-ssd':
    net = create_vgg_ssd(len(class_names), is_test=True)
elif net_type == 'mb1-ssd':
    net = create_mobilenetv1_ssd(len(class_names), is_test=True)
elif net_type == 'mb1-ssd-lite':
    net = create_mobilenetv1_ssd_lite(len(class_names), is_test=True)
elif net_type == 'mb2-ssd-lite':
    net = create_mobilenetv2_ssd_lite(len(class_names), is_test=True)
else:
    print("The net type is wrong. It should be one of vgg16-ssd, mb1-ssd and mb1-ssd-lite.")
    sys.exit(1)
net.load(model_path)


if net_type == 'vgg16-ssd':
    predictor = create_vgg_ssd_predictor(net, candidate_size=200)
elif net_type == 'mb1-ssd':
    predictor = create_mobilenetv1_ssd_predictor(net, candidate_size=200)
elif net_type == 'mb1-ssd-lite':
    predictor = create_mobilenetv1_ssd_lite_predictor(net, candidate_size=200)
elif net_type == 'mb2-ssd-lite' or net_type == "mb3-large-ssd-lite" or net_type == "mb3-small-ssd-lite":
    predictor = create_mobilenetv2_ssd_lite_predictor(net, candidate_size=200)
else:
    print("The net type is wrong. It should be one of vgg16-ssd, mb1-ssd and mb1-ssd-lite.")
    sys.exit(1)

timer = Timer()
while True:
    ret, orig_image = cap.read()
    if ret == True: 
        # orig_image = cv2.resize(orig_image,(540,960),fx=0,fy=0, interpolation = cv2.INTER_CUBIC)
        if orig_image is None:
            continue
        image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)
        timer.start()
        boxes, labels, probs = predictor.predict(image, 10, args.threshold)
        interval = timer.end()
        print('Time: {:.2f}s, Detect Objects: {:d}.'.format(interval, labels.size(0)))
        for i in range(boxes.size(0)):
            box = np.int16(boxes[i, :])
            print(box)
            # print(labels)
            label = f"{class_names[labels[i]]}: {100*probs[i]:.2f}%"
            print(label)
            cv2.rectangle(orig_image, (box[0], box[1]), (box[2], box[3]), (255, 255, 0), 4)

            cv2.putText(orig_image, label,
                        (box[0]+20, box[1]+40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,  # font scale
                        (255, 0, 255),
                        2)  # line type
        if not args.output == "":
            out.write(orig_image)
        cv2.imshow('annotated', orig_image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

cap.release()

if not args.output == "":
    out.release()
cv2.destroyAllWindows()

