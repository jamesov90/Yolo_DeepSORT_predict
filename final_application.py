import time
from absl import app, flags, logging
from absl.flags import FLAGS
import cv2
import tensorflow as tf
import numpy as np
from yolov3_tf2.models import (
    YoloV3, YoloV3Tiny
)
from yolov3_tf2.dataset import transform_images
from yolov3_tf2.utils import draw_bbox
from yolov3_tf2.utils import read_class_names

from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from deep_sort import generate_detections as gdet

from prediction import predict_trajectory
from pynput import keyboard


Track_only = ["person"]
K_bboxes_filter = 30 #constant to filter doble detected boxes in yolov3

flags.DEFINE_string('classes', './data/coco.names', 'path to classes file')
flags.DEFINE_string('weights', './checkpoints/yolov3.tf',
                    'path to weights file')
flags.DEFINE_boolean('tiny', False, 'yolov3 or yolov3-tiny') #tiny only for CPU
flags.DEFINE_integer('size', 416, 'resize images to')
flags.DEFINE_string('video', './test.mp4',
                    'path to video file or number for webcam)')
flags.DEFINE_string('output', None, 'path to output video')
flags.DEFINE_string('output_format', 'XVID', 'codec used in VideoWriter when saving video to file')
flags.DEFINE_integer('num_classes', 80, 'number of classes in the model')

flags.DEFINE_boolean('show', False, "show video or do not show video")

test_action_ratio = True #boolean that reduce the action radio of the observer


def press_key(key):
	if key == keyboard.KeyCode.from_char('q'):
		return False

def main(_argv):
    if FLAGS.show == False:
        listener = keyboard.Listener(press_key)
        listener.start()
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    for physical_device in physical_devices:
        tf.config.experimental.set_memory_growth(physical_device, True)

    if FLAGS.tiny:
        yolo = YoloV3Tiny(classes=FLAGS.num_classes)
    else:
        yolo = YoloV3(classes=FLAGS.num_classes)

    yolo.load_weights(FLAGS.weights)
    logging.info('weights loaded')

    class_names = [c.strip() for c in open(FLAGS.classes).readlines()]
    logging.info('classes loaded')

    # Definition of the parameters for deepsort
    max_euclidean_distance = 0.7
    nn_budget = None

    # initialize deep sort object
    model_filename = 'model_data/coco/mars-small128.pb'
    encoder = gdet.create_box_encoder(model_filename, batch_size=1)
    metric = nn_matching.NearestNeighborDistanceMetric("euclidean", max_euclidean_distance, nn_budget)
    tracker = Tracker(metric)

    times = []

    try:
        vid = cv2.VideoCapture(int(FLAGS.video))
    except:
        vid = cv2.VideoCapture(FLAGS.video)

    width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if FLAGS.output:
        # by default VideoCapture returns float instead of int
        fps = int(vid.get(cv2.CAP_PROP_FPS))
        codec = cv2.VideoWriter_fourcc(*FLAGS.output_format)
        out = cv2.VideoWriter(FLAGS.output, codec, fps, (width, height))

    NUM_CLASS = read_class_names(FLAGS.classes)
    key_list = list(NUM_CLASS.keys())
    val_list = list(NUM_CLASS.values())

    prediction = predict_trajectory.Prediction()
    count_frame = 0
    while True:
        _, img = vid.read()

        if img is None:
            logging.warning("Empty Frame")
            time.sleep(0.1)
            break
            #continue

        img_in = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_in = tf.expand_dims(img_in, 0)
        img_in = transform_images(img_in, FLAGS.size)

        t1 = time.time()
        global_boxes, global_scores, global_classes, global_nums = yolo.predict(img_in)

        boxes, scores, names = [], [], []
        bboxes, objectness, classes, nums = global_boxes[0], global_scores[0], global_classes[0], global_nums[0]
        wh = np.flip(img.shape[0:2])
        bboxes_centers_filtering = []  # this array will help to filter glitches from yolo taht identifies an object twice


        if test_action_ratio == True:
            min_x = int(0)
            min_y = int(0)
            max_x = int(width/2)
            max_y = int(height/2)
            cv2.rectangle(img, (min_x, min_y), (max_x, max_y), color=(0,255,0), thickness=3)
        else:
            min_x = int(0)
            min_y = int(0)
            max_x = int(width)
            max_y = int(height)


        for i in range(nums):
            double_bbox = 0
            bbox_detectable = 0
            if len(Track_only) != 0 and class_names[int(global_classes[0][i])] in Track_only or len(Track_only) == 0:
                x1y1 = tuple((np.array(bboxes[i][0:2]) * wh).astype(np.int32))
                x2y2 = tuple((np.array(bboxes[i][2:4]) * wh).astype(np.int32))
                # filter repeated boxes
                for bbox_center in bboxes_centers_filtering:
                    if (bbox_center[0]<=(x1y1[0] + (x2y2[0] - x1y1[0])/2 + K_bboxes_filter)
                        and bbox_center[0]>=(x1y1[0] + (x2y2[0] - x1y1[0])/2 - K_bboxes_filter)
                        and bbox_center[1]<=(x1y1[1] + (x2y2[1] - x1y1[1])/2 + K_bboxes_filter)
                        and bbox_center[1]>=(x1y1[1] + (x2y2[1] - x1y1[1])/2 - K_bboxes_filter)):
                        double_bbox = 1
                #box is detectable only if is inside the observation ratio
                if (    (min_x < x1y1[0] and x1y1[0] < max_x and min_y < x1y1[1] and x1y1[1] < max_y)
                    or  (min_x < x2y2[0] and x2y2[0] < max_x and min_y < x1y1[1] and x1y1[1] < max_y)
                    or  (min_x < x1y1[0] and x1y1[0] < max_x and min_y < x2y2[1] and x2y2[1] < max_y)
                    or  (min_x < x2y2[0] and x2y2[0] < max_x and min_y < x2y2[1] and x2y2[1] < max_y)):
                    bbox_detectable = 1
                if double_bbox == 0 and bbox_detectable == 1:
                    boxes.append([x1y1[0], x1y1[1], x2y2[0] - x1y1[0],x2y2[1] - x1y1[1]])
                    scores.append(objectness[i])
                    names.append(class_names[int(global_classes[0][i])])
                    bboxes_centers_filtering.append([x1y1[0] + (x2y2[0] - x1y1[0])/2,x1y1[1] + (x2y2[1] - x1y1[1])/2])
                    #cv2.circle(img, (int(x2y2[0] - x1y1[0]), int(x2y2[1] - x1y1[1])), radius=2, color=(0, 255, 255), thickness=3)
                #img = cv2.rectangle(img, x1y1, x2y2, (255, 0, 0), 2)

        # Obtain all the detections for the given frame.
        boxes = np.array(boxes)
        names = np.array(names)
        scores = np.array(scores)
        features = np.array(encoder(img, boxes))
        detections = [Detection(bbox, score, class_name, feature) for bbox, score, class_name, feature in
                      zip(boxes, scores, names, features)]

        # Pass detections to the deepsort object and obtain the track information.
        tracker.predict()
        tracker.update(detections)

        # Obtain info from the tracks
        tracked_bboxes = []
        x_location = []
        y_location = []
        object_id = []

        for track in tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 5:
                continue
            #print("track_id: " + str(track.track_id) + " state " + str(track.state))

            bbox = track.to_tlbr()  # Get the corrected/predicted bounding box
            class_name = track.get_class()  # Get the class name of particular object
            tracking_id = track.track_id  # Get the ID for the particular track
            index = key_list[val_list.index(class_name)]  # Get predicted object index by object name
            tracked_bboxes.append(bbox.tolist() + [tracking_id,
                                                   index])  # Structure data, that we could use it with our draw_bbox function
            # x_size = bbox.tolist()[2] - bbox.tolist()[0]
            # y_size = bbox.tolist()[3] - bbox.tolist()[1]
            x_location.append((bbox.tolist()[2] + bbox.tolist()[0]) / 2)
            y_location.append((bbox.tolist()[3] + bbox.tolist()[1]) / 2)
            object_id.append(tracking_id)

            #print("Class name " + class_name + str(tracking_id) + " detected at position x = " + str(x_location) + ", y = " + str(y_location))

        # draw detection on frame
        draw_bbox(img, tracked_bboxes, CLASSES=FLAGS.classes, tracking=True)

        prediction.predict_trajectory(img, object_id, x_location, y_location, height, width)


        t2 = time.time()
        times.append(t2 - t1)
        times = times[-20:]
        ms = sum(times) / len(times) * 1000
        fps = 1000 / ms
        cv2.putText(img, "Time: {:.1f} FPS".format(fps), (0, 30), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1,
                    (0, 0, 255), 2)

        if FLAGS.output:
            out.write(img)
        if FLAGS.show:
            cv2.imshow('output', img)
            if cv2.waitKey(1) == ord('q'):
                break
        else:
            print("Frame: ", count_frame, ", FPS: {:.1f}".format(fps))
            if(listener.is_alive() == False):
                break
        count_frame = count_frame + 1
    cv2.destroyAllWindows()


if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
