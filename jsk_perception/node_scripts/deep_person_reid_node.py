#!/usr/bin/env python

from jsk_topic_tools import ConnectionBasedTransport
import numpy as np
import rospy
import message_filters
from sensor_msgs.msg import Image
from opencv_apps.msg import RectArrayStamped
import cv2
from cv_bridge import CvBridge

from torchreid_libs.extractor import ReIDFeatureExtractor
import torch
cos = torch.nn.CosineSimilarity(dim=1)

class DeepPersonReIDNode(ConnectionBasedTransport):

    def __init__(self):
        super(DeepPersonReIDNode, self).__init__()
        self.model = rospy.get_param("~model", 'resnet50')
        self.checkpoint_file = rospy.get_param("~checkpoint_file", None)
        gpu = rospy.get_param('~gpu', -1)

        self._extractor = ReIDFeatureExtractor(
            model=self.model,
            ckpt_file=self.checkpoint_file,
            gpu=gpu,
            image_size=(256, 128),
            pixel_mean=[0.485, 0.456, 0.406],
            pixel_std=[0.229, 0.224, 0.225],
            pixel_norm=True
        )

        self.bridge = CvBridge()
        self.rect_pub = self.advertise('~output', RectArrayStamped, queue_size=1)
        self.template = self._extractor(cv2.imread('/tmp/person.jpg'))

    def subscribe(self):
        self.image_sub = message_filters.Subscriber('~input/image', Image)
        self.roi_sub = message_filters.Subscriber('~input/roi', RectArrayStamped)
        self.subs = [self.image_sub, self.roi_sub]
        queue_size = rospy.get_param('~queue_size', 100)
        if rospy.get_param('~approximate_sync', True):
            slop = rospy.get_param('~slop', 1.0)
            self.ts = message_filters.ApproximateTimeSynchronizer(
                self.subs, queue_size, slop, allow_headerless=True)
        else:
            self.ts = message_filters.TimeSynchronizer(
                self.subs, queue_size=queue_size)
        self.ts.registerCallback(self._callback)
        rospy.loginfo("Waiting for {} and {}".format(self.image_sub.name, self.roi_sub.name))
        
    def unsubscribe(self):
        for sub in self.subs:
            sub.unregister()

    def _callback(self, image, roi):
        img = self.bridge.imgmsg_to_cv2(image, desired_encoding='bgr8')
        for rect in roi.rects:
            try:
                cx = rect.x
                cy = rect.y
                w =  rect.width
                h =  rect.height
            except Exception as e:
                rospy.logerr(e)
                return
            image_roi_slice = np.index_exp[int(cy - h / 2):int(cy + h / 2),
                                           int(cx - w / 2):int(cx + w / 2)]
            feature = self._extractor(img[image_roi_slice])
            print(cos(self.template, feature))
        rect = RectArrayStamped(header=image.header)
        self.rect_pub.publish(rect)

if __name__ == '__main__':
    rospy.init_node('deep_person_reid_node')
    node = DeepPersonReIDNode()
    rospy.spin()
