import time
import math
import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from rclpy.exceptions import ROSInterruptException
from rclpy.action import ActionClient
from nav2_msgs.action import NavigateToPose


class Robot(Node):
    def __init__(self):
        super().__init__('robot')

        print('RUNNING SIMPLE PROJECT_FILE WITH DEBUG PRINTS')

        # Publisher
        self.publisher = self.create_publisher(Twist, '/cmd_vel', 10)

        # Nav2 action client
        self.navigate_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')
        self.goal_handle = None
        self.goal_sent = False
        self.goal_done = True

        # Detection flags
        self.green_flag = 0
        self.blue_flag = 0
        self.red_flag = 0
        self.blue_visible_now = 0

        # Seen at least once
        self.green_seen = 0
        self.blue_seen = 0
        self.red_seen = 0

        # HSV sensitivity
        self.sensitivity = 10

        # Area thresholds
        self.follow_area = 800
        self.stop_area = 302000

        # Blue tracking
        self.blue_area = 0
        self.blue_cx = 0
        self.image_width = 640
        self.centre_margin = 60
        self.last_seen_turn_direction = 'left'

        # Simple blue confirmation
        self.blue_detect_count = 0
        self.blue_confirm_frames = 3

        # Lost blue handling
        self.blue_lost_count = 0
        self.blue_lost_threshold = 12

        # Stop confirmation
        self.stop_count = 0
        self.stop_confirm_frames = 4

        # Mode flags
        self.pursue_blue = False
        self.task_finished = False

        # Search poses
        self.search_positions = self.create_search_positions()
        self.search_index = 0

        # Debug window
        self.show_debug = True
        if self.show_debug:
            cv2.namedWindow('Detection', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('Detection', 640, 480)

        # Image subscriber
        self.bridge = CvBridge()
        self.subscription = self.create_subscription(
            Image, '/camera/image_raw', self.callback, 10
        )
        self.subscription

        # Debug trackers to reduce spam
        self.last_printed_mode = None
        self.last_printed_blue_flag = None
        self.last_printed_goal_sent = None
        self.last_printed_goal_done = None

    def create_search_positions(self):
        # REPLACE THESE WITH YOUR REAL TESTED COORDINATES
        return [
            [1.89, 1.09, 1.57],   # bottom-right starting patch, look upward
            [-2.82, -4.22, -1.57],   # centre-right opening, look left
            [4.71, -5.33, 3.14],   # lower-middle patch, look upward
            [-2.82, -4.22, -1.57],   # lower-left patch, look up-right
            [-6.84, -3.22, 0.0],    # upper-left patch, look right toward blue
            [-8.00, -7.87, -1.57] 
        ]

    def reset_detection_flags(self):
        self.green_flag = 0
        self.red_flag = 0
        self.blue_visible_now = 0
        self.blue_area = 0
        self.blue_cx = 0

    def clean_mask(self, mask):
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        return mask

    def callback(self, data):
        try:
            image = self.bridge.imgmsg_to_cv2(data, 'bgr8')
        except Exception as e:
            print('ERROR in callback image conversion:', e)
            return

        self.image_width = image.shape[1]
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # GREEN
        green_lower = np.array([60 - self.sensitivity, 100, 100])
        green_upper = np.array([60 + self.sensitivity, 255, 255])

        # BLUE
        blue_lower = np.array([120 - self.sensitivity, 100, 100])
        blue_upper = np.array([120 + self.sensitivity, 255, 255])

        # RED - two ranges because red wraps around HSV
        red_lower_1 = np.array([0, 100, 100])
        red_upper_1 = np.array([self.sensitivity, 255, 255])
        red_lower_2 = np.array([180 - self.sensitivity, 100, 100])
        red_upper_2 = np.array([179, 255, 255])

        # Masks
        green_mask = cv2.inRange(hsv_image, green_lower, green_upper)
        blue_mask = cv2.inRange(hsv_image, blue_lower, blue_upper)
        red_mask_1 = cv2.inRange(hsv_image, red_lower_1, red_upper_1)
        red_mask_2 = cv2.inRange(hsv_image, red_lower_2, red_upper_2)
        red_mask = cv2.bitwise_or(red_mask_1, red_mask_2)

        # Clean masks
        green_mask = self.clean_mask(green_mask)
        blue_mask = self.clean_mask(blue_mask)
        red_mask = self.clean_mask(red_mask)

        # Find contours
        green_contours, _ = cv2.findContours(
            green_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
        )
        blue_contours, _ = cv2.findContours(
            blue_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
        )
        red_contours, _ = cv2.findContours(
            red_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
        )

        self.reset_detection_flags()

        self.detect_green(image, green_contours)
        self.detect_blue(image, blue_contours)
        self.detect_red(image, red_contours)
        self.update_blue_confirmation()
        self.draw_debug_info(image)

        if self.show_debug:
            cv2.imshow('Detection', image)
            cv2.waitKey(1)

    def detect_green(self, image, green_contours):
        if len(green_contours) > 0:
            c = max(green_contours, key=cv2.contourArea)
            area = cv2.contourArea(c)

            if area > self.follow_area:
                self.green_flag = 1
                self.green_seen = 1

                x, y, w, h = cv2.boundingRect(c)
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    def detect_blue(self, image, blue_contours):
        if len(blue_contours) > 0:
            c = max(blue_contours, key=cv2.contourArea)
            area = cv2.contourArea(c)

            if area > self.follow_area:
                self.blue_visible_now = 1
                self.blue_seen = 1
                self.blue_area = area

                x, y, w, h = cv2.boundingRect(c)
                cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)

                M = cv2.moments(c)
                if M['m00'] != 0:
                    self.blue_cx = int(M['m10'] / M['m00'])
                    blue_cy = int(M['m01'] / M['m00'])
                    cv2.circle(image, (self.blue_cx, blue_cy), 5, (255, 0, 0), -1)

                    image_centre = self.image_width // 2
                    if self.blue_cx < image_centre:
                        self.last_seen_turn_direction = 'left'
                    else:
                        self.last_seen_turn_direction = 'right'

    def detect_red(self, image, red_contours):
        if len(red_contours) > 0:
            c = max(red_contours, key=cv2.contourArea)
            area = cv2.contourArea(c)

            if area > self.follow_area:
                self.red_flag = 1
                self.red_seen = 1

                x, y, w, h = cv2.boundingRect(c)
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)

    def update_blue_confirmation(self):
        if self.blue_visible_now == 1:
            self.blue_detect_count += 1
            self.blue_lost_count = 0
        else:
            self.blue_detect_count = 0

        if self.blue_detect_count >= self.blue_confirm_frames:
            self.blue_flag = 1
        else:
            self.blue_flag = 0

    def draw_debug_info(self, image):
        cv2.putText(
            image,
            'Seen R:{} G:{} B:{}'.format(self.red_seen, self.green_seen, self.blue_seen),
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )

        cv2.putText(
            image,
            'Blue area: {}'.format(int(self.blue_area)),
            (10, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )

        cv2.putText(
            image,
            'Pursue blue: {}'.format(int(self.pursue_blue)),
            (10, 75),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )

        cv2.putText(
            image,
            'Blue flag: {}'.format(int(self.blue_flag)),
            (10, 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )

    def walk_forward(self):
        desired_velocity = Twist()
        desired_velocity.linear.x = 0.12
        self.publisher.publish(desired_velocity)

    def turn_left(self):
        desired_velocity = Twist()
        desired_velocity.angular.z = 0.35
        self.publisher.publish(desired_velocity)

    def turn_right(self):
        desired_velocity = Twist()
        desired_velocity.angular.z = -0.35
        self.publisher.publish(desired_velocity)

    def stop(self):
        desired_velocity = Twist()
        self.publisher.publish(desired_velocity)

    def send_goal(self, x, y, yaw):
        print(f'SENDING GOAL -> index:{self.search_index} x:{x:.2f} y:{y:.2f} yaw:{yaw:.2f}')

        if not self.navigate_client.wait_for_server(timeout_sec=1.0):
            print('NAV2 SERVER NOT AVAILABLE')
            return

        goal_msg = NavigateToPose.Goal()
        goal_msg.pose.header.frame_id = 'map'
        goal_msg.pose.header.stamp = self.get_clock().now().to_msg()

        goal_msg.pose.pose.position.x = float(x)
        goal_msg.pose.pose.position.y = float(y)
        goal_msg.pose.pose.position.z = 0.0
        goal_msg.pose.pose.orientation.z = math.sin(yaw / 2.0)
        goal_msg.pose.pose.orientation.w = math.cos(yaw / 2.0)

        self.goal_sent = True
        self.goal_done = False

        self.send_goal_future = self.navigate_client.send_goal_async(goal_msg)
        self.send_goal_future.add_done_callback(self.goal_response_callback)

    def goal_response_callback(self, future):
        print('GOAL RESPONSE CALLBACK CALLED')
        try:
            self.goal_handle = future.result()
        except Exception as e:
            print('ERROR in goal_response_callback:', e)
            self.goal_handle = None
            self.goal_sent = False
            self.goal_done = True
            return

        if self.goal_handle is None or not self.goal_handle.accepted:
            print('GOAL NOT ACCEPTED')
            self.goal_handle = None
            self.goal_sent = False
            self.goal_done = True
            return

        print('GOAL ACCEPTED')
        self.get_result_future = self.goal_handle.get_result_async()
        self.get_result_future.add_done_callback(self.get_result_callback)

    def get_result_callback(self, future):
        print('GOAL RESULT CALLBACK CALLED')
        self.goal_handle = None
        self.goal_sent = False
        self.goal_done = True

    def cancel_goal(self):
        print('CANCEL GOAL CALLED')
        try:
            if self.goal_handle is not None:
                self.goal_handle.cancel_goal_async()
        except Exception as e:
            print('ERROR in cancel_goal:', e)

        self.goal_handle = None
        self.goal_sent = False
        self.goal_done = True

    def search_for_blue(self):
        if self.goal_done and len(self.search_positions) > 0:
            x = self.search_positions[self.search_index][0]
            y = self.search_positions[self.search_index][1]
            yaw = self.search_positions[self.search_index][2]

            self.send_goal(x, y, yaw)

            self.search_index += 1
            if self.search_index >= len(self.search_positions):
                self.search_index = 0

    def approach_blue(self):
        image_centre = self.image_width // 2

        print(
            f'APPROACH BLUE -> area:{self.blue_area:.1f} '
            f'stop_area:{self.stop_area} '
            f'cx:{self.blue_cx} '
            f'image_centre:{image_centre}'
        )

        if self.blue_area > self.stop_area:
            self.stop_count += 1
            print(f'STOP COUNT INCREMENTED -> {self.stop_count}/{self.stop_confirm_frames}')
        else:
            if self.stop_count != 0:
                print('STOP COUNT RESET TO 0')
            self.stop_count = 0

        if self.stop_count >= self.stop_confirm_frames:
            print('TASK FINISHED CONDITION MET')
            self.stop()
            self.task_finished = True
            return

        if self.blue_cx < image_centre - self.centre_margin:
            print('TURNING LEFT TO CENTER BLUE')
            self.turn_left()
        elif self.blue_cx > image_centre + self.centre_margin:
            print('TURNING RIGHT TO CENTER BLUE')
            self.turn_right()
        else:
            print('MOVING FORWARD TOWARD BLUE')
            self.walk_forward()


def main():
    rclpy.init(args=None)
    robot = Robot()

    print('ENTERING MAIN LOOP')

    try:
        while rclpy.ok():
            rclpy.spin_once(robot, timeout_sec=0.1)

            if robot.pursue_blue != robot.last_printed_mode:
                print(f'MODE CHANGE -> pursue_blue = {robot.pursue_blue}')
                robot.last_printed_mode = robot.pursue_blue

            if robot.blue_flag != robot.last_printed_blue_flag:
                print(
                    f'BLUE FLAG CHANGE -> blue_flag:{robot.blue_flag} '
                    f'blue_visible_now:{robot.blue_visible_now} '
                    f'blue_detect_count:{robot.blue_detect_count}'
                )
                robot.last_printed_blue_flag = robot.blue_flag

            if robot.goal_sent != robot.last_printed_goal_sent:
                print(f'GOAL_SENT CHANGE -> {robot.goal_sent}')
                robot.last_printed_goal_sent = robot.goal_sent

            if robot.goal_done != robot.last_printed_goal_done:
                print(f'GOAL_DONE CHANGE -> {robot.goal_done}')
                robot.last_printed_goal_done = robot.goal_done

            # Once blue is confirmed, switch once into blue pursuit
            if robot.blue_flag == 1 and not robot.pursue_blue:
                print('SWITCHING TO PURSUE BLUE MODE')
                robot.pursue_blue = True

            if not robot.pursue_blue:
                robot.search_for_blue()

            else:
                # Cancel Nav2 once when switching away from search
                if robot.goal_sent:
                    print('PURSUE MODE ACTIVE -> CANCELLING CURRENT NAV GOAL')
                    robot.cancel_goal()
                    time.sleep(0.2)

                # If blue is visible now, approach it
                if robot.blue_visible_now == 1:
                    robot.blue_lost_count = 0
                    robot.approach_blue()
                else:
                    # If blue is lost, turn to try to reacquire it
                    robot.blue_lost_count += 1
                    print(
                        f'BLUE LOST TEMPORARILY -> count:{robot.blue_lost_count}/'
                        f'{robot.blue_lost_threshold} turning {robot.last_seen_turn_direction}'
                    )

                    if robot.last_seen_turn_direction == 'left':
                        robot.turn_left()
                    else:
                        robot.turn_right()

                if robot.task_finished:
                    print('MAIN LOOP BREAKING -> task_finished = True')
                    robot.stop()
                    break

            time.sleep(0.05)

    except ROSInterruptException as e:
        print('ROSInterruptException CAUGHT:', e)
    except Exception as e:
        print('UNEXPECTED EXCEPTION IN MAIN:', e)
        raise
    finally:
        print('ENTERING FINALLY BLOCK')

        if robot.goal_sent or robot.goal_handle is not None:
            print('FINAL CLEANUP -> cancelling goal')
            robot.cancel_goal()

        print('FINAL CLEANUP -> stopping robot')
        robot.stop()

        print('FINAL CLEANUP -> destroying windows')
        cv2.destroyAllWindows()

        print('FINAL CLEANUP -> destroying node')
        robot.destroy_node()

        print('FINAL CLEANUP -> shutting down rclpy')
        rclpy.shutdown()


if __name__ == '__main__':
    main()
