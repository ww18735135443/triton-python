import cv2
import numpy as np
def draw_one_area(img, area_num, area):
    cv2.polylines(img, [area.contour], True, area.color, 4)
    # area_sumx = 0
    # area_sumy = 0
    # for j in range(len(area.contour)):
    #     area_sumx = area_sumx + area.contour[j][0]
    #     area_sumy = area_sumy + area.contour[j][1]
    # area_midx = area_sumx // len(area.contour)
    # area_midy = area_sumy // len(area.contour)
    # # cv2.putText(img, "area" + str(area_num + 1), (int(area_midx), int(area_midy)), cv2.FONT_HERSHEY_PLAIN, 4, (255, 0, 255), 4)


def draw_areas(img, areas):
    for i, area in enumerate(areas):
        draw_one_area(img, i, area)

class Colors:
    """
    Ultralytics default color palette https://ultralytics.com/.

    This class provides methods to work with the Ultralytics color palette, including converting hex color codes to
    RGB values.

    Attributes:
        palette (list of tuple): List of RGB color values.
        n (int): The number of colors in the palette.
        pose_palette (np.array): A specific color palette array with dtype np.uint8.
    """

    def __init__(self):
        """Initialize colors as hex = matplotlib.colors.TABLEAU_COLORS.values()."""
        hexs = ('FF3838', 'FF9D97', 'FF701F', 'FFB21D', 'CFD231', '48F90A', '92CC17', '3DDB86', '1A9334', '00D4BB',
                '2C99A8', '00C2FF', '344593', '6473FF', '0018EC', '8438FF', '520085', 'CB38FF', 'FF95C8', 'FF37C7')
        self.palette = [self.hex2rgb(f'#{c}') for c in hexs]
        self.n = len(self.palette)
        self.pose_palette = np.array([[255, 128, 0], [255, 153, 51], [255, 178, 102], [230, 230, 0], [255, 153, 255],
                                      [153, 204, 255], [255, 102, 255], [255, 51, 255], [102, 178, 255], [51, 153, 255],
                                      [255, 153, 153], [255, 102, 102], [255, 51, 51], [153, 255, 153], [102, 255, 102],
                                      [51, 255, 51], [0, 255, 0], [0, 0, 255], [255, 0, 0], [255, 255, 255]],
                                     dtype=np.uint8)

    def __call__(self, i, bgr=False):
        """Converts hex color codes to RGB values."""
        c = self.palette[int(i) % self.n]
        return (c[2], c[1], c[0]) if bgr else c

    @staticmethod
    def hex2rgb(h):
        """Converts hex color codes to RGB values (i.e. default PIL order)."""
        return tuple(int(h[1 + i:1 + i + 2], 16) for i in (0, 2, 4))


colors = Colors()  # create instance for 'from utils.plots import colors'
def draw_detections(img, detections):
    """
    Draws bounding boxes and labels on the input image based on the detected objects.

    Args:
        img: The input image to draw detections on.
        box: Detected bounding box.
        score: Corresponding detection score.
        class_id: Class ID for the detected object.

    Returns:
        None
    """
    for detection in detections:
        # pass
    # for box, score, class_id in zip(bboxs, scores, class_ids):
        # Extract the coordinates of the bounding box
        x1, y1, x2, y2 = [int(i) for i in detection['xyxy']]
        class_label=detection['cls']
        score=detection['conf']
        # Retrieve the color for the class ID
        # color = colors.color_palette.palette[class_id]
        color=[255,0,0]

        # Draw the bounding box on the image
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)

        # Create the label text with class name and score
        label = f'{class_label}: {score:.2f}'
        # label = f'{self.classes[class_id]}'
        # Calculate the dimensions of the label text
        (label_width, label_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

        # Calculate the position of the label text
        label_x = x1
        label_y = y1 - 10 if y1 - 10 > label_height else y1 + 10

        # Draw a filled rectangle as the background for the label text
        cv2.rectangle(img, (label_x, label_y - label_height), (label_x + label_width, label_y + label_height), color,
                      cv2.FILLED)

        # Draw the label text on the image
        cv2.putText(img, label, (label_x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

    # return img
def draw_line_dir(img, lines,Thickness=1,color=(85, 90, 255)):
    for i, line in enumerate(lines):
        draw_line(img, i, line,Thickness,color)
        if line.single_dir:
            draw_direction_vector(img, line)


def draw_line(img, line_num, line,Thickness=1,color=(85, 90, 255)):
    x1, y1 = line.p0
    x2, y2 = line.p1
    cv2.line(img, (x1, y1), (x2, y2), color, line.lineThickness*Thickness)
    # cv2.putText(img, "boundary" + str(line_num + 1), (x1 - 20, y1 - 10), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 2)


def draw_direction_vector(img, line):
    x1, y1 = line.p_d_0
    x2, y2 = line.p_d_1
    color = (96, 206, 142)
    cv2.arrowedLine(img, (x1, y1), (x2, y2), color, 4, 8, 0, 0.2)

