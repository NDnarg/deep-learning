import os
import matplotlib.pyplot as plt
import scipy.io
import scipy.misc
from keras import backend as K
from keras.models import load_model, Model
from C4week3 import yolo_utils
from C4week3.yad2k.models.keras_yolo import yolo_head
from C4week3.yolo_utils import preprocess_image, yolo_eval, generate_colors, draw_boxes

sess = K.get_session()
class_names = yolo_utils.read_classes("model_data/coco_classes.txt")
anchors = yolo_utils.read_anchors("model_data/yolo_anchors.txt")
image_shape = (720.,1280.)
yolo_model = load_model("model_data/yolov2.h5")
yolo_outputs = yolo_head(yolo_model.output, anchors, len(class_names))
scores, boxes, classes = yolo_eval(yolo_outputs, image_shape)
def predict(sess, image_file, is_show_info=True, is_plot=True):
    # 图像预处理
    image, image_data = preprocess_image("test/" + image_file, model_image_size=(608, 608))

    # 运行会话并在feed_dict中选择正确的占位符.
    out_scores, out_boxes, out_classes = sess.run([scores, boxes, classes],
                                                  feed_dict={yolo_model.input: image_data, K.learning_phase(): 0})

    # 打印预测信息
    if is_show_info:
        print("在" + str(image_file) + "中找到了" + str(len(out_boxes)) + "个锚框。")

    # 指定要绘制的边界框的颜色
    colors = generate_colors(class_names)

    # 在图中绘制边界框
    draw_boxes(image, out_scores, out_boxes, out_classes, class_names, colors)

    # 保存已经绘制了边界框的图
    image.save(os.path.join("test", image_file), quality=100)

    # 打印出已经绘制了边界框的图
    if is_plot:
        output_image = scipy.misc.imread(os.path.join("test", image_file))
        plt.imshow(output_image)

    return out_scores, out_boxes, out_classes
out_scores, out_boxes, out_classes = predict(sess, "test.jpg")
