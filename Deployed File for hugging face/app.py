import gradio as gr
import torch
from ultralyticsplus import YOLO, render_result


# torch.hub.download_url_to_file(
#     'https://mattpearsonaviation.com/wp-content/uploads/2017/12/IMG_0560.jpg', 'one.jpg')
# # torch.hub.download_url_to_file(
# #     'https://cdn.airplane-pictures.net/images/uploaded-images/2011/11/25/169465.jpg', 'two.jpg')
# torch.hub.download_url_to_file(
#     'https://imgproc.airliners.net/photos/airliners/7/1/9/0767917.jpg?v=v40', 'three.jpg')


def yoloV8_func(image: gr.Image = None,
                image_size: gr.Slider = 640,
                conf_threshold: gr.Slider = 0.4,
                iou_threshold: gr.Slider = 0.50):
    """This function performs YOLOv8 object detection on the given image.

    Args:
        image (gr.inputs.Image, optional): Input image to detect objects on. Defaults to None.
        image_size (gr.inputs.Slider, optional): Desired image size for the model. Defaults to 640.
        conf_threshold (gr.inputs.Slider, optional): Confidence threshold for object detection. Defaults to 0.4.
        iou_threshold (gr.inputs.Slider, optional): Intersection over Union threshold for object detection. Defaults to 0.50.
    """
    # Trained dataset
    model_path = "best.pt"
    model = YOLO(model_path)

   
    results = model.predict(image,
                            conf=conf_threshold,
                            iou=iou_threshold,
                            imgsz=image_size)


    box = results[0].boxes
    print("Object type:", box.cls)
    print("Coordinates:", box.xyxy)
    print("Probability:", box.conf)


    render = render_result(model=model, image=image, result=results[0])
    return render


inputs = [
    gr.Image(type="filepath", label="Input Image"),
    gr.Slider(minimum=320, maximum=1280, value=640,
                     step=32, label="Image Size"),
    gr.Slider(minimum=0.0, maximum=1.0, value=0.25,
                     step=0.05, label="Confidence Threshold"),
    gr.Slider(minimum=0.0, maximum=1.0, value=0.45,
                     step=0.05, label="IOU Threshold"),
]


outputs = gr.Image(type="filepath", label="Output Image")

title = "👨‍💻Made By Team 8848(TataSafeguard)👨‍💻: Aircraft Damage Detection leveraging advanced IOT integration features."

# examples = [['one.jpg', 640, 0.5, 0.7],
#             ['two.jpg', 800, 0.5, 0.6],
#             ['three.jpg', 900, 0.5, 0.8]]

yolo_app = gr.Interface(
    fn=yoloV8_func,
    inputs=inputs,
    outputs=outputs,
    title=title,
    # examples=examples,
    # cache_examples=True,
)

# Launching Gradio interface 
yolo_app.launch(share=True, debug=True)
