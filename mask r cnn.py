
import torch # type: ignore
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.transforms import functional as F
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from io import BytesIO
import torchvision
import torch
from torchvision.models.detection import maskrcnn_resnet50_fpn, fasterrcnn_resnet50_fpn
from torchvision.transforms import functional as F
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from io import BytesIO

class Mask_RCNN_Annotator:

    def __init__(self, num_classes, detection_model_name="maskrcnn_resnet50_fpn"):
        # Load the model
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.model = self.get_model(detection_model_name, num_classes=num_classes)
        self.model.eval()
        self.score_threshold = 0.1

    def get_model(self, model_name, num_classes):
        if model_name == "maskrcnn_resnet50_fpn":
            model = maskrcnn_resnet50_fpn(pretrained=True)
            in_features = model.roi_heads.box_predictor.cls_score.in_features
            model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)
            return model.to(self.device)
        else:
            raise ValueError("Unknown model name. Currently only 'maskrcnn_resnet50_fpn' is supported.")

    def inference(self, image_bytes):
        image = Image.open(BytesIO(image_bytes)).convert('RGB')
        # Convert image to tensor
        image_tensor = F.to_tensor(image).unsqueeze(0).to(self.device)

        # Perform inference
        with torch.no_grad():
            outputs = self.model(image_tensor)

        # Filter results
        filtered_annotations = self.filter_results(outputs[0])
        # Convert predicted_boxes to MS COCO format
        results = self.convert_to_coco_format(filtered_annotations, image)

        return results

    def filter_results(self, outputs):
        filtered_annotations = []

        scores = outputs['scores'].cpu().detach().numpy()
        labels = outputs['labels'].cpu().detach().numpy()
        boxes = outputs['boxes'].cpu().detach().numpy()

        for score, box, label in zip(scores, boxes, labels):
            if score > self.score_threshold:
                prediction = {}
                prediction["bounding_box"] = box
                prediction["category"] = label
                prediction["confidence"] = float(score)
                filtered_annotations.append(prediction)

        return filtered_annotations

    def convert_to_coco_format(self, annotations, image):
        image_width, image_height = image.size

        for result in annotations:
            box = result["bounding_box"]
            x_min, y_min, x_max, y_max = box
            x_min = max(0, int(x_min))
            y_min = max(0, int(y_min))
            x_max = min(image_width, int(x_max))
            y_max = min(image_height, int(y_max))
            width = x_max - x_min
            height = y_max - y_min
            result["bounding_box"] = {"xmin": int(x_min), "ymin": int(y_min), "xmax": int(x_max), "ymax": int(y_max)}

        return annotations

    def plot_image(self, image_path, annotations=None):
        """
        Plot the image located at the specified path with optional bounding boxes in MS COCO format.

        Args:
        - image_path (str): The path to the image file.
        - annotations (list): A list of bounding boxes in MS COCO format, category, confidence from above function for testing.
        Each bounding box should be a list [x_min, y_min, width, height].
        Default is None.
        """
        # Open the image file
        image = Image.open(image_path)

        # Plot the image
        plt.imshow(image)
        plt.axis('off')  # Hide axis

        # Plot annotations if provided
        if not annotations: return
        for ann in annotations:
            bbox = ann['bounding_box']
            category = ann['category']
            confidence = ann['confidence']
            xmin, ymin = bbox['xmin'], bbox['ymin']
            xmax, ymax = bbox['xmax'], bbox['ymax']
            rect = patches.Rectangle((xmin, ymin), xmax-xmin, ymax-ymin, linewidth=1, edgecolor='r', facecolor='none')
            plt.gca().add_patch(rect)
            plt.text(xmin, ymin, f"{category}:{confidence:.3f}", color='r', verticalalignment='top', bbox={'color': 'white', 'pad': 0})

        plt.show()

# Example usage:
# annotator = Mask_RCNN_Annotator(num_classes=91)  # COCO has 80 classes + background
# with open("image.jpg", "rb") as image_file:
#     image_bytes = image_file.read()
# annotations = annotator.inference(image_bytes)
# annotator.plot_image("image.jpg", annotations)
