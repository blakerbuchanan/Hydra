import os
import cv2
import json
import torch
import numpy as np
import supervision as sv
import pycocotools.mask as mask_util
from pathlib import Path
from supervision.draw.color import ColorPalette

import importlib

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from PIL import Image

from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection 

CUSTOM_COLOR_MAP = [
    "#e6194b",
    "#3cb44b",
    "#ffe119",
    "#0082c8",
    "#f58231",
    "#911eb4",
    "#46f0f0",
    "#f032e6",
    "#d2f53c",
    "#fabebe",
    "#008080",
    "#e6beff",
    "#aa6e28",
    "#fffac8",
    "#800000",
    "#aaffc3",
]

def single_mask_to_rle(mask):
    rle = mask_util.encode(np.array(mask[:, :, None], order="F", dtype="uint8"))[0]
    rle["counts"] = rle["counts"].decode("utf-8")
    return rle

class GroundedSAM2():
    def __init__(self, sam2_checkpoint, model_cfg, device):
        if torch.cuda.get_device_properties(0).major >= 8:
            # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        
        self.device = device

        # build SAM2 image predictor
        sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=device)
        self.sam2_predictor = SAM2ImagePredictor(sam2_model)

        # build grounding dino from huggingface
        model_id = "IDEA-Research/grounding-dino-tiny"
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.grounding_model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)

    def visualize_results(self, rgb_images_list, results, masks_batch, scores_batch, return_mask, output_dir):
        # get the box prompt for SAM 2
        for idx in range(len(rgb_images_list)):
            img, result = rgb_images_list[idx], results[idx]
            if return_mask:
                masks, scores = masks_batch[idx].astype(bool), scores_batch[idx]
            else:
                masks, scores = None, np.zeros(len(result))

            input_boxes = result["boxes"].cpu().numpy()
            print(f"========Found mask for index {idx}")
            # masks, scores, logits = self.sam2_predictor.predict(
            #     point_coords=None,
            #     point_labels=None,
            #     box=input_boxes,
            #     multimask_output=False,
            # )

            """
            Post-process the output of the model to get the masks, scores, and logits for visualization
            """
            # convert the shape to (n, H, W)
            if masks is not None:
                if masks.ndim == 4:
                    masks = masks.squeeze(1)

            confidences = result["scores"].cpu().numpy().tolist()
            class_names = result["labels"]
            class_ids = np.array(list(range(len(class_names))))

            labels = [
                f"{class_name} {confidence:.2f}"
                for class_name, confidence
                in zip(class_names, confidences)
            ]

            """
            Visualize image with supervision useful API
            """
            bgr_image = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            detections = sv.Detections(
                xyxy=input_boxes,  # (n, 4)
                mask=masks,  # (n, h, w)
                class_id=class_ids
            )

            """
            Note that if you want to use default color map,
            you can set color=ColorPalette.DEFAULT
            """
            box_annotator = sv.BoxAnnotator(color=ColorPalette.from_hex(CUSTOM_COLOR_MAP))
            annotated_frame = box_annotator.annotate(scene=bgr_image.copy(), detections=detections)

            label_annotator = sv.LabelAnnotator(color=ColorPalette.from_hex(CUSTOM_COLOR_MAP))
            annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)
            cv2.imwrite(os.path.join(output_dir, f"groundingdino_annotated_image_{idx}.jpg"), annotated_frame)

            mask_annotator = sv.MaskAnnotator(color=ColorPalette.from_hex(CUSTOM_COLOR_MAP))
            annotated_frame = mask_annotator.annotate(scene=annotated_frame, detections=detections)
            cv2.imwrite(os.path.join(output_dir, f"grounded_sam2_annotated_image_with_mask_{idx}.jpg"), annotated_frame)

            # convert mask into rle format
            # mask_rles = [single_mask_to_rle(mask) for mask in masks]

            input_boxes = input_boxes.tolist()
            scores = scores.tolist()
            # save the results in standard format
            logs = {
                "annotations" : [
                    {
                        "class_name": class_name,
                        "bbox": box,
                        # "segmentation": mask_rle,
                        "score": score,
                        "confidence": confidence,
                    }
                    for class_name, box, score, confidence in zip(class_names, input_boxes, scores, confidences)
                ],
                "box_format": "xyxy",
                "img_width": img.shape[0],
                "img_height": img.shape[1],
            }
            
            with open(os.path.join(output_dir, f"grounded_sam2_hf_model_demo_results_{idx}.json"), "w") as f:
                json.dump(logs, f, indent=4)


    def predict(self, rgb_images_list, depth_list, extrinsics_list, texts, output_dir: Path, return_mask=True, visualize=False):
        pil_images_list = [Image.fromarray(img).convert("RGB") for img in rgb_images_list]
        
        
        inputs = self.processor(images=pil_images_list, text=texts, return_tensors="pt").to(self.device)
        
        with torch.autocast(device_type=self.device, dtype=torch.bfloat16):
            with torch.no_grad():
                outputs = self.grounding_model(**inputs)

        target_sizes = [img.size[::-1] for img in pil_images_list]
        results = self.processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            box_threshold=0.4,
            text_threshold=0.3,
            target_sizes=target_sizes
        )

        """
        Results is a list of dict with the following structure:
        [
            {
                'scores': tensor([0.7969, 0.6469, 0.6002, 0.4220], device='cuda:0'), 
                'labels': ['car', 'tire', 'tire', 'tire'], 
                'boxes': tensor([[  89.3244,  278.6940, 1710.3505,  851.5143],
                                [1392.4701,  554.4064, 1628.6133,  777.5872],
                                [ 436.1182,  621.8940,  676.5255,  851.6897],
                                [1236.0990,  688.3547, 1400.2427,  753.1256]], device='cuda:0')
            }
        ]
        """
        results = [result for result in results if len(result["boxes"])>0]
        rgb_images_list = [img for result, img in zip(results,rgb_images_list) if len(result["boxes"])>0]
        depth_list = [dep for result, dep in zip(results,depth_list) if len(result["boxes"])>0]
        extrinsics_list = [extr for result, extr in zip(results,extrinsics_list) if len(result["boxes"])>0]

        masks_batch, scores_batch = None, 0
        if return_mask:
            # Filtering out empty outputs
            if len(rgb_images_list)>0:
                self.sam2_predictor.set_image_batch(rgb_images_list)
                input_boxes_batch = [result["boxes"].cpu().numpy() for result in results]
                masks_batch, scores_batch, logits = self.sam2_predictor.predict_batch(
                    None,
                    None,
                    box_batch=input_boxes_batch,
                    multimask_output=False,
                )
        
        if visualize:
            self.visualize_results(rgb_images_list, results, masks_batch, scores_batch, return_mask, output_dir)
            
        return results, masks_batch, rgb_images_list, depth_list, extrinsics_list
