import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from detectron2.config import CfgNode, get_cfg
from detectron2.data import MetadataCatalog
from detectron2.engine.defaults import DefaultPredictor
from detectron2.utils.visualizer import ColorMode, VisImage, Visualizer
from omegaconf import DictConfig
from omegaconf import OmegaConf

# Detic libraries

sys.path.insert(0, '/home/saumyas/semnav_workspace/src/stretch_ai/src/stretch/perception/detection/detic/Detic/third_party/CenterNet2/')
from centernet.config import add_centernet_config
from stretch.perception.detection.detic.Detic.detic.config import add_detic_config
from stretch.perception.detection.utils import filter_depth, overlay_masks

from stretch.perception.detection.detic.Detic.detic.modeling.text.text_encoder import (  # noqa:E402
    build_text_encoder,
)
from stretch.perception.detection.detic.Detic.detic.modeling.utils import (  # noqa:E402
    reset_cls_test,
)

class DeticSegmenter:
    def __init__(self, cfg):
        """
        Args:
            cfg (CfgNode): The Detic-specific part of your config.
        """
        self.predictor = DefaultPredictor(self.setup_cfg(cfg.detic))

        self.vocabulary = 'custom'
        self.metadata = MetadataCatalog.get("__unused")
        self.cpu_device = torch.device("cpu")
        # Load and build a custom vocabulary based on hm3d_label_space
        vocab_cfg = OmegaConf.load(cfg.detic.custom_vocabulary_path)
        vocabulary = []
        for el in vocab_cfg.label_names:
            vocabulary.append(el['name'])

        self.metadata.thing_classes = vocabulary
        self.classifier = self.get_clip_embeddings(self.metadata.thing_classes)
        self.num_classes = len(self.metadata.thing_classes)
        reset_cls_test(self.predictor.model, self.classifier, self.num_classes)
    
    def __call__(self, img):
        outputs, viz, semantic, instance = self.run_on_image(img)
        return semantic
    
    def get_clip_embeddings(self, vocabulary, prompt='a '):
        text_encoder = build_text_encoder(pretrain=True)
        text_encoder.eval()
        texts = [prompt + x for x in vocabulary]
        emb = text_encoder(texts).detach().permute(1, 0).contiguous().cpu()
        return emb

    def setup_cfg(self, config: DictConfig) -> CfgNode:
        config_file = config.config_file
        weights = config.weights
    
        cfg = get_cfg() 
        add_centernet_config(cfg)
        add_detic_config(cfg)
        cfg.merge_from_file(config_file)
        cfg.MODEL.WEIGHTS = weights
        cfg.MODEL.RETINANET.SCORE_THRESH_TEST = config.confidence_threshold
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = config.confidence_threshold
        cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = config.confidence_threshold
        cfg.MODEL.ROI_BOX_HEAD.ZEROSHOT_WEIGHT_PATH = "rand"
        cfg.MODEL.ROI_BOX_HEAD.CAT_FREQ_PATH = cfg.MODEL.ROI_BOX_HEAD.CAT_FREQ_PATH
        cfg.MODEL.ROI_HEADS.ONE_CLASS_PER_PROPOSAL = True
        cfg.freeze()
        return cfg

    def run_on_image(self, image):
        """
        Args:
            image (np.ndarray): an image of shape (H, W, C) (in BGR order).
                This is the format used by OpenCV.

        Returns:
            predictions (dict): the output of the model.
            vis_output (VisImage): the visualized image output.
        """
        vis_output = None
        predictions = self.predictor(image)

        # Convert image from OpenCV BGR format to Matplotlib RGB format.

        height, width, _ = image.shape

        # Sort instances by mask size
        masks = predictions["instances"].pred_masks.cpu().numpy()
        class_idcs = predictions["instances"].pred_classes.cpu().numpy()

        semantic_map, instance_map = overlay_masks(masks, class_idcs, (height, width))

        semantic = semantic_map.astype(np.int32)
        instance = instance_map.astype(np.int32)

        # For visualization
        image = image[:, :, ::-1]
        visualizer = Visualizer(image, self.metadata)
        if "panoptic_seg" in predictions:
            panoptic_seg, segments_info = predictions["panoptic_seg"]
            vis_output = visualizer.draw_panoptic_seg_predictions(
                panoptic_seg.to(self.cpu_device), segments_info
            )
        else:
            if "sem_seg" in predictions:
                vis_output = visualizer.draw_sem_seg(
                    predictions["sem_seg"].argmax(dim=0).to(self.cpu_device)
                )
            if "instances" in predictions:
                instances = predictions["instances"].to(self.cpu_device)
                vis_output = visualizer.draw_instance_predictions(predictions=instances)

            return predictions, vis_output, semantic, instance



if __name__ == "__main__":
    import argparse

    # Change the model's vocabulary to a customized one and get their word-embedding 
    #  using a pre-trained CLIP model.
    parser = argparse.ArgumentParser()
    parser.add_argument("-cf", "--cfg_file", help="cfg file name", default="", type=str, required=True)
    args = parser.parse_args()

    config_path = Path(__file__).resolve().parent.parent / 'commands' / 'cfg' / f'{args.cfg_file}.yaml'
    cfg = OmegaConf.load(config_path)

    # predictor = DefaultPredictor(setup_cfg(cfg.detic))
    
    # def get_clip_embeddings(vocabulary, prompt='a '):
    #     text_encoder = build_text_encoder(pretrain=True)
    #     text_encoder.eval()
    #     texts = [prompt + x for x in vocabulary]
    #     emb = text_encoder(texts).detach().permute(1, 0).contiguous().cpu()
    #     return emb
    
    # vocabulary = 'custom'
    # metadata = MetadataCatalog.get("__unused")

    # # Load and build a custom vocabulary based on hm3d_label_space
    # vocab_cfg = OmegaConf.load(cfg.detic.custom_vocabulary_path)
    # vocabulary = []
    # for el in vocab_cfg.label_names:
    #     vocabulary.append(el['name'])

    # metadata.thing_classes = vocabulary
    # classifier = get_clip_embeddings(metadata.thing_classes)
    # num_classes = len(metadata.thing_classes)
    # reset_cls_test(predictor.model, classifier, num_classes)

    detic_segmenter = DeticSegmenter(cfg)

    for i in range(76):
        im = cv2.imread("/home/saumyas/semnav_workspace/src/hydra/python/src/hydra_python/detection/images_habitat/scene2/traj0/img_" + str(i) +".png")
        outputs, viz, semantic, instance = detic_segmenter.run_on_image(im)
        semantic = detic_segmenter(im)
        import ipdb; ipdb.set_trace()
        v = Visualizer(im[:, :, ::-1], detic_segmenter.metadata)
        out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        out.save('detic_test_out.png')
        import ipdb; ipdb.set_trace()