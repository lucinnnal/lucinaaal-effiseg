import numpy as np
import torch
from src.utils.iouEval import iouEval

def compute_metrics(eval_preds):
    metric = dict()

    # Extract preditions and labels from eval_preds
    """
    EvalPrediction 클래스는 predictions, label_ids, inputs 3개의 attribute를 가지고 있다.
    소스 코드를 확인해 보면, 세 attribute 모두 numpy의 ndarray 타입이거나, ndarray를 원소로 갖는 튜플 타입이어야 한다. 또한, inputs는 위에서 살펴본 include_inputs_for_metrics가 False인 경우 None이다.
    """
    logits = eval_preds.predictions
    if isinstance(logits, dict):
        logits = logits["logits"]  # dict에서 logits 키로 꺼냄

    labels = eval_preds.label_ids    # shape: [B, 1, H, W]

    # iouEval class
    iou_eval = iouEval(nClasses=20)

    with torch.no_grad():

        logits = torch.from_numpy(logits)
        labels = torch.from_numpy(labels).unsqueeze(1) 
        preds = torch.argmax(logits, dim=1, keepdim=True)

        iou_eval.addBatch(preds, labels)
        mean_iou, class_iou = iou_eval.getIoU()
        
        # results
        metric["mean_iou"] = mean_iou.item()
        class_names = ['road', 'sidewalk', 'building', 'wall', 'fence', 
                      'pole', 'traffic light', 'traffic sign', 'vegetation', 
                      'terrain', 'sky', 'person', 'rider', 'car', 'truck',
                      'bus', 'train', 'motorcycle', 'bicycle']
        
        for idx, (name, iou) in enumerate(zip(class_names, class_iou)):
            metric[f"iou_{name}"] = iou.item()

    return metric