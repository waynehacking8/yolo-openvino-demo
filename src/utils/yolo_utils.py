import numpy as np
import cv2

class YOLOUtils:
    @staticmethod
    def xywh2xyxy(x):
        """將 (x, y, w, h) 轉換為 (x1, y1, x2, y2)"""
        y = np.zeros_like(x)
        y[..., 0] = x[..., 0] - x[..., 2] / 2  # x1
        y[..., 1] = x[..., 1] - x[..., 3] / 2  # y1
        y[..., 2] = x[..., 0] + x[..., 2] / 2  # x2
        y[..., 3] = x[..., 1] + x[..., 3] / 2  # y2
        return y
    
    @staticmethod
    def box_iou(box1, box2):
        """計算兩個邊界框的 IoU"""
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        
        inter = (min(box1[2], box2[2]) - max(box1[0], box2[0])) * \
                (min(box1[3], box2[3]) - max(box1[1], box2[1]))
        
        union = area1 + area2 - inter
        return inter / union
    
    @staticmethod
    def non_max_suppression(boxes, scores, iou_threshold=0.45):
        """執行非極大值抑制"""
        indices = cv2.dnn.NMSBoxes(boxes, scores, 0.25, iou_threshold)
        return indices
    
    @staticmethod
    def process_output(output, conf_threshold=0.25, iou_threshold=0.45):
        """處理模型輸出"""
        # 假設輸出格式為 [batch, num_boxes, num_classes + 5]
        predictions = output[0]  # 取第一個批次
        
        # 獲取置信度最高的類別
        scores = np.max(predictions[:, 5:], axis=1)
        class_ids = np.argmax(predictions[:, 5:], axis=1)
        
        # 過濾低置信度的預測
        mask = scores > conf_threshold
        boxes = predictions[mask, :4]
        scores = scores[mask]
        class_ids = class_ids[mask]
        
        # 轉換為 xyxy 格式
        boxes = YOLOUtils.xywh2xyxy(boxes)
        
        # 執行 NMS
        indices = YOLOUtils.non_max_suppression(boxes, scores, iou_threshold)
        
        return {
            'boxes': boxes[indices],
            'scores': scores[indices],
            'class_ids': class_ids[indices]
        }
    
    @staticmethod
    def draw_detections(image, detections, class_names):
        """繪製檢測結果"""
        img = image.copy()
        
        for box, score, class_id in zip(detections['boxes'], 
                                      detections['scores'], 
                                      detections['class_ids']):
            x1, y1, x2, y2 = map(int, box)
            label = f"{class_names[class_id]} {score:.2f}"
            
            # 繪製邊界框
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # 繪製標籤
            cv2.putText(img, label, (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        return img 