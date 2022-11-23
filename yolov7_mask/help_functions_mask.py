import torch
import cv2
import yaml
import numpy as np
import datetime
import io
import base64

from torchvision import transforms
from PIL import Image

from utils.datasets import letterbox
from utils.general import non_max_suppression_mask_conf, increment_path

from detectron2.modeling.poolers import ROIPooler
from detectron2.structures import Boxes
from detectron2.utils.memory import retry_if_cuda_oom
from detectron2.layers import paste_masks_in_image

## INITIATIE VAN MODEL EN SETTINGS
device = torch.device("cuda:0" if torch.cuda.is_available() else cpu)
half = device.type != "cpu"

#read in hyperparameters for yolov7
with open('config/hyp.mask.yaml') as f:
        hyp = yaml.load(f, Loader=yaml.FullLoader)

#read in additional settings
with open('config/settings.yaml') as f:
        settings = yaml.load(f, Loader=yaml.FullLoader)

#read in pretrained weights and model
weights = torch.load(settings['weights'])
model = weights['model'].to(device)
np.random.seed(settings['seed'])

if half:
    model = model.half()


#read in images as bytes and convert to tensor for model
def transform_image(image_bytes: bytes):
    image = Image.open(io.BytesIO(image_bytes))
    image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    image_ = image.copy() #original image resized back-up
    image = letterbox(image, 640, stride=64, auto=True)[0]
    image = transforms.ToTensor()(image)
    image = torch.tensor(np.array([image.numpy()]))
    return image, image_

#get bytestring of predicted image 
def get_result(image_file: np.ndarray):
    start_time = datetime.datetime.now()
    image_bytes = image_file.file.read()
    with torch.no_grad():
        image, image_ = transform_image(image_bytes)
        img_size = image_.shape
        image_display = get_prediction(image, image_)
    
    print('original size: ', img_size)
    image_display = cv2.resize(image_display, (img_size[1], img_size[0]))
    
    end_time = datetime.datetime.now()
    time_diff = (end_time - start_time)
    execution_time = f'{round(time_diff.total_seconds() * 1000)} ms'
    _, image_bytes = cv2.imencode('.jpg', image_display) #werkt met meerdere afbeelding formaten
    encoded_string = base64.b64encode(image_bytes) #moet met base64 encoding zijn anders werkt niet
    bs64 = encoded_string.decode('utf-8')
    image_data = f'data:image/jpeg;base64,{bs64}'   
    result =  {"image_data": image_data,
              "inference_time": execution_time}
    return result

#get predictions for CV2 images
def get_prediction(image: torch.Tensor, image_: np.ndarray):
    image = image.to(device)
    image = image.half() if half else image.float()
    output = model(image)
    
    inf_out, train_out, attn, mask_iou, bases, sem_output = output['test'], output['bbox_and_cls'], output['attn'], output['mask_iou'], output['bases'], output['sem']
    
    bases = torch.cat([bases, sem_output], dim=1)
    
    nb, _, height, width = image.shape
    names= model.names
    
    pooler = ROIPooler(output_size = hyp['mask_resolution'], scales=(model.pooler_scale,), sampling_ratio=1, pooler_type= 'ROIAlignV2', canonical_level=2)
    #non-maxima supression
    output, output_mask, output_mask_score, output_ac, output_ab = non_max_suppression_mask_conf(inf_out, attn, bases, pooler, hyp, conf_thres=settings['conf_thres'], iou_thres=settings['iou_thres'], merge=False, mask_iou=None)
    
    pred, pred_masks = output[0], output_mask[0]
    base = bases[0]
    
    #if no predictions above certaincy threshold, return original image
    if pred is not None:
        bboxes = Boxes(pred[:, :4])
        original_pred_masks = pred_masks.view(-1 , hyp['mask_resolution'], hyp['mask_resolution'])
        pred_masks = retry_if_cuda_oom(paste_masks_in_image)(original_pred_masks, bboxes, (height, width), threshold=0.5)
        
        pred_masks_np = pred_masks.detach().cpu().numpy()
        pred_cls = pred[:, 5].detach().cpu().numpy()
        pred_conf = pred[:, 4].detach().cpu().numpy()
        nbboxes = bboxes.tensor.detach().cpu().numpy().astype(int)
        
        image_display = image[0].permute(1, 2, 0)*255
        image_display = image_display.cpu().numpy().astype(np.uint8)
        image_display = cv2.cvtColor(image_display, cv2.COLOR_RGB2BGR)
        
        for one_mask, bbox, cls, conf in zip(pred_masks_np, nbboxes, pred_cls, pred_conf):
            if conf < settings['conf_thres']:
                continue
            
            color = [255, 0, 0]
            
            image_display[one_mask] = image_display[one_mask] *0.5 + np.array(color, dtype = np.uint8) * 0.5
            
            label = '%s %.3f' % (names[int(cls)], conf)
            
            tf = max(settings['thickness'] -1, 1)
            t_size = cv2.getTextSize(label, 0, fontScale=settings['thickness'] / 3, thickness = tf)[0]
            c2 = bbox[0] + t_size[0], bbox[1] - t_size[1] - 3
            
            if not settings['nobbox']:
                cv2.rectangle(image_display, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, thickness=settings['thickness'], lineType=cv2.LINE_AA)
            
            if not settings['nolabel']:
                cv2.rectangle(image_display, (bbox[0], bbox[1]), c2, color, -1, cv2.LINE_AA) #filled label
                cv2.putText(image_display, label, (bbox[0], bbox[1] -2), 0, settings['thickness'] / 3, [255,255,255], thickness=tf, lineType=cv2.LINE_AA)
            
        return image_display
        
    return image_
