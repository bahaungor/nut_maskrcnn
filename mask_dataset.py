# For jupyter, save this section as "mask_dataset.py" and import "from mask_dataset import MaskRCNN_Dataset"
import torch
import cv2
import xml.etree.ElementTree as ET
import torchvision.transforms as T
import numpy as np
class MaskRCNN_Dataset(torch.utils.data.Dataset):
    def __init__(self, df, classes, im_size= None, transforms = None):
        super().__init__()
        self.df = df
        self.transforms = transforms
        self.im_size = im_size
        self.images = df["image_path"].tolist()
        # self.EncodedPixels = df["EncodedPixels"].tolist() # MODEL FROM ENCODEDPIXELS
        self.annot_paths = df["annot_path"].tolist() # MODEL FROM BBOXES
        self.labels = df["label"].tolist() # Required format -> [wheat, wheat, wheat]
        self.classes = classes
        self.labels_dict = {c: i+1 for i, c in enumerate(classes)}
    
    def rle_decode(mask_rle, shape):
        '''
        mask_rle: run-length as string formated (start length)
        shape: original (height,width) of the image (not resized, or else it mess up)
        Returns numpy array, 1 - mask, 0 - background
        '''
        s = mask_rle.split()
        starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
        starts -= 1
        ends = starts + lengths
        img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
        for lo, hi in zip(starts, ends):
            img[lo:hi] = 1
        # return img.reshape(shape[::-1]).T # One of these returns methods is the correct decoder
        return img.reshape(shape) # One of these returns methods is the correct decoder

    def get_box(self, mask):
        ''' Get the bounding box for a given mask '''
        try:
            pos = np.where(mask)
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            return [int(xmin), int(ymin), int(xmax), int(ymax)]
        except:
            return None
        
    def get_mask_from_rle(self, idx,org_h,org_w,res_h,res_w):
        labels = []
        if len(self.EncodedPixels[idx]) == 0:
            # TensorFlow → (height, width, channel) | PyTorch → (channel, height, width)
            masks = np.zeros((1, res_h, res_w), dtype=np.uint8)
            labels.append(0)
        else:
            masks = np.zeros((len(self.EncodedPixels[idx]), res_h, res_w), dtype=np.uint8)
            for m, (EncodedPixel, label) in enumerate(zip(self.EncodedPixels[idx], self.labels[idx])):
                mask = self.rle_decode(EncodedPixel, (org_h, org_w)) # Decode mask with original values (else FAIL)
                if self.im_size: mask = cv2.resize(mask, (self.im_size))
                mask = np.array(mask) > 0
                masks[m, :, :] = mask
                labels.append(self.classes.index(label))
        return masks,labels

    def extract_boxes_from_xml(self, filename):
        tree = ET.parse(filename) # load and parse the file
        root = tree.getroot() # get the root of the document
        
        # extract each bounding box
        boxes,labels = [],[]
        if len(root.findall('.//bndbox')) == 0:
            labels.append(0)
        else:
            for box in root.findall('object'):
                xmin = int(box.find('bndbox').find('xmin').text)
                ymin = int(box.find('bndbox').find('ymin').text)
                xmax = int(box.find('bndbox').find('xmax').text)
                ymax = int(box.find('bndbox').find('ymax').text)
                if xmax > xmin and ymax > ymin:
                    boxes.append([xmin, ymin, xmax, ymax])
                    labels.append(self.classes.index(box.find('name').text))
        return boxes,labels
        
    def get_mask_from_bboxes(self,boxes,org_h,org_w,res_h,res_w):
        if len(boxes) == 0:
            # TensorFlow → (height, width, channel) | PyTorch → (channel, height, width)
            masks = np.zeros((1, res_h, res_w), dtype=np.uint8)
        else:
            masks = np.zeros((len(boxes), res_h, res_w), dtype=np.uint8)
            for i in range(len(boxes)):
                mask = np.zeros((org_h, org_w), dtype=np.uint8)
                row_s, row_e = int(boxes[i][1]), int(boxes[i][3])
                col_s, col_e = int(boxes[i][0]), int(boxes[i][2])
                mask[row_s:row_e, col_s:col_e] = 1
                if self.im_size: mask = cv2.resize(mask, (res_w, res_h))
                mask = np.array(mask) > 0
                masks[i, :, :] = mask
        return masks
    
    def __getitem__(self, idx):
        img = cv2.imread(self.images[idx])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
        img /= 255.0
        
        org_h,org_w = img.shape[:2] # Original image height,width
        if self.im_size: img = cv2.resize(img.copy(),self.im_size)
        res_h,res_w = img.shape[:2] # Resized image height,width
        
        boxes,labels = self.extract_boxes_from_xml(self.annot_paths[idx]) # MODEL FROM BBOXES
        masks = self.get_mask_from_bboxes(boxes,org_h,org_w,res_h,res_w) # MODEL FROM BBOXES
        # masks,labels = self.get_mask_from_rle(idx,org_h,org_w,res_h,res_w) # MODEL FROM ENCODEDPIXELS
        
        if self.transforms:
            aug = self.transforms(image = img, masks = list(masks), category_ids = labels)
            img = aug['image']
            masks = aug['masks']
            labels = aug['category_ids']
        
        # In case mask disappears during augmentation
        final_boxes,final_masks,final_labels = [],[],[]
        for i,mask in enumerate(masks):
            boxes = self.get_box(mask)
            if boxes is not None and boxes[2] > boxes[0] and boxes[3] > boxes[1]:
                final_boxes.append(boxes)
                final_masks.append(mask)
                final_labels.append(labels[i])

        # convert boxes into a Torch Tensors
        boxes = torch.as_tensor(final_boxes, dtype=torch.float32)
        labels = torch.as_tensor(final_labels, dtype=torch.int64)
        masks = torch.as_tensor(np.array(final_masks), dtype=torch.uint8)

        if len(final_boxes)==0:
            target = {'boxes': torch.zeros((0, 4), dtype=torch.float32),
                      'labels': torch.zeros((0,), dtype=torch.int64),
                      'masks': torch.zeros((0, res_h, res_w), dtype=torch.uint8),
                      'image_id': torch.tensor([idx]),
                      'area': torch.zeros((0,), dtype=torch.float32),
                      'iscrowd': torch.zeros((0,), dtype=torch.int64)}
        else:
            target = {'boxes': boxes, 'labels': labels, 
                      'area': (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0]), 
                      'masks':masks, 'image_id':torch.tensor([idx]), 
                      'iscrowd': torch.zeros((boxes.shape[0],), dtype=torch.int64)}

        return T.Compose([T.ToTensor()])(img), target

    def __len__(self):
        return len(self.images)