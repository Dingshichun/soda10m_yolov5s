import json
import os
from pathlib import Path

def convert_bbox_to_yolo(bbox, img_w, img_h):
    x_center = (bbox[0] + bbox[2] / 2) / img_w
    y_center = (bbox[1] + bbox[3] / 2) / img_h
    w_norm = bbox[2] / img_w
    h_norm = bbox[3] / img_h
    return [max(0.0, min(1.0, x)) for x in [x_center, y_center, w_norm, h_norm]]

def convert_coco_to_yolo(json_path, img_dir, label_dir):
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # 1. 建立映射关系
    img_id_to_name = {img['id']: img['file_name'] for img in data['images']}
    cat_id_to_idx = {cat['id']: idx for idx, cat in enumerate(data['categories'])}
    
    # 2. 按图片分组标注（确保img_id为整数）
    annotations_by_img = {}
    for anno in data['annotations']:
        img_id = int(anno['image_id'])  # 强制转为整数
        if img_id not in annotations_by_img:
            annotations_by_img[img_id] = []
        annotations_by_img[img_id].append(anno)
    
    # 3. 创建标签输出目录
    Path(label_dir).mkdir(parents=True, exist_ok=True)
    missing_images = []
    
    # 4. 遍历图片（使用真实ID而非索引）
    for img_info in data['images']:  # 修复点：直接遍历图片信息
        img_id = int(img_info['id'])  # 真实ID
        img_name = img_info['file_name']
        img_path = None
        
        # 4.1 匹配图片文件
        img_stem = Path(img_name).stem
        for ext in ['.jpg', '.png', '.jpeg']:
            candidate = Path(img_dir) / f"{img_stem}{ext}"
            if candidate.exists():
                img_path = candidate
                break
        
        if not img_path:
            missing_images.append(img_name)
            print(f"⚠️ 图片未找到: {img_name}")
            continue
        
        # 4.2 获取图片尺寸
        img_w, img_h = img_info['width'], img_info['height']
        txt_path = Path(label_dir) / f"{img_stem}.txt"
        
        # 4.3 写入标签
        with open(txt_path, 'w') as f:
            if img_id in annotations_by_img:  # 用真实ID查询
                for anno in annotations_by_img[img_id]:
                    class_id = cat_id_to_idx[anno['category_id']]
                    bbox_yolo = convert_bbox_to_yolo(anno['bbox'], img_w, img_h)
                    f.write(f"{class_id} {' '.join(f'{x:.6f}' for x in bbox_yolo)}\n")
                print(f"✅ 生成标签: {txt_path} ({len(annotations_by_img[img_id])}个标注)")
            else:
                print(f"⚠️ 图片 {img_name} 无标注")
    
    # 5. 错误报告
    if missing_images:
        print(f"⚠️ 警告：{len(missing_images)}张图片未找到，首例：{missing_images[0]}")
if __name__=="__main__":
    # 使用示例
    # 1.转换训练集
    convert_coco_to_yolo(
        json_path = "./coco_to_yolo/instance_train.json",  # COCO格式JSON路径
        img_dir = "F:/丁世春/计算机视觉/深度学习数据集/SODA10M/labeled_trainval/SSLAD-2D/labeled/train",                         # 图片目录
        label_dir = "./soda10m/labels/train"                        # 标签输出目录
    ) 
    # 2.转换验证集
    convert_coco_to_yolo(
        json_path = "./coco_to_yolo/instance_val.json",  # COCO格式JSON路径
        img_dir = "F:/丁世春/计算机视觉/深度学习数据集/SODA10M/labeled_trainval/SSLAD-2D/labeled/val",                         # 图片目录
        label_dir = "./soda10m/labels/val"                        # 标签输出目录
    )  
    