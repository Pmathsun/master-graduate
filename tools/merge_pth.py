import torch

# 加载水平框模型的 checkpoint
horizontal_checkpoint = torch.load('/home/ssy/mmrotate/logs/idea1/1016-horihead/epoch_12.pth')
horizontal_state_dict = horizontal_checkpoint['state_dict']  # 提取水平框的 state_dict

# 加载旋转框模型的 checkpoint
rotated_checkpoint = torch.load('/home/ssy/mmrotate/logs/idea1/1018-only-oriented/oriented_rcnn_r50_fpn_1x_dota_le90_idea1/epoch_12.pth')
rotated_state_dict = rotated_checkpoint['state_dict']  # 提取旋转框的 state_dict

# 合并模型权重
combined_state_dict = rotated_state_dict.copy()  # 复制水平框的权重

# 用旋转框的权重替换掉旋转框部分
for key in horizontal_state_dict.keys():
    print(key)
    if 'cls_bbox_head' in key:  # 如果键属于旋转框部分，进行替换
        print(666)
        combined_state_dict[key] = horizontal_state_dict[key]

# 创建新的 checkpoint 包含合并后的权重和原来的 meta 信息
combined_checkpoint = {
    'meta': horizontal_checkpoint['meta'],  # 保持原始 meta 信息
    'state_dict': combined_state_dict,  # 合并后的 state_dict
    'optimizer': horizontal_checkpoint['optimizer']  # 保持原始 optimizer 信息
}

# 保存新的 checkpoint 文件
torch.save(combined_checkpoint, '/home/ssy/mmrotate/logs/idea1/1018-merged/combined_model.pth')
print("Combined model saved successfully!")
