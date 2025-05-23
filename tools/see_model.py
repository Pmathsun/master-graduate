import torch
combained = torch.load('/home/ssy/mmrotate/logs/1018-code-final/epoch_16.pth')
combained_dict = combained['state_dict']  # 提取水平框的 
print(combained_dict['roi_head.weights'])

# combained = torch.load('/home/ssy/mmrotate/logs/1018-code-final/epoch_16.pth')
# combained_dict = combained['state_dict']  # 提取水平框的
# print(combained_dict['roi_head.weights'])