# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F

from mmrotate.core import rbbox2roi,obb2xyxy,obb2poly
from ..builder import (ROTATED_HEADS,build_roi_extractor)
from .rotate_standard_roi_head import RotatedStandardRoIHead


@ROTATED_HEADS.register_module()
class OrientedAttentionRoIHead(RotatedStandardRoIHead):
    def __init__(self,
                 attention_heads,
                 bbox_roi_extractor=None,
                 bbox_head=None,
                 shared_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None,
                 version='oc',
                 hbbox_roi_extractor=None):  # 新增参数

        # 调用父类的构造函数，传递父类的参数
        super(OrientedAttentionRoIHead, self).__init__(
            bbox_roi_extractor=bbox_roi_extractor,
            bbox_head=bbox_head,
            shared_head=shared_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained,
            init_cfg=init_cfg,
            version=version,
            hbbox_roi_extractor=hbbox_roi_extractor)
        
        #需要改一下
        self.attention=SelfAttention(bbox_roi_extractor.out_channels,attention_heads)
        if hbbox_roi_extractor is not None:
            self.cls_bbox_roi_extractor = build_roi_extractor(hbbox_roi_extractor)
        
    """Oriented RCNN roi head including one bbox head.""" 
    def forward_dummy(self, x, proposals):
        outs = ()
        rois = rbbox2roi([proposals])
        if self.with_bbox:
            bbox_results = self.just_forward(x,rois)
            outs = outs + (bbox_results['cls_score'],
                           bbox_results['bbox_pred'])
        return outs

    def forward_train(self,
                      x,
                      img_metas,
                      proposal_list,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None):
        # assign gts and sample proposals
        if self.with_bbox:

            num_imgs = len(img_metas)
            if gt_bboxes_ignore is None:
                gt_bboxes_ignore = [None for _ in range(num_imgs)]
            sampling_results = []
            for i in range(num_imgs):
                assign_result = self.bbox_assigner.assign(
                    proposal_list[i], gt_bboxes[i], gt_bboxes_ignore[i],
                    gt_labels[i])
                sampling_result = self.bbox_sampler.sample(
                    assign_result,
                    proposal_list[i],
                    gt_bboxes[i],
                    gt_labels[i],
                    feats=[lvl_feat[i][None] for lvl_feat in x])

                if gt_bboxes[i].numel() == 0:
                    sampling_result.pos_gt_bboxes = gt_bboxes[i].new(
                        (0, gt_bboxes[0].size(-1))).zero_()
                else:
                    sampling_result.pos_gt_bboxes = \
                        gt_bboxes[i][sampling_result.pos_assigned_gt_inds, :]

                sampling_results.append(sampling_result)

        losses = dict()
        # bbox head forward and loss
        # 拿到了采样结果 正负样本 进行训练
        if self.with_bbox:
            bbox_results = self._bbox_forward_train(x, sampling_results,
                                                    gt_bboxes, gt_labels,
                                                    img_metas)
            losses.update(bbox_results['loss_bbox'])

        return losses

    def _bbox_forward_train(self, x, sampling_results, gt_bboxes, gt_labels,
                            img_metas):
        # 1.获取旋转框
        rois = rbbox2roi([res.bboxes for res in sampling_results])#batch,x,y,w,h,a
        bbox_results = self.just_forward(x,rois)
        # 8. 计算倾斜框的损失值
        bbox_targets = self.bbox_head.get_targets(sampling_results, gt_bboxes,
                                                  gt_labels, self.train_cfg)
        loss_bbox = self.bbox_head.loss(bbox_results['cls_score'],
                                        bbox_results['bbox_pred'], rois,
                                        *bbox_targets)
        bbox_results.update(loss_bbox=loss_bbox)
        return bbox_results
    
    def just_forward(self,x,rois):
        # 1.获取水平框
        hbboxes = self.roi2hbbox(rois)#batch,x1,y1,x2,y2
        #img_nums = len(img_metas)
        # 2.进行旋转框特征提取
        rot_bbox_feats = self.get_rot_features(x,rois)
        # 3.进行水平框特征提取
        hori_outside_bbox_feats = self.get_hori_features(x,hbboxes)
        # 4. 计算IOU比例
        rot_area=rois[:,2]*rois[:,3]
        hori_area=(hbboxes[:,3]-hbboxes[:,1])*(hbboxes[:,2]-hbboxes[:,0])
        area_ratio = (rot_area / (hori_area + 1e-6)).clamp(max=1.0)  # 避免极值，控制在0~1之间
        # 5. 融合旋转框特征和水平外接部分特征 采用自注意力机制
        fused_features = self.fuse_features(rot_bbox_feats,hori_outside_bbox_feats,area_ratio)
        # 6. 进行前向推理工作
        if self.with_shared_head:
            fused_features = self.shared_head(fused_features)
        cls_score, bbox_pred = self.bbox_head(fused_features)
        bbox_results = dict(
            cls_score=cls_score, bbox_pred=bbox_pred, bbox_feats=fused_features)
        return bbox_results
    
    def get_rot_features(self,x,rois):
        rot_bbox_feats = self.bbox_roi_extractor(
            x[:self.bbox_roi_extractor.num_inputs], rois)
        return rot_bbox_feats
    
    def get_hori_features(self, x, horis):
        hori_bbox_feats = self.cls_bbox_roi_extractor(
            x[:self.bbox_roi_extractor.num_inputs],horis)
        return hori_bbox_feats
        # hori_bbox_feats = []
        # feats = x[:self.cls_bbox_roi_extractor.num_inputs]

        # # 计算每张图片的 RoI 数量
        # total_rois = horis.size(0)
        # rois_per_img = total_rois // img_nums  # 假设每张图片的 RoI 数量相同
        # # 假设 horis 和 rois 前 512 个是第一张图，后 512 个是第二张图
        # horis_split = torch.split(horis, rois_per_img)
        # rois_split = torch.split(rois, rois_per_img)
        
        # for img_idx in range(img_nums):  # 假设是两张图片
        #     horis_img = horis_split[img_idx]
        #     rois_img = rois_split[img_idx]
        #     cur_hori_bbox_feats=[]
        #     # 对每张图像的 RoI 进行处理
        #     for i, hori in enumerate(horis_img):
        #         roi_ = rois_img[i].unsqueeze(0)  # 获取与 hori 对应的旋转框 roi
                
        #         # 对于每个 roi，在 feats 的每个特征图进行 mask 掉 得到该图片的新feat
        #         mask_feats = []
        #         for j, feat in enumerate(feats):
        #             cur_feat = feat[img_idx].unsqueeze(0)
        #             # 生成与特征图对应的旋转框掩码，并应用到特征图上
        #             mask = self.generate_mask(roi_, cur_feat.shape[-2:], stride=self.cls_bbox_roi_extractor.featmap_strides[j])
        #             mask_feat = cur_feat * mask.unsqueeze(0).unsqueeze(0)  # 扩展掩码到通道维度，应用掩码到特征图
        #             mask_feats.append(mask_feat)

        #         # 然后进行水平框特征的提取
        #         single_hori_bbox_feats = self.cls_bbox_roi_extractor(mask_feats, hori.unsqueeze(0))
        #         cur_hori_bbox_feats.append(single_hori_bbox_feats)
        #     cur_hori_bbox_feats = torch.cat(cur_hori_bbox_feats, dim=0)  # 将 list 拼接为 [512, 256, 7, 7]
        #     hori_bbox_feats.append(cur_hori_bbox_feats)  # 将每张图片的特征加入列表
    
        # # TODO 2: 将 hori_bbox_feats 拼接为 [512 * img_nums, 256, 7, 7]
        # hori_bbox_feats = torch.cat(hori_bbox_feats, dim=0)  # 将 img_nums 个 [512, 256, 7, 7]
        # return hori_bbox_feats
    
    def generate_mask(self,rbbox, feature_map_size, stride):
        """
        根据旋转框的几何信息生成掩码，将旋转框区域 mask 掉。
        
        Args:
            rbbox (Tensor): 旋转框的坐标 (cx, cy, w, h, angle)，分别为中心点、宽高和旋转角度。
            feature_map_size (Tuple[int, int]): 特征图的尺寸 (height, width)
            stride (int): 特征图相对于原图的缩放比例
        
        Returns:
            Tensor: 掩码矩阵，1 表示保留，0 表示旋转框区域被 mask 掉。
        """
        rbbox = rbbox.clone()
        rbbox[0][1]=rbbox[0][1] / stride
        rbbox[0][2]=rbbox[0][2] / stride
        rbbox[0][3]=rbbox[0][3] / stride
        rbbox[0][4]=rbbox[0][4] / stride
        rotated_corners = obb2poly(rbbox,'le90')
        rotated_corners = rotated_corners.view(4,2)
        # 使用旋转后的顶点坐标生成多边形掩码
        mask = self.fill_polygon(rotated_corners, feature_map_size)

        return mask

    def fill_polygon(self,vertices, feature_map_size):
        """
        根据旋转后的多边形顶点生成掩码。
        
        Args:
            vertices (Tensor): 多边形的顶点坐标，形状为 (4, 2)
            feature_map_size (Tuple[int, int]): 特征图尺寸 (height, width)
        
        Returns:
            Tensor: 二值掩码，1 表示保留，0 表示被 mask。
        """
        h, w = feature_map_size
        y_grid, x_grid = torch.meshgrid(torch.arange(h, device=vertices.device), torch.arange(w, device=vertices.device))
        points = torch.stack([x_grid, y_grid], dim=-1).reshape(-1, 2) 
        mask = torch.ones(feature_map_size, dtype=torch.float32,device=vertices.device)
        # 5. 批量判断每个像素点是否在多边形内
        mask_inside = self.point_in_polygon_batch(points, vertices)
        # 6. 更新掩码：将位于旋转框内的像素点设为 0
        mask.view(-1)[mask_inside] = 0
        return mask
    
    def point_in_polygon_batch(self,points, vertices):
        """
        判断批量点是否在旋转框内。使用交叉乘积的方法来批量判断点是否在多边形内。
        
        Args:
            points (Tensor): 形状为 (num_points, 2) 的张量，表示要判断的点的坐标。
            vertices (Tensor): 形状为 (4, 2) 的张量，表示多边形的 4 个顶点。
        
        Returns:
            Tensor: 布尔张量，表示每个点是否在多边形内。
        """
        num_points = points.size(0)
        vertices_ext = torch.cat([vertices, vertices[:1]], dim=0)  # 闭合多边形 (5, 2)

        # 扩展点到 (num_points, 4, 2)，以便进行批量计算
        p1 = vertices_ext[:-1].unsqueeze(0).expand(num_points, -1, -1)
        p2 = vertices_ext[1:].unsqueeze(0).expand(num_points, -1, -1)
        
        # 计算点是否在边界的左侧
        cross_product = (p2[:, :, 0] - p1[:, :, 0]) * (points[:, 1].unsqueeze(1) - p1[:, :, 1]) - \
                        (p2[:, :, 1] - p1[:, :, 1]) * (points[:, 0].unsqueeze(1) - p1[:, :, 0])

        # 判断点是否在所有边的左侧（内点的标准）
        is_inside = torch.all(cross_product >= 0, dim=1)

        return is_inside
    def fuse_features(self,rot_feats,hori_feats):
        fused_feats = self.attention(rot_feats,hori_feats)
        return fused_feats
    
    def simple_test_bboxes(self,
                           x,
                           img_metas,
                           proposals,
                           rcnn_test_cfg,
                           rescale=False):

        rois = rbbox2roi(proposals)
        bbox_results = self.just_forward(x,rois)
        img_shapes = tuple(meta['img_shape'] for meta in img_metas)
        scale_factors = tuple(meta['scale_factor'] for meta in img_metas)

        bbox_pred = bbox_results['bbox_pred']
        num_proposals_per_img = tuple(len(p) for p in proposals)
        rois = rois.split(num_proposals_per_img, 0)
        cls_score = bbox_results['cls_score'].split(num_proposals_per_img, 0)

        # some detector with_reg is False, bbox_pred will be None
        if bbox_pred is not None:
            # the bbox prediction of some detectors like SABL is not Tensor
            if isinstance(bbox_pred, torch.Tensor):
                bbox_pred = bbox_pred.split(num_proposals_per_img, 0)
            else:
                bbox_pred = self.bbox_head.bbox_pred_split(
                    bbox_pred, num_proposals_per_img)
        else:
            bbox_pred = (None, ) * len(proposals)

        # apply bbox post-processing to each image individually
        det_bboxes = []
        det_labels = []
        for i in range(len(proposals)):
            det_bbox, det_label = self.bbox_head.get_bboxes(
                rois[i],
                cls_score[i],
                bbox_pred[i],
                img_shapes[i],
                scale_factors[i],
                rescale=rescale,
                cfg=rcnn_test_cfg)
            det_bboxes.append(det_bbox)
            det_labels.append(det_label)
        return det_bboxes, det_labels
    
    def roi2hbbox(self,rois):
        batch_inds = rois[:, 0].unsqueeze(1)
        rbboxes = rois[:, 1:]  # [cx, cy, w, h, angle]
        
        # Convert the rotated bounding boxes to horizontal bounding boxes
        hbboxes = obb2xyxy(rbboxes, 'le90')
        
        # Concatenate the batch indices back with the horizontal bounding boxes
        hbboxes_with_inds = torch.cat([batch_inds, hbboxes], dim=1)
        
        return hbboxes_with_inds     
    

class SelfAttention(nn.Module):
    def __init__(self, input_dim, heads):
        super(SelfAttention, self).__init__()
        self.num_heads = heads
        self.head_dim = input_dim // heads

        assert input_dim % heads == 0, "Input dimension must be divisible by the number of heads."

        # 线性层，用于生成 Q, K, V
        self.query = nn.Linear(input_dim, input_dim)
        self.key = nn.Linear(input_dim, input_dim)
        self.value = nn.Linear(input_dim, input_dim)

        # 条件注意力层
        self.condition_layer = nn.Linear(1, input_dim)  # 将面积比映射到特征维度
        
        # 最后输出层
        self.fc_out = nn.Linear(input_dim, input_dim)

    def forward(self, rot_feats,hori_feats,area_ratio):
       # Get the batch size and feature dimension
        batch_size = rot_feats.size(0)
        input_dim = rot_feats.size(1)  # 获取通道数 (原本的 256)

        # Flatten the spatial dimensions (7x7) to get (batch_size, input_dim, 49)
        rot_feats_flattened = rot_feats.view(batch_size, input_dim, -1).permute(0, 2, 1) # (batch_size, input_dim, 49)
        hori_feats_flattened = hori_feats.view(batch_size, input_dim, -1).permute(0, 2, 1)  # (batch_size, input_dim, 49) 

        # 根据面积比生成条件权重
        condition_weight = self.condition_layer(area_ratio.unsqueeze(-1))  # (batch_size, input_dim)
        condition_weight = condition_weight.unsqueeze(1)  # (batch_size, 1, input_dim) 以便与空间位置对齐

        # 调整后的水平特征
        hori_feats_adjusted = hori_feats_flattened * (1 + condition_weight)

        # Linear projections for Q, K, V
        Q = self.query(rot_feats_flattened)  # (batch_size, num_proposals, input_dim)
        K = self.key(hori_feats_adjusted)    # (batch_size, num_proposals, input_dim)
        V = self.value(hori_feats_adjusted)  # (batch_size, num_proposals, input_dim)

        # Reshape for multi-head attention
        Q = Q.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)  # (batch_size, num_heads, num_proposals, head_dim)
        K = K.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)  # (batch_size, num_heads, num_proposals, head_dim)
        V = V.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)  # (batch_size, num_heads, num_proposals, head_dim)

        # Scaled dot-product attention
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)  # (batch_size, num_heads, num_proposals, num_proposals)
        attention_weights = torch.softmax(attention_scores, dim=-1)  # Apply softmax to get attention probabilities

        # Apply attention weights to V
        attention_output = torch.matmul(attention_weights, V)  # (batch_size, num_heads, num_proposals, head_dim)

        # Concatenate heads and apply final linear transformation
        attention_output = attention_output.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.head_dim)  # (batch_size, num_proposals, input_dim)
        output = self.fc_out(attention_output)  # 输出经过线性层处理的结果
        output = output.permute(0, 2, 1)  # 先将维度变为 (batch_size, input_dim, 49)
        output = output.view(batch_size, input_dim, 7, 7)  # 再将其变为 (batch_size, input_dim, 7, 7)

        return output
