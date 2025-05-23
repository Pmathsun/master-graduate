# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F

from mmrotate.core import rbbox2roi,obb2xyxy,obb2poly
from ..builder import (ROTATED_HEADS,build_roi_extractor)
from .rotate_standard_roi_head import RotatedStandardRoIHead
from mmcv.ops import DeformConv2d

@ROTATED_HEADS.register_module()
class OrientedFeatureRoIHead(RotatedStandardRoIHead):
    def __init__(self,
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
        super(OrientedFeatureRoIHead, self).__init__(
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
        self.double_feature=DoubleFeature(bbox_roi_extractor.out_channels)
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
        rot_area=rois[:,3]*rois[:,4]
        hori_area=(hbboxes[:,4]-hbboxes[:,2])*(hbboxes[:,3]-hbboxes[:,1])
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
    def fuse_features(self,rot_feats,hori_feats,area_ratio):
        fused_feats = self.double_feature(rot_feats,hori_feats,area_ratio)
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
    

class DoubleFeature(nn.Module):
    def __init__(self, input_dim):
        super(DoubleFeature, self).__init__()
        
        # 条件权重生成层
        self.condition_layer = nn.Sequential(
            nn.Linear(1, input_dim),
            nn.Sigmoid()
        )
                # 上下文卷积和特征卷积
        self.context_conv = nn.Conv2d(input_dim, input_dim, kernel_size=3, padding=1)
        self.offset_conv = nn.Conv2d(input_dim, 18, kernel_size=3, padding=1)  # 18个通道为3x3卷积的偏移量
        # 上下文卷积和特征卷积
        self.deform_conv = DeformConv2d(input_dim, input_dim, kernel_size=3, padding=1,im2col_step=2)

    def forward(self, rot_feats, hori_feats, area_ratio):
        # # Step 1: 计算偏移并应用可变形卷积
        # offset = self.offset_conv(hori_feats)  # 生成可变形卷积所需的偏移量
        # context_feats = self.deform_conv(hori_feats,offset)
        
        # # Step 3: 标准化旋转特征
        # rot_feats_norm = F.normalize(rot_feats, p=2, dim=1)
        # context_feats_norm = F.normalize(context_feats, p=2, dim=1)
        # # Step 5: 生成条件权重并调整形状
        # condition_weight = self.condition_layer(area_ratio.unsqueeze(-1)).view(-1, rot_feats.size(1), 1, 1)
        
        # # Step 6: 加权融合特征
        # fused_feats = rot_feats_norm + condition_weight * context_feats_norm
        
        # return fused_feats
                # Step 1: 提取上下文特征
        context_feats = self.context_conv(hori_feats)
    
        # Step 5: 生成条件权重并调整形状
        condition_weight = self.condition_layer(area_ratio.unsqueeze(-1)).view(-1, rot_feats.size(1), 1, 1)
        
        # Step 6: 加权融合特征
        fused_feats = rot_feats + condition_weight * context_feats
        
        return fused_feats