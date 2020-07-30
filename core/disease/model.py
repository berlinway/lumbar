from copy import deepcopy
from typing import Tuple
import torch
import torchvision.transforms.functional as tf
from ..structure import Study
from ..key_point import KeyPointModel
from ..data_utils import SPINAL_VERTEBRA_ID, SPINAL_VERTEBRA_DISEASE_ID, SPINAL_DISC_ID, SPINAL_DISC_DISEASE_ID


VERTEBRA_POINT_INT2STR = {v: k for k, v in SPINAL_VERTEBRA_ID.items()}
VERTEBRA_DISEASE_INT2STR = {v: k for k, v in SPINAL_VERTEBRA_DISEASE_ID.items()}
DISC_POINT_INT2STR = {v: k for k, v in SPINAL_DISC_ID.items()}
DISC_DISEASE_INT2STR = {v: k for k, v in SPINAL_DISC_DISEASE_ID.items()}


class DiseaseModelBase(torch.nn.Module):
    def __init__(self,
                 kp_model: KeyPointModel,
                 sagittal_size: Tuple[int, int],
                 num_vertebra_diseases=len(SPINAL_VERTEBRA_DISEASE_ID),
                 num_disc_diseases=len(SPINAL_DISC_DISEASE_ID)):
        super().__init__()
        self.sagittal_size = sagittal_size
        self.num_vertebra_diseases = num_vertebra_diseases             #2
        self.num_disc_disease = num_disc_diseases                      #5
        self.backbone = deepcopy(kp_model)

    @property
    def out_channels(self):
        return self.backbone.out_channels

    @property
    def num_vertebra_points(self):
        return self.backbone.num_vertebra_points

    @property
    def num_disc_points(self):
        return self.backbone.num_disc_point

    @property
    def kp_parameters(self):
        return self.backbone.kp_parameters

    @property
    def resnet_out_channels(self):
        return self.backbone.resnet_out_channels

    @staticmethod
    def _gen_annotation(study: Study, vertebra_coords, vertebra_scores, disc_coords, disc_scores) -> dict:
        """

        :param study:
        :param vertebra_coords: Nx2
        :param vertebra_scores: V
        :param disc_scores: Dx1
        :return:
        """
        z_index = study.t2_sagittal.instance_uids[study.t2_sagittal_middle_frame.instance_uid]
        point = []
        for i, (coord, score) in enumerate(zip(vertebra_coords, vertebra_scores)):
            vertebra = int(torch.argmax(score, dim=-1).cpu())
            point.append({
                'coord': coord.cpu().int().numpy().tolist(),
                'tag': {
                    'identification': VERTEBRA_POINT_INT2STR[i],
                    'vertebra': VERTEBRA_DISEASE_INT2STR[vertebra]
                },
                'zIndex': z_index
            })
        for i, (coord, score) in enumerate(zip(disc_coords, disc_scores)):
            disc = int(torch.argmax(score, dim=-1).cpu())
            point.append({
                'coord': coord.cpu().int().numpy().tolist(),
                'tag': {
                    'identification': DISC_POINT_INT2STR[i],
                    'disc': DISC_DISEASE_INT2STR[disc]
                },
                'zIndex': z_index
            })
        annotation = {
            'studyUid': study.study_uid,
            'data': [
                {
                    'instanceUid': study.t2_sagittal_middle_frame.instance_uid,
                    'seriesUid': study.t2_sagittal_middle_frame.series_uid,
                    'annotation': [
                        {
                            'data': {
                                'point': point,
                            }
                        }
                    ]
                }
            ]
        }
        return annotation

    def get_one_hot(self, label, N):
        size = list(label.size())
        label = label.view(-1)              # reshape 为向量
        ones = torch.sparse.torch.eye(N)
        ones = ones.index_select(0, label)  # 用上面的办法转为换one hot
        size.append(N)  # 把类别输目添到size的尾后，准备reshape回原来的尺寸
        return ones.view(*size)

    def _train(self, sagittals, _, distmaps, v_labels, d_labels, v_masks, d_masks, t_masks) -> tuple:
        masks = torch.cat([v_masks, d_masks], dim=-1)
        v_labels_ctg = v_labels[:, :, 2]
        d_labels_ctg = d_labels[:, :, 2]
        # vertebar_labels = self.get_one_hot(v_labels_ctg, 2).view(-1, 10).cuda()
        # dics_labels = self.get_one_hot(d_labels_ctg, 5).view(-1, 30).cuda()
        vertebar_labels = self.get_one_hot(v_labels_ctg, 2).cuda()
        dics_labels = self.get_one_hot(d_labels_ctg, 5).cuda()
        return self.backbone(sagittals, distmaps=distmaps, masks=masks, v_labels=vertebar_labels, d_labels=dics_labels)
        #return self.backbone(sagittals, distmaps, masks)

    def _inference(self, study: Study, to_dict=False):
        kp_frame = study.t2_sagittal_middle_frame
        # 将图片放缩到模型设定的大小
        sagittal = tf.resize(kp_frame.image, self.sagittal_size)
        sagittal = tf.to_tensor(sagittal).unsqueeze(0)

        v_coord, d_coord, _, feature_maps, v_scores, d_scores = self.backbone(sagittal, return_more=True)
        v_scores = v_scores.view(-1, 5, 2)
        d_scores = d_scores.view(-1, 6, 5)
        v_scores = torch.nn.functional.softmax(v_scores, dim=-1)
        d_scores = torch.nn.functional.softmax(d_scores,dim=-1)
        # 将预测的坐标调整到原来的大小，注意要在extract_point_feature之后变换
        height_ratio = self.sagittal_size[0] / kp_frame.size[1]
        width_ratio = self.sagittal_size[1] / kp_frame.size[0]
        ratio = torch.tensor([width_ratio, height_ratio], device=v_coord.device)

        v_coord = (v_coord.float() / ratio).round()[0]
        d_coord = (d_coord.float() / ratio).round()[0]
        v_index = torch.argmax(v_scores, dim=-1)
        d_index = torch.argmax(d_scores, dim=-1)
        v_score = torch.zeros(v_coord.shape[0], self.num_vertebra_diseases)
        # v_score[:, 1] = 1
        for i in range(5):
            v_score[i, v_index[0, i]] = 1

        d_score = torch.zeros(d_coord.shape[0], self.num_disc_disease)
        # d_score[:, 0] = 1
        for i in range(6):
            d_score[i, d_index[0, i]] = 1
        if to_dict:
            return self._gen_annotation(study, v_coord, v_score, d_coord, d_score)
        else:
            return v_coord, v_score, d_coord, d_score

    def forward(self, *args, **kwargs):
        if self.training:
            return self._train(*args, **kwargs)
        else:
            return self._inference(*args, **kwargs)
