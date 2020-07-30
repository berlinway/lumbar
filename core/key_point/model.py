import torch
from torch.nn.functional import interpolate
from torchvision.models.detection.backbone_utils import BackboneWithFPN
from .loss import KeyPointBCELoss
from .spinal_model import SpinalModelBase
from ..data_utils import SPINAL_VERTEBRA_ID, SPINAL_DISC_ID


class KeyPointModel(torch.nn.Module):
    def __init__(self, backbone: BackboneWithFPN, num_vertebra_points: int = len(SPINAL_VERTEBRA_ID),
                 num_disc_points: int = len(SPINAL_DISC_ID), pixel_mean=0.5, pixel_std=1,
                 loss=KeyPointBCELoss(), spinal_model=SpinalModelBase()):
        super().__init__()
        self.backbone = backbone
        self.num_vertebra_points = num_vertebra_points
        self.num_disc_point = num_disc_points
        self.fc = torch.nn.Conv2d(backbone.out_channels, num_vertebra_points + num_disc_points, kernel_size=1)
        self.register_buffer('pixel_mean', torch.tensor(pixel_mean))
        self.register_buffer('pixel_std', torch.tensor(pixel_std))
        self.spinal_model = spinal_model
        self.classeifie_loss = torch.nn.BCEWithLogitsLoss()
        self.loss = loss
        self.inter_layer1 = torch.nn.Sequential(
                  torch.nn.Conv2d(256, 1, kernel_size=3, padding=1 ),
                  torch.nn.ReLU()
        )

        self.classifie_vertebra = torch.nn.Linear(1*128*128, 10)    # vertebra:5*2
        self.classifie_vertebra.weight.requires_grad = False
        self.classeifie_disc = torch.nn.Linear(1*128*128, 30)       # disc:6*5
        self.classeifie_disc.weight.requires_grad = True
        self.fc.weight.requires_grad = False
        for param in self.inter_layer1.parameters():
            param.requires_grad = True
    @property
    def out_channes(self):
        return self.backbone.out_channels

    @property
    def resnet_out_channels(self):
        return self.backbone.fpn.inner_blocks[-1].in_channels

    def kp_parameters(self):
        for p in self.fc.parameters():
            yield p

    def set_spinal_model(self, spinal_model: SpinalModelBase):
        self.spinal_model = spinal_model

    def _preprocess(self, images: torch.Tensor) -> torch.Tensor:
        images = images.to(self.pixel_mean.device)
        ages = (images - self.pixel_mean) / self.pixel_std
        images = images.expand(-1, 3, -1, -1)
        return images

    def cal_vertebra(self, feature_map):
        feature = self.inter_layer1(feature_map)
        x = torch.flatten(feature, start_dim=1)
        output = self.classifie_vertebra(x)
        return output

    def cal_disc(self, feature_map):
        feature = self.inter_layer1(feature_map)
        x = torch.flatten(feature, start_dim=1)
        output = self.classeifie_disc(x)
        return output

    def cal_scores(self, images, if_clf=False):
        images = self._preprocess(images)
        feature_pyramids = self.backbone(images)
        feature_maps = feature_pyramids['0']
        # x = feature_maps.shape
        scores = self.fc(feature_maps)
        scores = interpolate(scores, images.shape[-2:], mode='bilinear', align_corners=True)
        if if_clf:
            v_scores = self.cal_vertebra(feature_maps)
            d_scores = self.cal_disc(feature_maps)
            return scores, feature_maps, v_scores, d_scores
        else:
            return scores, feature_maps

    def cal_backbone(self, images: torch.Tensor) -> torch.Tensor:
        images = self._preprocess(images)
        output = self.backbone.body(images)
        return list(output.values())[-1]

    def pred_coords(self, scores, v_scores=None, d_scores=None, split=True):
        heat_maps = scores.sigmoid()
        coords = self.spinal_model(heat_maps)
        if split:
            vertebra_coords = coords[:, :self.num_vertebra_points]
            disc_coords = coords[:, self.num_vertebra_points:]
            return vertebra_coords, disc_coords, heat_maps
        else:
            return coords, heat_maps

    def forward(self, images, distmaps=None, masks=None, v_labels=None, d_labels=None, return_more=False) -> tuple:
    #def forward(self, images, distmaps=None, masks=None, return_more=False) -> tuple:
        scores, feature_maps, v_scores, d_scores = self.cal_scores(images, True)
        if self.training:
            if distmaps is None:
                loss = None
            else:
                loss = self.loss(scores, distmaps, masks)
            if v_scores is not None:
                v_scores = v_scores.view(-1, 5, 2)
                d_scores = d_scores.view(-1, 6, 5)
                classifies_loss_1 = self.classeifie_loss(v_scores, v_labels)
                classifies_loss_2 = self.classeifie_loss(d_scores, d_labels)

            if return_more:
                vertebra_coords, disc_coords, heat_maps = self.pred_coords(scores)
                return loss, vertebra_coords, disc_coords, heat_maps, feature_maps
            else:
                return classifies_loss_2,
                #return lss,
        else:
            vertebra_coords, disc_coords, heat_maps = self.pred_coords(scores)           #add by lq 2020.7.23
            # v_scores = v_scores.softmax(dim=1)
            # d_scores = d_scores.softmax(dim=1)
            if return_more:
                return vertebra_coords, disc_coords, heat_maps, feature_maps, v_scores, d_scores
            else:
                return vertebra_coords, disc_coords
