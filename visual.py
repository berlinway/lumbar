import json
import sys
import time

import torch
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone


import matplotlib.pyplot as plt
import cv2
from core.disease.data_loader import DisDataLoader
from core.disease.evaluation import Evaluator
from core.disease.model import DiseaseModelBase
from core.key_point import KeyPointModel, NullLoss
from core.structure import construct_studies

sys.path.append('./nn_tools/')
from nn_tools import torch_utils


if __name__ == '__main__':
    # start_time = time.time()
    # train_studies, train_annotation, train_counter = construct_studies(
    #     'data/lumbar_train150', 'data/lumbar_train150_annotation.json', multiprocessing=True)
    # valid_studies, valid_annotation, valid_counter = construct_studies(
    #     'data/lumbar_train51/train/', 'data/lumbar_train51_annotation.json', multiprocessing=True)

    # # 设定模型参数
    # train_images = {}
    # for study_uid, study in train_studies.items():
    #     frame = study.t2_sagittal_middle_frame
    #     # train_images[(study_uid, frame.series_uid, frame.instance_uid)] = frame.image

    backbone = resnet_fpn_backbone('resnet50', True)
    kp_model = KeyPointModel(backbone)   #点的预测
    dis_model = DiseaseModelBase(kp_model, sagittal_size=(512, 512))
    dis_model.cuda()
    print(dis_model)

    # # 设定训练参数
    # train_dataloader = DisDataLoader(
    #     train_studies, train_annotation, batch_size=8, num_workers=3, num_rep=10, prob_rotate=1, max_angel=180,
    #     sagittal_size=dis_model.sagittal_size, transverse_size=dis_model.sagittal_size, k_nearest=0
    # )

    # valid_evaluator = Evaluator(
    #     dis_model, valid_studies, 'data/lumbar_train51_annotation.json', num_rep=20, max_dist=6,
    # )

    # step_per_batch = len(train_dataloader)
    # optimizer = torch.optim.AdamW(dis_model.parameters(), lr=1e-5)
    # max_step = 20 * step_per_batch
    # fit_result = torch_utils.fit(
    #     dis_model,
    #     train_data=train_dataloader,
    #     valid_data=None,
    #     optimizer=optimizer,
    #     max_step=max_step,
    #     loss=NullLoss(),
    #     metrics=[valid_evaluator.metric],
    #     is_higher_better=True,
    #     evaluate_per_steps=step_per_batch,
    #     evaluate_fn=valid_evaluator,
    # )

    # torch.save(dis_model.cpu().state_dict(), 'models/baseline_072101.dis_model')
    # 预测
    dis_model.load_state_dict(torch.load('models/baseline_072302.dis_model'))
    testA_studies = construct_studies('data/data/lumbar_testA50/')

    result = []
    for study in testA_studies.values():
        vis = dis_model.eval()(study, True)
        image = study.t2_sagittal_middle_frame.image

        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)

        print(image.size,type(image))
        for point in vis['data'][0]['annotation'][0]['data']['point']:
            coord = point['coord']
            # image[coord[0]-5:coord[0]+5,coord[1]-5:coord[1]+5,:] = 0


            top_left_x, top_left_y = coord[0]-2, coord[1]-2
            width, height = 4, 4 # 使目标点在正中间
            rect = plt.Rectangle((top_left_x, top_left_y), width, height, fill=False, edgecolor='red', linewidth=1)
            ax.add_patch(rect)
        plt.imshow(image,cmap='gray')
        plt.show()



        result.append(dis_model.eval()(study, True))

    # with open('predictions/baseline.json', 'w') as file:
    #     json.dump(result, file)
    # print('task completed, {} seconds used'.format(time.time() - start_time))
