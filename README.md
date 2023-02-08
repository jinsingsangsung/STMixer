# main_nsml

This branch is to reproduce the training accuracy of the official code, so small edit was done to debug some errors in the original code, no significant change in the model structure was added.

![Screen Shot 2023-02-08 at 10 28 56 AM](https://media.oss.navercorp.com/user/36297/files/62f540a0-e4c4-40e0-af17-88b219444bd9)


# TubeR: Tubelet Transformer for Video Action Detection

This repo copied the supported code of [TubeR: Tubelet Transformer for Video Action Detection](https://openaccess.thecvf.com/content/CVPR2022/papers/Zhao_TubeR_Tubelet_Transformer_for_Video_Action_Detection_CVPR_2022_paper.pdf). 

```
# if running this code other than nsml, you can use docker image that contains pretrained models and files as well
docker run -it -v /data02/ava:/datasets  -v /home/jinsung/tuber:/tuber --gpus all --shm-size 32g --name tuber jinsingsangsung/tuber:1.2 /bin/bash 

# if running in nsml,
sh nsml_setup # brings pretrained models
sh nsml_setup_{gpu type} # brings AVA dataset to scratchpad

# example running command
python3 train_tuber_ava.py --config-file ./configuration/TubeR_CSN152_AVA21.yaml
```
# Reproduction result

| Dataset | Backbone | Backbone pretrained on | DETR pretrained on | #view | Original mAP | Reproduced mAP | config |
| :---: | :---: | :-----: | :-----: |  :---: | :----: | :---: | :---: |
| AVA 2.1 | CSN-50 | Kinetics-400 | *COCO*, AVA | 1 view | 27.2 |  *26.1* | [config](configuration/TubeR_CSN50_AVA21.yaml) |

Still have no idea where the 1% drop comes from. 

## Citing TubeR
```
@inproceedings{zhao2022tuber,
  title={TubeR: Tubelet transformer for video action detection},
  author={Zhao, Jiaojiao and Zhang, Yanyi and Li, Xinyu and Chen, Hao and Shuai, Bing and Xu, Mingze and Liu, Chunhui and Kundu, Kaustav and Xiong, Yuanjun and Modolo, Davide and others},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={13598--13607},
  year={2022}
}
```
