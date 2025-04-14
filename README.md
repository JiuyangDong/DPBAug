# DPBAug
The official implementation of "Disentangled Pseudo-bag Augmentation for Whole Slide Image Multiple Instance Learning"

## training & validation & testing
#### step1: Pre-train
```shell
python run.py --task pretrain --gpu_id 0 --feature_extractor ResNet50_ImageNet --fold 10 --label_frac 1.00 --dataset TCGA-NSCLC \
  --main_model ABMIL --aux_model Inst-Max --lr 3e-4 --patch_ratio 0.1
```

#### step2: Fine-tune
```shell
python run.py --task dpbaug --gpu_id 0 --feature_extractor ResNet50_ImageNet --fold 10 --label_frac 1.00 --dataset TCGA-NSCLC \
    --main_model ABMIL --aux_model Inst-Max --lr 3e-4 --patch_ratio 0.5 \
    --pretrain_path experiments/stage-pretrain/TCGA-NSCLC-20x-10fold/ResNet50_ImageNet/ABMIL+Inst-Max/label_frac=1.0/patch_ratio=0.5/ckpts/
```
