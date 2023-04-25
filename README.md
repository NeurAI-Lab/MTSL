# Multi-Task Structural Learning using Local Task Similarity induced Neuron Creation and Removal

OUTPUT_DIR: Directory to save output contents.<br />
DATA_DIR: Directory containing the datasets.<br />
MODEL_DIR: Directory containing the trained models.

### Baselines Training script:

To train the One model on Cityscapes dataset:

python train.py --batch-size 16 --workers 8 --data-folder /DATA_DIR/Cityscapes --crop-size 256 512 --checkname train_one_cs --config-file ./model_configs/cityscapes/one.yaml --epochs 80 --lr .0001 --output-dir OUTPUT_DIR --lr-strategy stepwise --lr-decay 60 70 --base-optimizer Adam --dataset cs

Other model configs can be found in 'model_configs' directory.


### MTSL Training Script:

To train MTSL on Cityscapes dataset:

python train_stubs.py --batch-size 16 --workers 8 --crop-size 256 512 --data-folder /DATA_DIR/Cityscapes --data-folder-1 /DATA_DIR/Cityscapes/leftImg8bit/train --dataset cs --checkname train_mtsl_cs --config-file ./model_configs/cityscapes/sep.yaml --epochs 80 --pretrained --copy-opt-state --lr .0001 --output-dir OUTPUT_DIR --lr-strategy stepwise --lr-decay 60 70 --base-optimizer Adam

### MTSL converged architectures:
The config files of converged MTSL architectures are provided in the model_configs folder with names "mtsl_a_1.yaml","mtsl_a_2.yaml" and "mtsl_a_3.yaml" representing 3 seeds.

### Eval models:

Models can be evaluated using --eval-only flag along with train script and using the --resume flag to provide the trained model.


### Test image corruptions:
CORRUPT_DATA_DIR: saved images for 15 corruptions at 5 severity levels using https://github.com/bethgelab/imagecorruptions

Within CORRUPT_DATA_DIR each corruption has its own folder and within each corruption folder there are 5 severity folders.

python image_corruptions.py --workers 8 --batch-size 8 --crop-size 320 448 --dataset nyuv2 --checkname test_nyuv2_corruptions --config-file ./model_configs/nyuv2/one.yaml --resume MODEL_DIR/model_latest_080.pth --output-dir OUTPUT_DIR --data-folder /DATA_DIR/NYUv2 --corrupted-data-path CORRUPT_DATA_DIR
