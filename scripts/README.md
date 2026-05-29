# Scripts

## `finetune_classification.py`

Fine-tunes Hugging Face image-classification backbones on local plant disease image-folder datasets.

Example:

```bash
python scripts/finetune_classification.py --model swin --dataset PlantVillage --data-root datasets --epochs 10 --batch-size 16
```

Run every supported model-dataset pair:

```bash
python scripts/finetune_classification.py --model all --dataset all --data-root datasets --epochs 10 --batch-size 16
```

Supported models: `swin`, `vit`, `resnet`, `convnext`, and `all`.
Supported datasets: `PlantVillage`, `FieldPlant`, `PlantDoc`, `CroppedPlantDoc`, `Cropped PlantDoc`, and `all`.
