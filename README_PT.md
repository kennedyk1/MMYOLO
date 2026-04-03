# MMYOLO

Versao em ingles: [README.md](README.md)

Este repositorio mantem o clone local [`ultralytics/`](ultralytics/) intacto e implementa toda a logica multicanal dentro de [`MMYOLO/`](MMYOLO/).

O pacote `MMYOLO` atual suporta:

- datasets `.npy` multicanais normalizados
- subconjuntos de canais como `RGB`, `T`, `D`, `I`, `RGBT`, `RGBTDI`
- arquiteturas custom do YOLO26 com attention
- treino `raw N channels`
- treino com `N stems`, criando um stem leve por canal selecionado

## Estrutura principal

Pacote principal:

- [`MMYOLO/__init__.py`](MMYOLO/__init__.py)
- [`MMYOLO/factory.py`](MMYOLO/factory.py)
- [`MMYOLO/trainer.py`](MMYOLO/trainer.py)
- [`MMYOLO/dataset.py`](MMYOLO/dataset.py)
- [`MMYOLO/modeling.py`](MMYOLO/modeling.py)
- [`MMYOLO/custom_modules.py`](MMYOLO/custom_modules.py)

YAMLs customizados:

- [`MMYOLO/custom_models/yolo26_raw_channelattention.yaml`](MMYOLO/custom_models/yolo26_raw_channelattention.yaml)
- [`MMYOLO/custom_models/yolo26_raw_cbam.yaml`](MMYOLO/custom_models/yolo26_raw_cbam.yaml)
- [`MMYOLO/custom_models/yolo26_nstems_channelattention.yaml`](MMYOLO/custom_models/yolo26_nstems_channelattention.yaml)
- [`MMYOLO/custom_models/yolo26_nstems_cbam.yaml`](MMYOLO/custom_models/yolo26_nstems_cbam.yaml)

Exemplos:

- treino simples: [`train_example.py`](train_example.py)
- treino em lote: [`train_batch.py`](train_batch.py)
- CLI: [`MMYOLO/train.py`](MMYOLO/train.py)
- uso rapido: [`MMYOLO/example_usage.py`](MMYOLO/example_usage.py)
- documentacao do pacote: [`MMYOLO/README.md`](MMYOLO/README.md)

## Variantes com attention

O `MMYOLO` agora incorpora diretamente estas variantes:

- `raw N + ChannelAttention`
- `raw N + CBAM`
- `N stems + ChannelAttention`
- `N stems + CBAM`

Essas variantes sao registradas em runtime. Nenhum arquivo dentro de [`ultralytics/`](ultralytics/) e alterado.

## Instalacao

Crie ou ative o ambiente Python e depois instale as dependencias:

```bash
pip install -r requirements.txt
```

## Fluxo tipico

1. Gere o dataset `.npy` normalizado com [`download_dataset.py`](download_dataset.py).
2. Escolha um dos YAMLs em [`MMYOLO/custom_models/`](MMYOLO/custom_models/).
3. Rode um experimento com [`train_example.py`](train_example.py) ou varios com [`train_batch.py`](train_batch.py).

## Uso em Python

```python
from MMYOLO import MMYOLO

wrapper = MMYOLO(
    model="yolo26n.pt",
    architecture="yolo26_nstems_channelattention.yaml",
    dataset_type="RGBTDI",
    channel_order="RGBTDI",
)

data_yaml = wrapper.create_data_yaml(
    dataset_root="MID-3K-NPY",
    class_names=["person"],
)

wrapper.train(
    data=data_yaml,
    epochs=100,
    imgsz=640,
    batch=8,
    workers=4,
    device="0",
    pretrained=True,
    project="runs/mmyolo",
    name="rgbtdi_nstems_channelattention",
)
```

## Uso por CLI

```bash
python MMYOLO/train.py \
  --data MID-3K-NPY \
  --model yolo26n.pt \
  --architecture yolo26_raw_cbam.yaml \
  --dataset-type RGBTDI \
  --channel-order RGBTDI \
  --epochs 100 \
  --imgsz 640 \
  --batch 8 \
  --workers 4 \
  --device 0 \
  --pretrained
```

## Observacoes sobre pretrained weights

- arquiteturas raw reaproveitam pretrained parcialmente
- variantes `n-stems` costumam preservar mais pesos compativeis do que as variantes raw com attention inserida cedo
- todo o carregamento continua sendo feito externamente pelo `MMYOLO`, sem editar internals do Ultralytics
