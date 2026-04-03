# Pacote MMYOLO

Este pacote estende o clone local do Ultralytics com treino multicanal em `.npy` e arquiteturas custom do YOLO26 com attention, mantendo [`../ultralytics/`](../ultralytics/) intacto.

## API publica

Pontos principais:

- [`MMYOLO`](__init__.py)
- [`create_multichannel_yolo`](factory.py)
- [`train_multichannel_yolo`](factory.py)
- [`write_detection_data_yaml`](factory.py)

Registro em runtime:

- [`register_attention_modules`](modeling.py)
- [`ChannelSlice`](custom_modules.py)
- [`MultiInputStem`](custom_modules.py)
- `ChannelAttention`, `SpatialAttention`, `CBAM`

## YAMLs custom do YOLO26

- [`custom_models/yolo26_raw_channelattention.yaml`](custom_models/yolo26_raw_channelattention.yaml)
- [`custom_models/yolo26_raw_cbam.yaml`](custom_models/yolo26_raw_cbam.yaml)
- [`custom_models/yolo26_nstems_channelattention.yaml`](custom_models/yolo26_nstems_channelattention.yaml)
- [`custom_models/yolo26_nstems_cbam.yaml`](custom_models/yolo26_nstems_cbam.yaml)

## Como usar

```python
from MMYOLO import MMYOLO

wrapper = MMYOLO(
    model="yolo26n.pt",
    architecture="yolo26_nstems_cbam.yaml",
    dataset_type="RGBTDI",
    channel_order="RGBTDI",
)
```

Se `architecture=None`, o pacote continua funcionando como wrapper multicanal da arquitetura base do YOLO.  
Se `architecture` apontar para um dos YAMLs acima, o grafo custom com attention sera usado.

## Observacoes

- o pacote preserva nomes compativeis como `create_multichannel_yolo` e `train_multichannel_yolo`
- os blocos de attention sao registrados apenas no processo Python atual
- nenhum arquivo dentro de `ultralytics/` e modificado
