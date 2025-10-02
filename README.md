# modeling-mask2former


## Быстрый запуск:
- ```make install``` - установка зависимостей (см. requirements.txt)
- ```make train``` - старт обучения

## Настройка обучения и гиперпарметры:
- [Файл конфигурации](./configs/config.yaml)

## Трекинг модели
- ClearML: 

## Модели

<table style="margin: auto">
  <thead>
    <tr>
      <th>Расположение в shared_data</th>
      <th>Обучение<br />на датасете</th>
      <th>Метрика<br />MSE</th>
      <th>Ссылка<br />на эксперимент</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td align="right">/../../shared_data/SatelliteTo3D-Models/dpt/dfc2018</td>
      <td align="right">DFC2018</td>
      <td align="right">2.13</td>
      <td align="right"><a href="">ClearML</a></td>
    </tr>
  </tbody>
</table>

 ## Пример использования

```python
import torch
from transformers import DPTForDepthEstimation
from predict_utils import make_prediction_dpt

dummy_tensor = torch.empty((1, 3, 512, 512), dtype=torch.float32)

model_path = '/../../shared_data/SatelliteTo3D-Models/dpt/dfc2018'


model = DPTForDepthEstimation.from_pretrained(
            pretrained_model_name_or_path=model_path,
        )
preds = make_prediction_dpt(model, dummy_tensor)

```
