# modeling-mask2former


## Быстрый запуск:
- ```make install``` - установка зависимостей (см. requirements.txt)
- ```make train``` - старт обучения

## Настройка обучения и гиперпарметры:
- [Файл конфигурации](./configs/config.yaml)

## Трекинг модели
- ClearML: [dpt](http://neuron:7771/projects/3cef04caeed2488ba1e6e11ec64b9902/experiments/558fb375ca864ef1af97d085c28af8be/info-output/metrics/scalar?columns=selected&columns=type&columns=name&columns=tags&columns=status&columns=project.name&columns=users&columns=started&columns=last_update&columns=last_iteration&columns=parent.name&order=-last_update&filter=)
- [История экспериментов](./experiments.md)

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
      <td align="right"><a href="http://neuron:7771/projects/3cef04caeed2488ba1e6e11ec64b9902/experiments/e89f6ab71ea34191b7b540796ec7b03e/info-output/metrics/scalar?columns=selected&columns=type&columns=name&columns=tags&columns=status&columns=project.name&columns=users&columns=started&columns=last_update&columns=last_iteration&columns=parent.name&order=-last_update&filter=">ClearML</a></td>
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
