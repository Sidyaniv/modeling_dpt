# История экспериментов

## 6.03.2024

[ClearML](https://app.clear.ml/projects/51748cecb3eb4781bf011e39c6a71721/experiments/a08d8b11d10e473e97f5ef4a01fde564/output/execution)

### Замечания
- Обучение на двух датасетах сразу, Vaihingen и DFC2018. Так как домены датасетов разные, то модель не может выучит паттерны для более маленького датасета, аугментации не помогли значимо улучшить результаты. Конфиг и результаты эксперимента см. по ссылке.   

### Идеи на будущее
- Нужно "размазать" распределения каждого из датасетов путем качественных аугментаций или путем перекрашивания одного датасета под другой. Или использовать другой датасет для увеличения общей выборки 

## 12.03.2024

[ClearML](https://app.clear.ml/projects/51748cecb3eb4781bf011e39c6a71721/experiments/48c1c2fe3af34652be94abb0bd8229eb/output/execution)

### Замечания
- Обучение проиходило на датасете Vaihingen. Получил неплохие резульаты на Vaihingen, SILoss на валидации 68 

### Идеи на будущее
- Нужно "размазать" распределения каждого из датасетов путем качественных аугментаций или путем перекрашивания одного датасета под другой. Или использовать другой датасет для увеличения общей выборки.



## 15.03.2024

[ClearML](http://neuron:7771/projects/3cef04caeed2488ba1e6e11ec64b9902/experiments/2029f1fe90924221bc1011cb975806cb/info-output/debugImages?columns=selected&columns=type&columns=name&columns=tags&columns=status&columns=project.name&columns=users&columns=started&columns=last_update&columns=last_iteration&columns=parent.name&order=-last_iteration&filter=)

### Замечания
 - Провальная попытка изменить аугментации, библиотека albumentations перестала нормлаьно отрабатывать при обучении, аугментация применяется только к маске, а к исходному изображению аугментация не применяется. Проблема взята на фикс

## 19.03.2024

### Замечания
 - Проблема была в нампае, который кешировал изображение и не давал его изменитть при аугментации


## 21.03.2024

[ClearML](http://neuron:7771/projects/3cef04caeed2488ba1e6e11ec64b9902/experiments/f7670090669c47a8bd4c619d37d8606e/info-output/metrics/scalar?columns=selected&columns=type&columns=name&columns=tags&columns=status&columns=project.name&columns=users&columns=started&columns=last_update&columns=last_iteration&columns=parent.name&order=-last_update&filter=)

### Замечания
 - После 2 дней экспериментов с архитектурой ДПТ не достигнуто никакхи результатов, модель ведет себя странно, сваливатеся в какие-то пиксельные овраги. Основная проблема в том, что при увеличении LR  модель сваливается в (глобальный минимум?), где просто красит все нулевой высотой.
