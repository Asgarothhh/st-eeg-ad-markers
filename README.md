# ST-Conformer на SEED

Программная часть научной работы [«Нейронные маркеры эффективности рекламы: интеграция ЭЭГ и глубокого обучения»](docs/Статья.pdf) (Ковальчук, Мархель; рук. Монтик Н. С.). Классифицируется эмоциональная валентность по многоканальной ЭЭГ: негатив / нейтрал / позитив.

Исходная точка — [EEG-Conformer](https://ieeexplore.ieee.org/document/9991178) ([Song et al., 2023](https://doi.org/10.1109/TNSRE.2022.3230250)): локальные признаки берутся свёрткой, глобальные зависимости — самовниманием. В оригинале после пространственной свёртки размерность электродов схлопывается, и Transformer видит только временную последовательность. Здесь сетка «электрод × время» сохраняется. Это прямо связано с гипотезой [тезисов](docs/Тезисы.pdf): внутреннее внимание должно учитывать и длинные зависимости по ролику, и взаимодействие зон коры, в том числе фронтальную асимметрию ([Davidson, 1992](https://doi.org/10.1016/0278-2626(92)90065-T); [Harmon-Jones, Gable, 2018](https://doi.org/10.1111/psyp.12879)).

## Данные

Используется пакет **[SEED](https://bcmi.sjtu.edu.cn/home/seed/)** ([Zheng, Lu, 2015](https://doi.org/10.1109/TAMD.2015.2431497)), каталог `data/seed/`. В сеть подаются не сырые отсчёты 200 Гц, а заранее вычисленные признаки differential entropy (DE) в неперекрывающихся окнах 1 с — обычный вход для работ на этом наборе.

Читается только `ExtractedFeatures_1s/` (15 субъектов, по три сессии, файлы `{id}_{дата}.mat`) и `label.mat`. Каталог `ExtractedFeatures_4s/` в обучении не участвует. Координаты 62 электродов — из `channel_62_pos.locs` (система 10–20).

Для ролика `k` берётся массив `de_LDS{k}` формы **(62, T, 5)**: каналы × секунды ролика (T ≈ 185–265) × полосы δ, θ, α, β, γ. Метки пятнадцати роликов одинаковы во всех сессиях: `+1`, `0`, `−1` (позитив, нейтрал, негатив); в классификаторе им соответствуют классы 0, 1, 2. Три сессии одного субъекта объединяются: окна с одним и тем же номером ролика попадают в одну часть выборки.

Ролик на обучение не подаётся целиком. Его нарезают скользящими фрагментами **60 с** с шагом **10 с** и переставляют оси в тензор **`[5 полос, 62 канала, 60 с]`**. Метка окна совпадает с меткой ролика. Для ролика длительностью около 235 с получается порядка 18 окон.

Среднее и стандартное отклонение считаются **только по обучающим окнам**, отдельно для каждой пары (полоса, канал), по осям «окно» и «время». Теми же μ и σ нормируются валидация и тест; статистики сохраняются в чекпоинте.

## Разбиение и обучение

Эксперимент **внутри субъекта**: для каждого из 15 человек строится своя модель, записи остальных не используются. Нумерация роликов следует протоколу [Zheng и Lu](https://doi.org/10.1109/TAMD.2015.2431497) (1–9 / 10–15), с тем отличием, что из девяти «обучающих» роликов два откладываются, чтобы ранняя остановка не опиралась на тест:

| Ролики | Назначение |
|---|---|
| 1–7 | обучение (градиент; по ним же μ, σ) |
| 8–9 | валидация (выбор эпохи) |
| 10–15 | тест (итоговое качество) |

Чекпоинт соответствует максимуму точности на валидации. Тест в журнал эпох записывается, на отбор весов не влияет. Если 12 эпох подряд валидация не улучшается, обучение останавливается.

Пакет — 64 окна, обучающая выборка перемешивается. В свёрточном блоке время дополнительно сжимается пулингом с шагом 2 (60 с → 30 токенов внимания).

Для карт внимания (`visualize`) в сеть идёт **весь ролик** `[1, 5, 62, T]`, нормированный μ и σ из чекпоинта, без нарезки на 60 с.

## Архитектура

Вход `X ∈ R^{B × 5 × 62 × 60}`.

1. **Свёрточный модуль.** Временная свёртка `1×3` по секундам (локальный контекст соседних окон DE), затем `1×1` к размерности эмбеддинга F = 40. BatchNorm, ELU, dropout 0,5. Средний пулинг по времени с шагом 2. Электроды **не** смешиваются ядром `(62, 1)` — в этом отличие от [Song et al. (2023)](https://doi.org/10.1109/TNSRE.2022.3230250). Выход: `B × 62 × T' × 40`, T' = 30.

2. **Позиции.** К признакам электрода добавляется проекция его координат 10–20 из `channel_62_pos.locs`. По времени — синусоидальная кодировка ([Vaswani et al., 2017](https://arxiv.org/abs/1706.03762)).

3. **Два блока осевого внимания** (схема pre-LN + остаток).
   - Пространственное Multi-Head Attention: токены — 62 электрода, отдельно в каждом временном срезе. Модель может сопоставить, например, F3 и F4 в одну и ту же секунду.
   - Временное Multi-Head Attention: токены — секундные окна, отдельно по каждому электроду. Связываются реакции в разных фрагментах ролика.
   - Позиционный MLP (GELU, расширение ×2).

   Восемь голов, dropout 0,5.

4. **Классификатор.** Среднее по осям электродов и времени (global average pooling), LayerNorm, линейный слой 40 → 64 → 3.

Веса внимания обоих модулей при `return_attn=True` записываются: по ним строятся карта «электрод × электрод», топография входящего внимания и кривая по секундам ролика.

Регуляризация: dropout 0,5, L2 (`weight_decay` 10⁻⁴), label smoothing 0,05, отсечение градиента.

## Как читать результаты

После `train` для каждого субъекта в `results/st/subject_XX/` появляются:

| Файл | Смысл |
|---|---|
| `best.pt` | веса на лучшей валидации, эпоха, μ и σ нормировки |
| `log.csv` | по эпохам: потери и точность на обучении, валидации и (справочно) тесте |
| `learning_curves.png` | те же кривые; вертикаль — эпоха, с которой взят чекпоинт |
| `confusion.png` | ошибки на **тесте** в этой эпохе |
| `metrics.json` | val / test, разрыв обуч.–вал., размеры выборок |
| `y_true.npy`, `y_pred.npy` | истинные и предсказанные метки теста |

Сводка по субъектам: `results/summary.csv`. Если разрыв «обучение − валидация» больше 0,15, `evaluate` помечает это как признак переобучения. Обобщающая цифра работы — тестовая точность (и macro-F1) в выбранной по валидации эпохе, усреднённая по 15 субъектам.

Карты внимания (`visualize`) пишутся в `results/attn/subject_XX_sessS_trialK/`: матрица 62×62, топография входящего внимания, кривая по секундам ролика, внимание во фронтальных парах (FAA). Индексы: сессия `S` = 0…2, ролик `K` с нуля.

## Структура каталога

```
EEG Conformer/
  README.md
  st_conformer/
    config.py       пути к data/seed, гиперпараметры
    channels.py     62 канала, FAA-пары, координаты из .locs
    dataset.py      de_LDS, окна, z-оценка по train, loaders
    model.py        ST-Conformer
    utils.py        устройство, зерно, сборка модели
    train.py        цикл по субъектам, ранняя остановка
    evaluate.py     проверка чекпоинта, сводная таблица
    visualize.py    карты внимания
    plots.py        русские подписи графиков
    requirements.txt
  results/
    st/subject_XX/  best.pt, log.csv, learning_curves.png, confusion.png
    attn/           карты внимания по роликам
    summary.csv
```

## Запуск

Окружение: conda `eeg-conformer`. Рабочая директория — каталог `EEG Conformer`.

```bash
conda activate eeg-conformer
cd "/home/asgaroth/University/4 курс/Научная работа/EEG Conformer"
```

Зависимости (если ставить заново):

```bash
pip install -r st_conformer/requirements.txt
```

Обучение всех 15 субъектов:

```bash
python -m st_conformer.train --subjects all
```

Один субъект или диапазон:

```bash
python -m st_conformer.train --subjects 1
python -m st_conformer.train --subjects 1-5
python -m st_conformer.train --subjects 1,3,8
```

Каталог признаков (по умолчанию уже `data/seed/ExtractedFeatures_1s`):

```bash
python -m st_conformer.train --subjects all --features-dir "../data/seed/ExtractedFeatures_1s"
```

Короткий прогон (две эпохи, субъект 1):

```bash
python -m st_conformer.train --quick
```

Число эпох и размер пакета:

```bash
python -m st_conformer.train --subjects 1 --epochs 40 --batch-size 32
```

Устройство (`cpu`, `cuda`):

```bash
python -m st_conformer.train --subjects all --device cuda
```

Проверка чекпоинта субъекта (точность на обучении, валидации и тесте, разрыв, матрица ошибок):

```bash
python -m st_conformer.evaluate --subject 1
```

Сводка по всем уже посчитанным `metrics.json`:

```bash
python -m st_conformer.evaluate --all
```

Карты внимания на ролике (сессия 0…2, ролик с нуля; нужен `results/st/subject_XX/best.pt`):

```bash
python -m st_conformer.visualize --subject 1 --session 0 --trial 0
python -m st_conformer.visualize --subject 1 --session 0 --trial 2
```

## Литература

1. [Zheng W. L., Lu B. L. Investigating critical frequency bands and channels for EEG-based emotion recognition with deep neural networks // IEEE Transactions on Autonomous Mental Development. 2015. Vol. 7, no. 3. P. 162–175.](https://doi.org/10.1109/TAMD.2015.2431497) Набор данных: [SEED (BCMI / SJTU)](https://bcmi.sjtu.edu.cn/home/seed/).
2. [Song Y., Zheng Q., Liu B., Gao X. EEG Conformer: convolutional transformer for EEG decoding and visualization // IEEE Transactions on Neural Systems and Rehabilitation Engineering. 2023. Vol. 31. P. 710–719.](https://doi.org/10.1109/TNSRE.2022.3230250) IEEE Xplore: [document/9991178](https://ieeexplore.ieee.org/document/9991178). Код: [eeyhsong/EEG-Conformer](https://github.com/eeyhsong/EEG-Conformer).
3. [Vaswani A. et al. Attention is all you need // Advances in Neural Information Processing Systems. 2017. Vol. 30.](https://arxiv.org/abs/1706.03762)
4. [Davidson R. J. Anterior cerebral asymmetry and the nature of emotion // Brain and Cognition. 1992. Vol. 20, no. 1. P. 125–151.](https://doi.org/10.1016/0278-2626(92)90065-T)
5. [Harmon-Jones E., Gable P. A. On the role of asymmetric frontal cortical activity in approach and withdrawal motivation: an updated review of the evidence // Psychophysiology. 2018. Vol. 55, no. 1. e12879.](https://doi.org/10.1111/psyp.12879)
6. Ковальчук А. И., Мархель А. Д. Нейронные маркеры эффективности рекламы: интеграция ЭЭГ и глубокого обучения, 2026: [статья](docs/Статья.pdf), [тезисы](docs/Тезисы.pdf).
