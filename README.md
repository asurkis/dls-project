# Проект DLS 1 семестр осень 2020 image generation.

Суркис Антон (Stepik id: [83694640](https://stepik.org/users/83694640))

Мне удалось сделать только 1 пункт &mdash; повторить существующее решение
из [статьи о pix2pix](https://arxiv.org/pdf/1611.07004.pdf)
[пример из статьи](https://colab.research.google.com/github/tensorflow/docs/blob/master/site/en/tutorials/generative/pix2pix.ipynb#scrollTo=Kn-k8kTXuAlv)
с существующей архитектурой модели, но моей реализацией.
Придумать другую задачу и найти подходящие датасеты мне не удалось.

Из-за отсутствия собственных вычислительных мощностей повторял решение в Google Colab,
после этого портировал в отдельные Python-файлы,
но работоспособность самого тренировочного цикла проверить не на чем.

Скрипт для загрузки датасета: `bin/dataset_download.sh` (запускать из корневой директории проекта).

Только весов после 300 эпох обучения: `bin/model_download_300.sh`.

Запуск самой сети: `python3 src/train.py`.<br>
Возможные аргументы:
- `-t`/`--train` &mdash; обучать нейросеть, а не использовать для вывода результатов;
- `-e`/`--epoch <количество эпох>` &mdash; задать количество эпох (также работает и при загрузке предобученных сетей, но нужно убедиться, что соответствующий слепок существует). По-умолчанию 300;
- `-d`/`--dataset <датасет>` &mdash; задать имя датасета. По умолчанию `facades`. Поиск файлов будет по пути `src/dataset/<датасет>/(test|train|val)/**.jpg`.

[Этот код в Google Colab](https://colab.research.google.com/drive/1oXobmdumJuvfxPpjZwfr2dyi84Gn0Ueg?usp=sharing)

История обучения сети:<br>![](report/history.png)

Как можно заметить, проблему переобучения дискриминатора решить не удалось,
поэтому генератор также будет переобучен.

Примеры результатов выполнения на валидационном датасете
(слева &mdash; разметка, по центру &mdash; реальное изображение,
справа &mdash; сгенерированный фасад):

10 эпох:<br>![](report/010.png)

20 эпох:<br>![](report/020.png)

30 эпох:<br>![](report/030.png)

40 эпох:<br>![](report/040.png)

50 эпох:<br>![](report/050.png)

60 эпох:<br>![](report/060.png)

70 эпох:<br>![](report/070.png)

80 эпох:<br>![](report/080.png)

90 эпох:<br>![](report/090.png)

100 эпох:<br>![](report/100.png)

110 эпох:<br>![](report/110.png)

120 эпох:<br>![](report/120.png)

130 эпох:<br>![](report/130.png)

140 эпох:<br>![](report/140.png)

150 эпох:<br>![](report/150.png)

160 эпох:<br>![](report/160.png)

170 эпох:<br>![](report/170.png)

180 эпох:<br>![](report/180.png)

190 эпох:<br>![](report/190.png)

200 эпох:<br>![](report/200.png)

210 эпох:<br>![](report/210.png)

220 эпох:<br>![](report/220.png)

230 эпох:<br>![](report/230.png)

240 эпох:<br>![](report/240.png)

250 эпох:<br>![](report/250.png)

260 эпох:<br>![](report/260.png)

270 эпох:<br>![](report/270.png)

280 эпох:<br>![](report/280.png)

290 эпох:<br>![](report/290.png)

300 эпох:<br>![](report/300.png)
