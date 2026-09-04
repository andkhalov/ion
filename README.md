---
type: readme
status: active
links:
  - CLAUDE.md
  - STATUS.md
  - DIARY.md
  - research/README.md
  - research/06-experiment-plan.md
  - pyonto/README.md
  - lit/MISSING.md
---
# Нейросимвольная интерпретация ионограмм (ВЗ → НЗ)

Исследовательский проект: онтология интерпретации ионограмм + нейросетевой автоскейлер,
цель — **автоскейлер ионограмм наклонного зондирования** (аналога ARTIST для НЗ не существует).
Задача текущего этапа и правила работы агента — **[CLAUDE.md](CLAUDE.md)**; текущее состояние —
**[STATUS.md](STATUS.md)**; журнал итераций — **[DIARY.md](DIARY.md)**.

## Структура (после рефакторинга 2026-09-04)

| папка | содержимое |
|---|---|
| [`pyon/`](pyon/__init__.py) | библиотека проекта: парсеры форматов дигизонда, физика синтеза НЗ (сферический секанс/Мартин/Бугер, O/X), сборщики кэшей, SHACL-валидация. Изолированные модули, чистые импорты, monkeypatching запрещён |
| [`pynasonde_git/`](pynasonde_git/) | сторонняя читалка pynasonde (копия GitHub main с нашими патчами SBF/частот/EOI/MPA) — используется только для сверки |
| [`pyonto/`](pyonto/README.md) | онтология (OWL 2 RL + SHACL), замкнутая директория: вход `iono-all.ttl`, формы `iono-shapes*.ttl`, эталоны `ground-*.ttl`. Включает переработанные материалы А. О. Щирого; переиспользует классы BFO 2020/IAO (явные декларации в ядре). Проверка: `python -m pyon.validate` |
| [`research/`](research/README.md) | документы двух волн исследования; актуальные: Э1 форматы+курс+постановка, Э2 методика синтеза НЗ и релаксации, Э3 план эксперимента с метриками |
| [`lit/`](lit/MISSING.md) | литература (PDF); соответствие «цитата → файл» и недостающее — `lit/MISSING.md` |
| [`figures/`](figures/) | все графические артефакты (обзорные картинки форматов, сравнения с ARTIST, образцы реальных НЗ Тромсё) |
| [`data/`](data/) | данные: образцы Щирого (`RSF-…`, `SBF-…`), корпус NOAA (`corpus/`, ~40 тыс. файлов, 15 ГБ), кэши (`corpus_cache/` 29 166 пар ВЗ; `oblique_cache/` 27 тыс. НЗ-масок) |

Корень: ноутбуки-прототипы с выполненными выходами (исторические, НЕ редактировать):
`ion.ipynb` → `prototype.ipynb` → `prototype2.ipynb` → `oblique.ipynb` → **`iono_study.ipynb`**
(итоговое объединённое исследование); `download_corpus.sh` — докачка корпуса NOAA (bash);
`local/` — временные файлы скриптов (вне git). Весь код текущего состояния — proof-of-concept
прототипной фазы: перед серверным экспериментом проходит ревизию (CLAUDE.md §2.12).

Все `*.md` проекта начинаются с YAML-шапки (`type`/`status`/`links`) — правило CLAUDE.md §2.13.

## Быстрый старт

```bash
python3 -m venv .venv                              # окружение ТОЛЬКО внутри проекта (см. CLAUDE.md §2.9)
PIP_CACHE_DIR=local/pip-cache .venv/bin/pip install -r requirements.txt   # torch под CUDA сервера
source .venv/bin/activate
python -m pyon.validate                            # регрессия онтологии: «ВСЕ ТЕСТЫ PASS» (18)
./download_corpus.sh smoke                         # 1 день JI91J в data/corpus (bash, возобновляемый)
python -m pyon.manifest --limit 200                # опись корпуса (smoke -> data/manifest_smoke.csv)
python - <<'PY'                                    # прочитать одну ионограмму
import sys; sys.path.insert(0, ".")
from pyon import digi_formats as dfm
pf, df = dfm.read_ionogram("data/RSF-samples-w-img-n-sao-n-dft/ionogram/JI91J_2022001000000.RSF")
print(pf.date, len(df), "точек")
PY
```

Репозиторий проекта — отдельный: https://github.com/andkhalov/ion. Служебные файлы агента
(`CLAUDE.md`, `DIARY.md`, `STATUS.md`), окружение `.venv/`, данные `data/` (кроме `data/manual/`),
логи `runs/`, литература `lit/` и `local/` — вне git (см. `.gitignore`); на сервере они есть.

## Данные-образцы (легенда исходной поставки Щирого)

`RSF-samples…` — RSF + SAO + картинки (+DFT), Хикамарка; `SBF-samples…` — SBF + SAO, Рим;
форматы описаны в [research/format_description.md](research/format_description.md) (Part II)
и спецификациях в `lit/` (Digisonde4D manual Annex 5C, ARTIST Tape Format, D256 16C).
