# -*- coding: utf-8 -*-
"""
pyon — библиотека проекта «Нейросимвольная интерпретация ионограмм».

Изолированные модули (только чистые импорты, monkeypatching запрещён):
  pyon.canon          — каноническая решётка ВЗ (оси, классы), растеризация SAO → маска,
                        жёсткие ридауты foF2/h′ (единственный источник констант решётки)
  pyon.digi_formats   — парсеры форматов дигизонда: RSF/SBF (read_ionogram — pandas,
                        read_canon — векторизованный декодер в матрицу canon), SAO, EDP, DFT
  pyon.ion_report     — отчёт в стиле Ion2PNG, MUF(D) по секансу, сравнение с ARTIST
  pyon.oblique_synth  — физика синтеза НЗ: сферический секанс+Мартин, Бугер-трассировка,
                        растеризация масок, МПЧ-метки, O/X-компоненты
  pyon.manifest       — опись корпуса (пары сырьё+SAO, характеристики ARTIST, сплиты)
                        -> data/manifest.csv; пересобирать после докачки корпуса
  pyon.loader         — ПОТОКОВЫЕ torch-датасеты (декодирование на лету в воркерах;
                        ВЗ через digi_formats.read_canon ~10 мс/файл, НЗ-синтез из SAO);
                        полнокорпусные тензорные кэши в обучении не используются
  pyon.dataset_cache  — smoke-сборка малого фиксированного ВЗ-набора (--limit -> *_smoke)
  pyon.oblique_cache  — smoke-сборка малого фиксированного НЗ-набора
  pyon.validate       — SHACL/owlrl-валидация сцен по онтологии pyonto/ (гигиена -> замыкание
                        -> формы; регрессионный набор: python -m pyon.validate)
  pyon.models         — U-Net (перенос из iono_study.ipynb)
  pyon.logic          — логическая голова: релаксации P1–P4 (ВЗ) и S1/S2/P2 (НЗ), правила детача
  pyon.gates          — SHACL-гейт на инференсе: маска → сцена → validate_scene; gate_rate
  (E1/E4, ещё нет)    — pyon.renderer (нейрорендерер шума), pyon.training, pyon.tblog

Тесты: tests/ (pytest; `.venv/bin/python -m pytest tests -q`) — парсеры на образцах, физика,
логика, гейт, лоадер; запускать перед каждым full-прогоном (CLAUDE.md §2.6).

Сырые данные: data/;  онтология: pyonto/ (вход iono-all.ttl);  внешняя читалка: pynasonde_git/.
"""
