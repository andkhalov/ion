#!/usr/bin/env bash
# Скачивание обучающего корпуса «сырые ионограммы + SAO» из открытого архива NOAA NCEI.
# Архив: .../data/<URSI>/<год>/individual/<день>/{ionogram,scaled,image}/
# Качаем только ionogram/ (RSF/SBF/MMM/16C) и scaled/ (SAO/EDP); image/ и drift/ пропускаем.
# Повторный запуск дотягивает только недостающие файлы. Пустые дни пропускаются молча.
# Портировано с zsh на bash 2026-09-04 (на сервере zsh нет); временные файлы — в local/tmp
# (правило замкнутости CLAUDE.md §2.9: никаких записей в /tmp).
#
#   ./download_corpus.sh          — P0 (~7 ГБ: MO155-2019, JI91J-2022, JR055-2022, PQ052-2022)
#   ./download_corpus.sh p1       — P1: все станция-годы с ПОДТВЕРЖДЁННЫМ сырьём (проба
#                                    2026-09-05): MO155 2012/2018/2019, JI91J 2019/2020/2022,
#                                    JR055 2012/2019/2020/2022, PQ052 2019/2022, RO041 2022
#                                    (SBF). Оценка: ~45-55 ГБ сырья.
#   ./download_corpus.sh p2       — P2: кандидаты 2013-2014 + RO041 2019-2021 (сырьё не
#                                    проверялось поимённо; пустые дни скрипт пропустит).
#   ./download_corpus.sh smoke    — один день для проверки
#   ./download_corpus.sh p3       — P3: ВЫСОКИЕ ШИРОТЫ (проба 2026-09-04 на сервере, 24 кода,
#                                    по 4 дня на каждый год): TR169 Тромсё 69.6N (RSF 2007-2015,
#                                    DPS-4), EI764 Eielson 64.7N (RSF 2013-2023), GA762 Gakona
#                                    62.4N (RSF 2003-2024), NO369 Норильск 69.3N (RSF 2010-2013),
#                                    YA462 Якутск 62N (RSF/SBF 2010-2017). Выбраны годы по фазам
#                                    цикла: min 23/24 (TR169-2009), max 24 (TR169-2013, GA762-2012,
#                                    NO369-2011, YA462-2011), min 24/25 (EI764-2019), max 25
#                                    (EI764-2023, GA762-2024). Оценка: 8 станция-лет, ~60-90 ГБ
#                                    (у EI764/GA762 с 2020 г. — 192 файла/сутки).
#   ./download_corpus.sh year ST YYYY — произвольный станция-год (напр. year SMJ67 2010)
#   PAR=16 ./download_corpus.sh p1   — число параллельных загрузок (по умолчанию 8)
# ФАКТЫ (проба 2026-09-05 с локальной машины + проба 2026-09-04 на сервере, research-дисциплина):
# (1) TR170 (Тромсё, новый код), SO166, KI167, MM168, MG560, SD266, EB040, EA036, AT138,
#     MO155-2022 — только scaled/. (2) ОПРОВЕРГНУТО «высокоширотных станций с сырьём нет»:
#     сырьё RSF/SBF есть у TR169 (старый код Тромсё!), EI764, GA762, NO369, YA462, SMJ67
#     (Sondrestrom 67N, SBF 2000-2012); формат MMM (Digisonde-256, парсера у нас нет) — THJ77,
#     NQJ61, CO764, KS759. (3) ОПРОВЕРГНУТО «сырья за 2023-2024 нет ни у кого»: GA762 2023-2024
#     и EI764 2023 — RSF по 192 файла/сутки. Полный набор E0 = smoke -> p1 -> p3 -> p2;
#     повторный запуск дотягивает недостающее. После каждой докачки: python -m pyon.manifest
set -u
BASE="https://data.ngdc.noaa.gov/instruments/remote-sensing/active/profilers-sounders/ionosonde/data"
HERE="$(cd "$(dirname "$0")" && pwd)"
OUT="$HERE/data/corpus"
TMPDIR_LOCAL="$HERE/local/tmp"; mkdir -p "$TMPDIR_LOCAL"
PAR=${PAR:-8}   # параллельных загрузок (переопределяется окружением: PAR=16 ./download_corpus.sh p1)

P0=( MO155:2019 JI91J:2022 JR055:2022 PQ052:2022 )
P2=( MO155:2013 MO155:2014 JI91J:2013 JI91J:2014
     JR055:2013 JR055:2014 PQ052:2013 PQ052:2014
     RO041:2019 RO041:2020 RO041:2021 )
P1=( MO155:2012 MO155:2018 MO155:2019
     JI91J:2019 JI91J:2020 JI91J:2022
     JR055:2012 JR055:2019 JR055:2020 JR055:2022
     PQ052:2019 PQ052:2022
     RO041:2022 )
P3=( TR169:2009 TR169:2013
     EI764:2019 EI764:2023
     GA762:2012 GA762:2024
     NO369:2011 YA462:2011 )

list_files() {  # $1 = URL каталога → имена файлов нужных расширений (по строке)
  curl -sf -m 60 "$1" | grep -o 'href="[A-Za-z0-9_]*\.\(RSF\|SBF\|MMM\|16C\|SAO\|EDP\)"' | sed 's/href="//;s/"$//'
}

fetch_day() {   # $1 станция, $2 год, $3 день (001..366) → curl-config недостающих файлов в stdout
  local st=$1 yr=$2 dd=$3 sub url dst files f
  for sub in ionogram scaled; do
    url="$BASE/$st/$yr/individual/$dd/$sub/"
    dst="$OUT/$st/$yr/$dd/$sub"
    files=$(list_files "$url") || continue
    [[ -z "$files" ]] && continue
    mkdir -p "$dst"
    while read -r f; do
      [[ -z "$f" || -s "$dst/$f" ]] && continue
      printf 'url = "%s"\noutput = "%s"\n' "$url$f" "$dst/$f"
    done <<< "$files"
  done
}

download() {    # $1 = curl-config файл
  local n; n=$(grep -c '^url' "$1")
  echo "  к скачиванию: $n файлов"
  [[ "$n" -gt 0 ]] && curl -sf --retry 3 -m 300 --parallel --parallel-max $PAR --config "$1"
  return 0
}

run_year() {    # $1 станция, $2 год
  local st=$1 yr=$2 tmp days dd
  tmp=$(mktemp "$TMPDIR_LOCAL/curl.XXXXXX")
  echo "=== $st $yr"
  days=$(curl -sf -m 60 "$BASE/$st/$yr/individual/" | grep -o 'href="[0-9][0-9][0-9]/"' | tr -dc '0-9\n')
  while read -r dd; do
    [[ -z "$dd" ]] && continue
    fetch_day "$st" "$yr" "$dd"
  done <<< "$days" >> "$tmp"
  download "$tmp"; rm -f "$tmp"
  echo "  итого: $(find "$OUT/$st/$yr" -type f 2>/dev/null | wc -l | tr -d ' ') файлов, $(du -sh "$OUT/$st/$yr" 2>/dev/null | cut -f1)"
}

case "${1:-p0}" in
  smoke) tmp=$(mktemp "$TMPDIR_LOCAL/curl.XXXXXX"); fetch_day JI91J 2022 002 > "$tmp"; download "$tmp"; rm -f "$tmp"
         echo "RSF: $(find "$OUT/JI91J/2022/002/ionogram" -name '*.RSF' 2>/dev/null | wc -l | tr -d ' '), SAO: $(find "$OUT/JI91J/2022/002/scaled" -name '*.SAO' 2>/dev/null | wc -l | tr -d ' '), EDP: $(find "$OUT/JI91J/2022/002/scaled" -name '*.EDP' 2>/dev/null | wc -l | tr -d ' ')"; du -sh "$OUT/JI91J/2022/002" ;;
  p1)    for item in "${P1[@]}"; do run_year "${item%%:*}" "${item##*:}"; done ;;
  p2)    for item in "${P2[@]}"; do run_year "${item%%:*}" "${item##*:}"; done ;;
  p3)    for item in "${P3[@]}"; do run_year "${item%%:*}" "${item##*:}"; done ;;
  year)  run_year "${2:?станция}" "${3:?год}" ;;
  *)     for item in "${P0[@]}"; do run_year "${item%%:*}" "${item##*:}"; done ;;
esac
echo "Корпус: $OUT"; du -sh "$OUT" 2>/dev/null
