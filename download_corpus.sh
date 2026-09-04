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
#   ./download_corpus.sh year ST YYYY — произвольный станция-год (напр. year CO764 2019)
# ФАКТ (проба 2026-09-05, research-дисциплина): NOAA НЕ зеркалирует сырьё ionogram/ за
# 2023-2024 ни у одной проверенной станции и НИКОГДА — для TR170 (Тромсё), SO166 (Соданкюля),
# EB040, EA036, AT138, MO155-2022 (там только scaled/). Высокоширотной станции с сырьём в
# открытом архиве не найдено; фазы цикла покрываем иначе: 2012-2014 = максимум 24-го цикла,
# 2018-2020 = минимум, 2022 = рост 25-го. Полный набор E0 = p1 + p2; повторный запуск
# дотягивает недостающее. После каждой докачки: python -m pyon.manifest
set -u
BASE="https://data.ngdc.noaa.gov/instruments/remote-sensing/active/profilers-sounders/ionosonde/data"
HERE="$(cd "$(dirname "$0")" && pwd)"
OUT="$HERE/data/corpus"
TMPDIR_LOCAL="$HERE/local/tmp"; mkdir -p "$TMPDIR_LOCAL"
PAR=8   # параллельных загрузок

P0=( MO155:2019 JI91J:2022 JR055:2022 PQ052:2022 )
P2=( MO155:2013 MO155:2014 JI91J:2013 JI91J:2014
     JR055:2013 JR055:2014 PQ052:2013 PQ052:2014
     RO041:2019 RO041:2020 RO041:2021 )
P1=( MO155:2012 MO155:2018 MO155:2019
     JI91J:2019 JI91J:2020 JI91J:2022
     JR055:2012 JR055:2019 JR055:2020 JR055:2022
     PQ052:2019 PQ052:2022
     RO041:2022 )

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
  year)  run_year "${2:?станция}" "${3:?год}" ;;
  *)     for item in "${P0[@]}"; do run_year "${item%%:*}" "${item##*:}"; done ;;
esac
echo "Корпус: $OUT"; du -sh "$OUT" 2>/dev/null
