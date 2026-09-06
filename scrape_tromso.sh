#!/usr/bin/env bash
# Скрейпер живых НЗ-ионограмм Тромсё (chirpsounder2, J. Vierinen): раз в 15 мин обе трассы;
# сохраняем только новую картинку (md5 отличается от предыдущей). Имена: data/tromso/<трасса>/<UTC>.png
cd "$(dirname "$0")" || exit 1
BASE="http://4.235.86.214/iono"
declare -A T=( [sgo-tgo]=latest-lfm-SGO-TGO.png [juliusruh-tgo]=latest-digisonde-Juliusruh-TGO.png )
while true; do
  for k in "${!T[@]}"; do
    d="data/tromso/$k"; mkdir -p "$d"; tmp="$d/.latest.png"
    if curl -sf -m 60 -o "$tmp" "$BASE/${T[$k]}"; then
      new=$(md5sum "$tmp" | cut -c1-32); old=$(cat "$d/.last_md5" 2>/dev/null)
      if [ "$new" != "$old" ]; then
        mv "$tmp" "$d/$(date -u '+%Y%m%dT%H%M%SZ').png"; echo "$new" > "$d/.last_md5"
        echo "$(date -u '+%F %T') $k новая ($new)"
      else rm -f "$tmp"; fi
    else echo "$(date -u '+%F %T') $k ошибка загрузки"; fi
  done
  sleep 900
done
