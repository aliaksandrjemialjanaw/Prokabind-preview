"""Скачивание RegPrecise 3.2 (regprecise.lbl.gov) для дорожки «сайты на своих местах и у своих белков» (решение
владельца 21, 10.11.2026): страницы таксономических коллекций и страницы регулонов. На странице регулона — локус-тег
регулятора этого генома, геном, и сайты с положением относительно гена и последовательностью.

Только чтение открытого источника: наружу ничего не отправляется (ни данных проекта, ни адреса владельца).
Один запрос за раз, пауза между запросами, повторы при обрыве связи (сеть владельца рвётся часто), докачка: готовые
файлы пропускаются. Запись происхождения каждого файла — download_log.tsv (адрес, время UTC, sha256, байты).

Запуск: python3 -I -B dev/regprecise/download.py [OUT_DIR [reverse]]      (по умолчанию data/incoming/regprecise_3.2)
Второй поток (`reverse`) идёт по тому же списку с конца: готовые файлы оба потока пропускают, так что встретившись они
заканчивают (10.11, 11:15: один поток давал ~42 страницы в минуту, остаток — около трёх часов)."""
import hashlib
import os
import re
import sys
import time
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, 'data/incoming/regprecise_3.2')
REVERSE = len(sys.argv) > 2 and sys.argv[2] == 'reverse'      # второй поток: тот же список регулонов с конца
BASE = 'https://regprecise.lbl.gov/'
AGENT = 'prokabind-research-client'
PAUSE = 0.7
os.makedirs(os.path.join(OUT, 'collections'), exist_ok=True)
os.makedirs(os.path.join(OUT, 'regulons'), exist_ok=True)
LOG = os.path.join(OUT, 'download_log.tsv')
if not os.path.exists(LOG):
    with open(LOG, 'w') as f:
        f.write('url\tpath\tretrieved_utc\tsha256\tbytes\n')


def fetch(url, path, must_contain, tries=6):
    """скачать в файл; True — готово (или уже было), False — не вышло"""
    if os.path.exists(path) and os.path.getsize(path) > 200:
        return True
    wait = 5.0
    for attempt in range(tries):
        try:
            req = urllib.request.Request(url, headers={'User-Agent': AGENT})
            with urllib.request.urlopen(req, timeout=60) as r:
                body = r.read()
            if must_contain.encode() not in body:
                raise ValueError('ответ без «%s» (%d байт)' % (must_contain, len(body)))
            tmp = path + '.tmp%d' % os.getpid()        # свой временный файл у каждого потока
            with open(tmp, 'wb') as f:
                f.write(body)
            os.replace(tmp, path)
            with open(LOG, 'a') as f:
                f.write('%s\t%s\t%s\t%s\t%d\n' % (url, os.path.relpath(path, OUT), time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                                                  hashlib.sha256(body).hexdigest(), len(body)))
            time.sleep(PAUSE)
            return True
        except Exception as err:                       # обрыв связи, отказ сервера, пустой ответ — подождать и повторить
            print('  повтор %d: %s — %s' % (attempt + 1, url, str(err)[:120]), flush=True)
            time.sleep(wait)
            wait = min(wait * 2, 120.0)
    return False


index = os.path.join(OUT, 'collections', 'index.html')
if not fetch(BASE + 'collections_tax.jsp', index, 'collection_tax.jsp'):
    sys.exit('список коллекций не скачан')
ids = sorted({int(x) for x in re.findall(r'collection_tax\.jsp\?collection_id=(\d+)', open(index, errors='replace').read())})
print('таксономических коллекций: %d' % len(ids), flush=True)
regulons = set()
for cid in ids:
    path = os.path.join(OUT, 'collections', 'tax_%d.html' % cid)
    if fetch(BASE + 'collection_tax.jsp?collection_id=%d' % cid, path, 'regulon'):
        regulons |= {int(x) for x in re.findall(r'regulon\.jsp\?regulon_id=(\d+)', open(path, errors='replace').read())}
regulons = sorted(regulons)
print('регулонов по страницам коллекций: %d' % len(regulons), flush=True)
t0 = time.time()
for sweep in range(4):
    missing = [r for r in regulons if not (os.path.exists(os.path.join(OUT, 'regulons', '%d.html' % r))
                                           and os.path.getsize(os.path.join(OUT, 'regulons', '%d.html' % r)) > 200)]
    if REVERSE:
        missing.reverse()
    print('проход %d: осталось %d' % (sweep + 1, len(missing)), flush=True)
    if not missing:
        break
    for n, r in enumerate(missing):
        fetch(BASE + 'regulon.jsp?regulon_id=%d' % r, os.path.join(OUT, 'regulons', '%d.html' % r), 'Regulog', tries=4)
        if n % 250 == 0:
            print('  %d из %d, %.0f мин' % (n, len(missing), (time.time() - t0) / 60), flush=True)
left = sum(1 for r in regulons if not os.path.exists(os.path.join(OUT, 'regulons', '%d.html' % r)))
print('готово: регулонов %d, не скачано %d, %.0f мин' % (len(regulons), left, (time.time() - t0) / 60), flush=True)
