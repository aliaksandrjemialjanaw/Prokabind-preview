"""Признаки ESM-2 для белков вне кэшей проекта — тем же путём, что кэш проекта (CONTRACTS § 2.8; T08, T511 stage2_esm.py):
`esm.pretrained.esm2_t33_650M_UR50D()` в режиме eval, repr_layers=[33], строки служебных токенов отброшены, расчёт FP32,
запись float16 «длина × 1280», один белок на проход. Ключ файла — sha256 последовательности (§ 1).

Запуск: ./.venv/bin/python -I -B dev/library/esm_embed.py QUEUE.tsv OUT_DIR [THREADS] [DEVICE]
QUEUE.tsv — строки «ключ<TAB>последовательность» в порядке счёта; готовый файл верной формы не пересчитывается.
Разработка по решению владельца 20 (BWH), `work/PLAN.md`."""
import os, sys, time, hashlib
import numpy as np
import torch

queue, out_dir = sys.argv[1], sys.argv[2]
threads = int(sys.argv[3]) if len(sys.argv) > 3 else 8
device = torch.device(sys.argv[4] if len(sys.argv) > 4 else 'cpu')
torch.set_num_threads(threads)
os.makedirs(out_dir, exist_ok=True)
rows = [l.rstrip('\n').split('\t') for l in open(queue) if l.strip()]
import esm
model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
model.eval()
model.to(device)
convert = alphabet.get_batch_converter()
t0 = time.time(); done = 0; residues = 0
for key, seq in rows:
    if hashlib.sha256(seq.encode()).hexdigest() != key:
        raise SystemExit('ключ не равен sha256 последовательности: %s' % key[:12])
    if len(seq) > 1022:
        continue
    path = os.path.join(out_dir, key + '.npy')
    if os.path.exists(path):
        try:
            a = np.load(path, mmap_mode='r')
            if a.dtype == np.float16 and a.shape == (len(seq), 1280):
                continue
        except Exception:
            pass
    _, _, tokens = convert([(key, seq)])
    with torch.no_grad():
        rep = model(tokens.to(device), repr_layers=[33])['representations'][33]
    a = rep[0, 1:len(seq) + 1].float().cpu().numpy().astype(np.float16)
    if not np.isfinite(a).all():
        raise SystemExit('значение вне float16 у %s' % key[:12])
    tmp = path + '.tmp'
    with open(tmp, 'wb') as h:
        np.save(h, a)
    os.replace(tmp, path)
    done += 1; residues += len(seq)
    if done % 20 == 0:
        print('посчитано %d белков, %d остатков, %.0f с (%.2f с на белок)' % (done, residues, time.time() - t0, (time.time() - t0) / done), flush=True)
print('готово: посчитано %d, %d остатков, %.0f с' % (done, residues, time.time() - t0), flush=True)
