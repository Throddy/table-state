"""Проверка документации: ссылки, пути, числа против карточек сабмитов, запрещённые обороты.

Запуск: python tools/check_docs.py

1. Ссылки вида [текст](путь) во всех .md ведут на существующие файлы.
2. Пути в обратных кавычках вида `каталог/файл` в документах (кроме журналов, где команды историчны) существуют.
3. Public и oof_all в таблице experiments/README.md совпадают с карточками submissions/*.json; каждое
   пятизначное число Public рядом с именем файла в README.md и report/REPORT.md совпадает с его карточкой.
4. Стиль: тире внутри предложения, «не X, а Y», слова-штампы, жирный шрифт посреди фразы, эмодзи, вопросы.
Код возврата 1, если найдены нарушения пунктов 1–3; находки пункта 4 печатаются для ручной проверки.
"""
from __future__ import annotations

import glob
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LOGS = {'experiments/LOG_core.md', 'experiments/LOG_det.md', 'experiments/LOG_ov.md'}
# в журналах пути историчны: файлы могли быть перенесены или удалены позже
HISTORICAL = LOGS
INTERNAL = ['контракт', 'CONTRACT', 'DoD', 'TASK ', 'PROPOSALS', 'HANDOFF', 'C6_SLIDES', 'ORG_ANSWERS', 'роль A', 'роли A', 'решение A', 'решения A']
TOP = ('ts/', 'tools/', 'tracks/', 'configs/', 'notebooks/', 'submissions/', 'report/', 'experiments/', 'tests/')
GENERATED = ('artifacts/', 'data/', 'out/')
EXPECTED = {'0.69993'}   # расчётное ожидание COMBO по сумме трёх разниц, не Public
BANNED = ['ключев', 'важно отметить', 'стоит подчеркнуть', 'в целом', 'таким образом', 'позволя', 'обеспечива',
          'эффективн', 'надёжн', 'надежн']


def markdown_files() -> list[str]:
    files = subprocess.run(['git', 'ls-files', '*.md'], capture_output=True, text=True, cwd=ROOT).stdout.split()
    return [f for f in files if (ROOT / f).exists()] + [str(p.relative_to(ROOT)) for p in ROOT.glob('**/*.md')
                                                        if str(p.relative_to(ROOT)) not in files
                                                        and not str(p.relative_to(ROOT)).startswith(('.venv', 'artifacts', 'data'))]


def prose(text: str) -> list[tuple[int, str]]:
    """Строки вне блоков кода, без содержимого обратных кавычек и ссылок."""
    out, in_code = [], False
    for i, line in enumerate(text.splitlines(), 1):
        if line.strip().startswith('```'):
            in_code = not in_code
            continue
        if not in_code:
            out.append((i, re.sub(r'https?://\S+', '', re.sub(r'`[^`]*`', '', line))))
    return out


def main() -> int:
    problems, style = [], []
    cards = {Path(p).stem: json.load(open(p, encoding='utf-8')) for p in glob.glob(str(ROOT / 'submissions' / '*.json'))}
    public = {n: c['public_lb'] for n, c in cards.items() if c.get('public_lb') is not None}
    for f in sorted(set(markdown_files())):
        text = (ROOT / f).read_text(encoding='utf-8')
        base = (ROOT / f).parent
        for target in re.findall(r'\]\(([^)#\s]+)(?:#[^)]*)?\)', text):
            if target.startswith(('http://', 'https://', 'mailto:')):
                continue
            if not (base / target).exists():
                problems.append(f'{f}: битая ссылка {target}')
        for word in INTERNAL:
            if word in text:
                problems.append(f'{f}: внутренняя пометка «{word}»')
        if f not in HISTORICAL:
            for path in re.findall(r'`([A-Za-z0-9_./*-]+)`', text):
                if path.startswith(TOP) and '*' not in path and '<' not in path and not path.startswith(GENERATED):
                    if not (ROOT / path.rstrip('/')).exists():
                        problems.append(f'{f}: нет пути {path}')
        for i, line in prose(text):
            checks = [(r' [—–] ', 'тире внутри предложения'), (r'(?i)(?<![А-Яа-яё])не [^.;:!?()|]{1,60}, а ', 'оборот «не X, а Y»'),
                      (r'^(?!\s*(?:[-*]|\d+\.)\s+\*\*).*\S.*\*\*[^*]+\*\*', 'жирный посреди фразы'),
                      (r'[\U0001F300-\U0001FAFF☀-➿]', 'эмодзи'), (r'[А-Яа-яё]\?', 'вопрос')]
            for pattern, label in checks:
                if re.search(pattern, line):
                    style.append(f'{f}:{i}: {label}: {line.strip()[:120]}')
            for word in BANNED:
                if word in line.lower():
                    style.append(f'{f}:{i}: слово «{word}»: {line.strip()[:120]}')
    # 3. числа против карточек
    rows = [l for l in (ROOT / 'experiments' / 'README.md').read_text(encoding='utf-8').splitlines() if re.match(r'\| \d+ \|', l)]
    for row in rows:
        cells = [c.strip() for c in row.strip('|').split('|')]
        name = cells[1].strip('`')
        card = cards.get(name)
        if card is None:
            problems.append(f'experiments/README.md: нет карточки {name}')
            continue
        if abs(float(cells[4]) - card['public_lb']) > 1e-9:
            problems.append(f'experiments/README.md: {name} Public {cells[4]} против {card["public_lb"]}')
        oof = round(card['local_metrics']['oof_all']['score'], 4)
        if abs(float(cells[6]) - oof) > 1e-9:
            problems.append(f'experiments/README.md: {name} oof_all {cells[6]} против {oof}')
    for f in ['README.md', 'report/REPORT.md', 'experiments/README.md']:
        for line in (ROOT / f).read_text(encoding='utf-8').splitlines():
            names = [n for n in re.findall(r'`([A-Za-z0-9_]+)`', line) if n in public]
            for value in re.findall(r'(?<![\d.])0\.\d{5}(?!\d)', line):
                if names and value not in EXPECTED and not any(abs(public[n] - float(value)) < 1e-9 for n in names) and float(value) not in public.values():
                    problems.append(f'{f}: число {value} не совпадает с Public файлов {names}')
                if not names and float(value) not in public.values() and value not in EXPECTED:
                    problems.append(f'{f}: число {value} не найдено среди Public карточек')
    for p in problems:
        print('ОШИБКА', p)
    for s in style:
        print('СТИЛЬ ', s)
    print(f'ошибок {len(problems)}, замечаний по стилю {len(style)}')
    return 1 if problems else 0


if __name__ == '__main__':
    sys.exit(main())
