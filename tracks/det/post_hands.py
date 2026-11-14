"""Вариант проверочного файла с постобработкой рук: hands_need_people и hands_cap (флаги `post` конфига).

Запуск: python tracks/det/post_hands.py --pred AB_ft_hands --name AB_ft_hands_pp
Берёт ART/preds/<pred>.csv (4964 строки), применяет `ts.post.apply` с двумя флагами (people берётся из
того же файла), пишет ART/preds/<name>.csv с карточкой. Регистрация: tools/submit_gate.py.
"""
from __future__ import annotations

import argparse
import json

import common

from ts import PRED_COLUMNS, io, paths, post
from ts.check import check_pred

FLAGS = {'hands_need_people': True, 'hands_cap': True, 'clip': True}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--pred', required=True)
    parser.add_argument('--name', required=True)
    args = parser.parse_args()
    source = paths.ART / 'preds' / f'{args.pred}.csv'
    pred = io.read_csv_exact(source)
    out = post.apply(pred, {'post': FLAGS})[PRED_COLUMNS]
    changed = int((out['hands'] != pred['hands']).sum())
    path = paths.art('preds', f'{args.name}.csv')
    out.to_csv(path, index=False)
    card = json.loads(source.with_suffix('.json').read_text(encoding='utf-8'))
    card.update({'name': args.name, 'owner': common.OWNER, 'created_utc': common.utc_now(), 'git_sha': common.git_sha(),
                 'command': common.command_line(), 'post': FLAGS, 'post_of': args.pred, 'post_rows_changed': changed,
                 'notes': f'{args.pred} с постобработкой hands_need_people и hands_cap (ts.post.apply)'})
    path.with_suffix('.json').write_text(json.dumps(card, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    check_pred(path)
    print(f'{args.name}: hands изменены в {changed} строках из {len(out)} -> {path}')


if __name__ == '__main__':
    main()
