#!/bin/zsh
# 더블클릭하면 하이라이트 메이커를 켜고 브라우저를 엽니다.
cd "$(dirname "$0")"
PY=work/venv/bin/python
[ -x "$PY" ] || PY=.venv/bin/python
(sleep 2 && open http://127.0.0.1:8765) &
exec "$PY" -m autohl.app
