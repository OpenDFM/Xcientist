from pathlib import Path
import json
from typing import Any, Dict, Iterable, List, Union


class JsonFileStore:
    def __init__(self, path: Union[str, Path]):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self._write_all([])

    def load_all(self) -> List[Dict[str, Any]]:
        with self.path.open("r", encoding="utf-8") as fh:
            try:
                return json.load(fh)
            except json.JSONDecodeError:
                return []

    def _write_all(self, items: Iterable[Dict[str, Any]]) -> None:
        as_list = list(items)
        with self.path.open("w", encoding="utf-8") as fh:
            json.dump(as_list, fh, indent=2)

    def append(self, item: Dict[str, Any]) -> None:
        items = self.load_all()
        items.append(item)
        self._write_all(items)

    def update(self, item_id: str, new_item: Dict[str, Any]) -> None:
        items = self.load_all()
        for idx, existing in enumerate(items):
            if existing["id"] == item_id:
                items[idx] = new_item
                break
        else:
            items.append(new_item)
        self._write_all(items)
