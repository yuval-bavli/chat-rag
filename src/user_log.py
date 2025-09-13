from datetime import datetime
from typing import NamedTuple


class UserLog(NamedTuple):
    id: str
    timestamp: datetime
    name: str
    message: str

    def to_dict(self) -> dict:
        return self._asdict()

