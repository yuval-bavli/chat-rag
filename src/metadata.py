from typing import NamedTuple


class Metadata(NamedTuple):
    name: str
    timestamp: str
    type: str  # e.g., "question" or ""

    def to_dict(self) -> dict:
        return self._asdict()
    
    @classmethod
    def from_dict(cls, data) -> "Metadata":
        # Accept a dict, another Metadata, or an object with attributes
        if isinstance(data, cls):
            return data
        if isinstance(data, dict):
            name = data.get("name", "")
            timestamp = data.get("timestamp", "")
            typ = data.get("type", "")
        else:
            name = getattr(data, "name", "")
            timestamp = getattr(data, "timestamp", "")
            typ = getattr(data, "type", "")
        return cls(name=name, timestamp=timestamp, type=typ)
