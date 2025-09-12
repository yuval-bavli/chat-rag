from datetime import datetime
import json
from src.user_log import UserLog


class DataReader:

    def __init__(self, input_file: str) -> None:
        self._input_file = input_file


    def read_logs(self) -> list[UserLog]:

        comments = []
        json_obj = self._read_json(self._input_file)
        for i, comment in enumerate(json_obj):
            message = comment.get("message", "")
            if not message.strip():
                continue
            
            name = comment.get("name", "")
            if not name.strip():
                continue

            timestamp_str = comment.get("timestamp", "")
            if not timestamp_str or not timestamp_str.strip():
                continue
            
            timestamp =  datetime.fromisoformat(timestamp_str.strip())
            id = f"comment_{i}"
            comments.append(UserLog(id, timestamp, name, message ))

        return comments
    

    def _read_json(self, json_path: str) -> list[dict]:
        print(f"Reading file: {json_path}")
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        return data
