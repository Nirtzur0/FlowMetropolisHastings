import json
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Optional

import numpy as np

@dataclass
class ExperimentLogger:
    root_dir: str
    name: Optional[str] = None

    def __post_init__(self) -> None:
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        run_name = timestamp if not self.name else f"{timestamp}_{self.name}"
        self.run_dir = os.path.join(self.root_dir, run_name)
        os.makedirs(self.run_dir, exist_ok=True)

    def save_config(self, config: Dict[str, Any]) -> str:
        path = os.path.join(self.run_dir, "config.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2, sort_keys=True)
        return path

    def save_stats(self, stats: Any) -> str:
        path = os.path.join(self.run_dir, "stats.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2, sort_keys=True)
        return path

    def save_diagnostics(self, diagnostics: Dict[str, Any]) -> str:
        path = os.path.join(self.run_dir, "diagnostics.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(diagnostics, f, indent=2, sort_keys=True)
        return path

    def save_chain(self, chain: np.ndarray, name: str) -> str:
        path = os.path.join(self.run_dir, f"{name}.npy")
        np.save(path, chain)
        return path
