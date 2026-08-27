import yaml
from pathlib import Path
from pydantic import BaseModel, Field

class DataConfig(BaseModel):
    raw_data_path: Path
    processed_data_path: Path
    report_path: Path
    target_column: str
    temporal_column: str
    train_ratio: float = Field(..., gt=0, lt=1)
    validation_ratio: float = Field(..., gt=0, lt=1)
    test_ratio: float = Field(..., gt=0, lt=1)
    random_seed: int
    required_columns: list[str]

    @classmethod
    def load_from_yaml(cls, path: str | Path) -> "DataConfig":
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return cls(**data)

class BaselineConfig(BaseModel):
    random_seed: int
    target_column: str
    temporal_column: str
    model_artifact_dir: Path
    report_dir: Path
    data_config_path: Path
    feature_sets: dict[str, list[str]]
    false_positive_cost: float
    false_negative_cost: float
    min_recall_targets: list[float]

    @classmethod
    def load_from_yaml(cls, path: str | Path) -> "BaselineConfig":
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return cls(**data)
