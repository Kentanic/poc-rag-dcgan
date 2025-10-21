from pydantic import BaseModel
import yaml, os

class Settings(BaseModel):
    cfg_path: str = os.getenv("POC_CFG", "configs/default.yaml")
    cfg: dict = {}

    def load(self):
        with open(self.cfg_path, "r", encoding="utf-8") as f:
            self.cfg = yaml.safe_load(f)

settings = Settings()
settings.load()