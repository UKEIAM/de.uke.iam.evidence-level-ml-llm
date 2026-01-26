import enum
from pydantic import BaseModel


class EvidenceLevelCivic(enum.Enum):
    A = "A"
    B = "B"
    C = "C"
    D = "D"
    E = "E"
    UNSURE = "unsure"


class Confidence(enum.Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class EvLevelClassificationCivic(BaseModel):
    evidence_level: EvidenceLevelCivic
    explanation: str
    confidence: Confidence
