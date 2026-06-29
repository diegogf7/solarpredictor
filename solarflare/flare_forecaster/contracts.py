from dataclasses import dataclass

@dataclass
class Forecast:
    probability: float
    probability_lower: float
    probability_upper: float
    pil_location: tuple
    magnitude_estimate: str
    horizon_hours: int
    timestamp: str
    region_id: str

    