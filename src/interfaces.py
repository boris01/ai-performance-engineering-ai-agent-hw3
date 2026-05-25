from abc import ABC, abstractmethod
from typing import Optional, List, Dict
from src.domain import DatasetRow

class IBitextRepository(ABC):
    """Abstract baseline interface for data extraction rules."""

    @abstractmethod
    def count_records(self, category: Optional[str] = None, intent: Optional[str] = None) -> int:
        """Calculate row tallies matching target filters."""
        pass

    @abstractmethod
    def get_distribution(self, category: Optional[str] = None) -> Dict[str, int]:
        """Build value frequencies of intents."""
        pass

    @abstractmethod
    def get_row_samples(self, category: Optional[str] = None, intent: Optional[str] = None, limit: int = 3) -> List[DatasetRow]:
        """Fetch historical customer query examples."""
        pass