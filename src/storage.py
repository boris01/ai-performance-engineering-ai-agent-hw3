import os

import pandas as pd
from typing import Optional, List, Dict
from src.interfaces import IBitextRepository
from src.domain import DatasetRow


class PandasBitextRepository(IBitextRepository):
    """Concrete data storage engine accessing the raw Bitext CSV file via Pandas."""

    def __init__(self, csv_filepath: str):
        csv_file_url = os.getenv("DATA_URL_PATH",
                                 "hf://datasets/bitext/Bitext-customer-support-llm-chatbot-training-dataset/Bitext_Sample_Customer_Support_Training_Dataset_27K_responses-v11.csv")

        if not os.path.exists(csv_filepath):
            df = pd.read_csv(csv_file_url)
            # Save to the given local path
            df.to_csv(csv_filepath, index=False)
        self.df = pd.read_csv(csv_filepath)

    def _apply_filters(self, category: Optional[str] = None, intent: Optional[str] = None) -> pd.DataFrame:
        working_df = self.df.copy()
        if category:
            working_df = working_df[working_df['category'].str.upper() == category.upper()]
        if intent:
            working_df = working_df[working_df['intent'].str.lower() == intent.lower()]
        return working_df

    def count_records(self, category: Optional[str] = None, intent: Optional[str] = None) -> int:
        return len(self._apply_filters(category, intent))

    def get_distribution(self, category: Optional[str] = None) -> Dict[str, int]:
        working_df = self._apply_filters(category=category)
        if working_df.empty:
            return {}
        return working_df['intent'].value_counts().to_dict()

    def get_row_samples(self, category: Optional[str] = None, intent: Optional[str] = None, limit: int = 3) -> List[
        DatasetRow]:
        working_df = self._apply_filters(category, intent).head(limit)

        results = []
        for idx, row in working_df.iterrows():
            results.append(DatasetRow(
                flags=str(row.get('flags', '')),
                instruction=str(row.get('instruction', '')),
                category=str(row.get('category', '')),
                intent=str(row.get('intent', '')),
                response=str(row.get('response', ''))
            ))
        return results

    def list_unique_dataset_categories(self) -> List[str]:
        if self.df.empty or "category" not in self.df.columns:
            return "Error: The dataset is empty or does not contain a 'category' column."
        return self.df['category'].dropna().unique().tolist()

    def sample_category_records(self, category: Optional[str] = None, intent: Optional[str] = None, limit: int = 5) -> str:
        working_df = self._apply_filters(category, intent).head(limit)

        if working_df.empty:
            available_categories = self.df["category"].unique().tolist() if "category" in self.df.columns else []
            return f"Query empty. Category '{category}' not found. Available sections: {available_categories}"

        sample_rows = working_df.head(limit)

        output_lines = []
        for idx, row in sample_rows.iterrows():
            customer_text = row.get("utterance", "N/A")
            agent_intent = row.get("intent", "N/A")
            output_lines.append(f"- [Log ID: {idx}] Intent: {agent_intent} | Text: \"{customer_text}\"")

        return "\n".join(output_lines)

    def get_category_metrics(self) -> str:
        """
        Scans the entire customer service log dataset and returns a complete list
        of all unique category names present along with their distribution totals.
        Use this tool when users ask 'what categories exist', 'list the categories',
        or ask for a structural summary of the dataset.
        """
        if self.df.empty or "category" not in self.df.columns:
            return "Error: Dataset column calculations unavailable."

        counts = self.df["category"].value_counts()
        output = ["--- DATASET VOLUME DISTRIBUTION METRICS ---"]
        for cat, total in counts.items():
            output.append(f"📦 Category: {cat:<18} | Record Count: {total:,} rows")

        return "\n".join(output)