"""
Manager for metrics table in Excel format.
Handles creation, updating and comparison of metrics across files.
"""

import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional
import logging


class MetricsTableManager:
    """
    Manages Excel table with metrics for processed files.
    Tracks improvements and updates data accordingly.
    """
    
    def __init__(self, data_dir: str = "data"):
        """
        Initialize metrics table manager.
        
        Args:
            data_dir: Directory for data files
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        
        # Generate filename with current date
        current_date = datetime.now().strftime("%d%m%y")
        self.table_file = self.data_dir / f"Table_metrics_{current_date}.xlsx"
        
        # Define columns for the metrics table
        self.columns = [
            "filename",
            "delta_WER",
            "Lev_Rating", 
            "delta_Lev",
            "Perplexity",
            "CorScore",
            "best_prompt_cor",
            "G_Eval",
            "METEOR",
            "LLM_Judge",
            "BertScore",
            "SumScore",
            "best_prompt_sum"
        ]
        
        # Prompt type mapping
        self.prompt_type_mapping = {
            "basic": 1,
            "saved": 2, 
            "few-shot": 3,
            "cot": 4,
            "aggregator": 5
        }
        
        self.logger = logging.getLogger(__name__)
    
    def _load_existing_table(self) -> Optional[pd.DataFrame]:
        """
        Load existing metrics table if it exists.
        
        Returns:
            DataFrame with existing data or None if file doesn't exist
        """
        if self.table_file.exists():
            try:
                df = pd.read_excel(self.table_file)
                self.logger.info(f"Loaded existing metrics table with {len(df)} rows")
                return df
            except Exception as e:
                self.logger.error(f"Error loading metrics table: {e}")
                return None
        return None
    
    def _extract_metrics_from_state(self, state: Dict[str, Any], filename: str) -> Dict[str, Any]:
        """
        Extract metrics from processing state.
        
        Args:
            state: Processing state dictionary from orchestrator
            filename: Name of processed file
            
        Returns:
            Dictionary with extracted metrics
        """
        metrics = {"filename": filename}
        
        # Correction metrics - extract from different possible locations
        correction_metrics = {}
        
        # Try different locations for correction metrics
        if "correction_metrics" in state:
            correction_metrics = state["correction_metrics"]
        elif "metrics" in state and "correction" in state["metrics"]:
            correction_metrics = state["metrics"]["correction"]
        elif "best_correction_variant" in state:
            best_variant = state["best_correction_variant"]
            if isinstance(best_variant, dict) and "metrics" in best_variant:
                correction_metrics = best_variant["metrics"]
        
        if correction_metrics:
            metrics["delta_WER"] = correction_metrics.get("delta_wer", 0)
            metrics["Lev_Rating"] = correction_metrics.get("lev_rating", 0) 
            metrics["delta_Lev"] = correction_metrics.get("delta_lev", 0)
            metrics["Perplexity"] = correction_metrics.get("perplexity", 0)
            metrics["CorScore"] = correction_metrics.get("cor_score", 0)
            
            # Best prompt type for correction
            best_prompt_cor = state.get("best_prompt_type_cor", "basic")
            if "best_correction_variant" in state:
                best_variant = state["best_correction_variant"]
                if isinstance(best_variant, dict) and "prompt_type" in best_variant:
                    best_prompt_cor = best_variant["prompt_type"]
            metrics["best_prompt_cor"] = self.prompt_type_mapping.get(best_prompt_cor, 1)
        
        # Summarization metrics - extract from different possible locations
        summarization_metrics = {}
        
        # Try different locations for summarization metrics
        if "summarization_metrics" in state:
            summarization_metrics = state["summarization_metrics"]
        elif "metrics" in state and "summarization" in state["metrics"]:
            summarization_metrics = state["metrics"]["summarization"]
        elif "best_summary_variant" in state:
            best_variant = state["best_summary_variant"]
            if isinstance(best_variant, dict) and "metrics" in best_variant:
                summarization_metrics = best_variant["metrics"]
        
        if summarization_metrics:
            metrics["G_Eval"] = summarization_metrics.get("geval", summarization_metrics.get("g_eval", 0))
            metrics["METEOR"] = summarization_metrics.get("meteor", 0)
            metrics["LLM_Judge"] = summarization_metrics.get("llm_judge", summarization_metrics.get("llm_judge", 0))
            metrics["BertScore"] = summarization_metrics.get("bert_score", summarization_metrics.get("bertscore", 0))
            metrics["SumScore"] = summarization_metrics.get("sum_score", summarization_metrics.get("sumscore", 0))
            
            # Best prompt type for summarization
            best_prompt_sum = state.get("best_prompt_type_sum", "basic")
            if "best_summary_variant" in state:
                best_variant = state["best_summary_variant"]
                if isinstance(best_variant, dict) and "prompt_type" in best_variant:
                    best_prompt_sum = best_variant["prompt_type"]
            metrics["best_prompt_sum"] = self.prompt_type_mapping.get(best_prompt_sum, 1)
        
        return metrics
    
    def _compare_metrics(self, new_metrics: Dict[str, Any], existing_row: pd.Series) -> bool:
        """
        Compare new metrics with existing ones to determine if update is needed.
        
        Args:
            new_metrics: New metrics dictionary
            existing_row: Existing row from DataFrame
            
        Returns:
            True if new metrics are better, False otherwise
        """
        # Define which metrics should be higher (better if increased)
        higher_is_better = [
            "Lev_Rating", "CorScore", "G_Eval", "METEOR", 
            "LLM_Judge", "BertScore", "SumScore"
        ]
        
        # Define which metrics should be lower (better if decreased)
        lower_is_better = ["delta_WER", "delta_Lev", "Perplexity"]
        
        improvements = 0
        total_comparable = 0
        
        for metric in higher_is_better:
            if metric in new_metrics and metric in existing_row:
                new_val = new_metrics[metric]
                old_val = existing_row[metric]
                if pd.notna(new_val) and pd.notna(old_val):
                    total_comparable += 1
                    if new_val > old_val:
                        improvements += 1
        
        for metric in lower_is_better:
            if metric in new_metrics and metric in existing_row:
                new_val = new_metrics[metric]
                old_val = existing_row[metric]
                if pd.notna(new_val) and pd.notna(old_val):
                    total_comparable += 1
                    if new_val < old_val:
                        improvements += 1
        
        # Update if majority of metrics are better
        if total_comparable > 0:
            return improvements > (total_comparable / 2)
        
        return False  # Default to no update if can't compare
    
    def update_metrics_table(self, state: Dict[str, Any], filename: str) -> None:
        """
        Update metrics table with new file processing results.
        
        Args:
            state: Processing state dictionary with metrics
            filename: Name of processed file
        """
        try:
            # Extract metrics from state
            new_metrics = self._extract_metrics_from_state(state, filename)
            
            # Load existing table
            existing_df = self._load_existing_table()
            
            if existing_df is None:
                # Create new table
                new_row = pd.DataFrame([new_metrics], columns=self.columns)
                new_row.to_excel(self.table_file, index=False)
                self.logger.info(f"Created new metrics table with {filename}")
            else:
                # Check if file already exists in table
                existing_row = existing_df[existing_df["filename"] == filename]
                
                if len(existing_row) > 0:
                    # File exists, compare metrics
                    existing_data = existing_row.iloc[0]
                    should_update = self._compare_metrics(new_metrics, existing_data)
                    
                    if should_update:
                        # Update existing row
                        for col in self.columns:
                            if col in new_metrics:
                                existing_df.loc[existing_df["filename"] == filename, col] = new_metrics[col]
                        existing_df.to_excel(self.table_file, index=False)
                        self.logger.info(f"Updated metrics for {filename} (improved)")
                    else:
                        self.logger.info(f"Metrics for {filename} not updated (not improved)")
                else:
                    # Add new row
                    new_row = pd.DataFrame([new_metrics], columns=self.columns)
                    updated_df = pd.concat([existing_df, new_row], ignore_index=True)
                    updated_df.to_excel(self.table_file, index=False)
                    self.logger.info(f"Added new metrics for {filename}")
                    
        except Exception as e:
            self.logger.error(f"Error updating metrics table: {e}")
    
    def get_table_info(self) -> Dict[str, Any]:
        """
        Get information about the metrics table.
        
        Returns:
            Dictionary with table information
        """
        info = {
            "table_file": str(self.table_file),
            "exists": self.table_file.exists(),
            "columns": self.columns
        }
        
        if self.table_file.exists():
            try:
                df = pd.read_excel(self.table_file)
                info["rows_count"] = len(df)
                info["files"] = df["filename"].tolist() if "filename" in df.columns else []
            except Exception as e:
                info["error"] = str(e)
        
        return info
