"""
Utility functions for processing files from INPUT directories
"""

import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging

class FileProcessor:
    """
    Utility class for processing files from INPUT directories
    """
    
    def __init__(self, base_dir: str = "inputs"):
        """
        Initialize file processor.
        
        Args:
            base_dir: Base directory for input files
        """
        self.base_dir = Path(base_dir)
        self.incorrect_dir = self.base_dir / "incorrect"
        self.etalon_dir = self.base_dir / "etalon"  
        self.summary_etalon_dir = self.base_dir / "summary_etalon"
        self.logger = logging.getLogger("file_processor")
        
        # Create output directories
        self.correction_output_dir = Path("data/correction")
        self.summary_output_dir = Path("data/summary")
        
        # Ensure directories exist
        self.correction_output_dir.mkdir(parents=True, exist_ok=True)
        self.summary_output_dir.mkdir(parents=True, exist_ok=True)
    
    def get_file_pairs(self) -> List[Tuple[Path, Optional[Path], Optional[Path]]]:
        """
        Get pairs of incorrect files with their corresponding etalon files.
        
        Returns:
            List of tuples (incorrect_file, etalon_file, summary_etalon_file)
        """
        file_pairs = []
        
        if not self.incorrect_dir.exists():
            self.logger.error(f"Directory {self.incorrect_dir} does not exist")
            return file_pairs
            
        # Get all incorrect files
        incorrect_files = list(self.incorrect_dir.glob("*.txt"))
        incorrect_files.extend(list(self.incorrect_dir.glob("*.md")))
        incorrect_files.extend(list(self.incorrect_dir.glob("*.text")))
        
        for incorrect_file in sorted(incorrect_files):
            # Find corresponding etalon file
            etalon_file = self._find_etalon_file(incorrect_file)
            
            # Find corresponding summary etalon file
            summary_etalon_file = self._find_summary_etalon_file(incorrect_file)
            
            file_pairs.append((incorrect_file, etalon_file, summary_etalon_file))
        
        self.logger.info(f"Found {len(file_pairs)} file pairs")
        return file_pairs
    
    def _find_etalon_file(self, incorrect_file: Path) -> Optional[Path]:
        """
        Find corresponding etalon file for incorrect file.
        
        Args:
            incorrect_file: Path to incorrect file
            
        Returns:
            Path to etalon file or None if not found
        """
        if not self.etalon_dir.exists():
            return None
            
        # Try different naming patterns
        base_name = incorrect_file.stem
        
        # Pattern 1: Same name in etalon directory
        etalon_file = self.etalon_dir / incorrect_file.name
        if etalon_file.exists():
            return etalon_file
            
        # Pattern 2: With _etalon suffix
        etalon_file = self.etalon_dir / f"{base_name}_etalon{incorrect_file.suffix}"
        if etalon_file.exists():
            return etalon_file
            
        # Pattern 3: With _correct suffix
        etalon_file = self.etalon_dir / f"{base_name}_correct{incorrect_file.suffix}"
        if etalon_file.exists():
            return etalon_file
            
        # Pattern 4: With _reference suffix
        etalon_file = self.etalon_dir / f"{base_name}_reference{incorrect_file.suffix}"
        if etalon_file.exists():
            return etalon_file
            
        return None
    
    def _find_summary_etalon_file(self, incorrect_file: Path) -> Optional[Path]:
        """
        Find corresponding summary etalon file for incorrect file.
        
        Args:
            incorrect_file: Path to incorrect file
            
        Returns:
            Path to summary etalon file or None if not found
        """
        if not self.summary_etalon_dir.exists():
            return None
            
        base_name = incorrect_file.stem
        
        # Try different naming patterns
        summary_file = self.summary_etalon_dir / f"{base_name}_summary{incorrect_file.suffix}"
        if summary_file.exists():
            return summary_file
            
        summary_file = self.summary_etalon_dir / f"{base_name}_etalon_summary{incorrect_file.suffix}"
        if summary_file.exists():
            return summary_file
            
        summary_file = self.summary_etalon_dir / f"{base_name}_reference_summary{incorrect_file.suffix}"
        if summary_file.exists():
            return summary_file
            
        return None
    
    def read_file_content(self, file_path: Path) -> str:
        """
        Read content from file.
        
        Args:
            file_path: Path to file
            
        Returns:
            File content as string
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read().strip()
        except Exception as e:
            self.logger.error(f"Error reading file {file_path}: {e}")
            return ""
    
    def save_correction_result(self, 
                             filename: str,
                             incorrect_text: str,
                             correct_text: str,
                             etalon_text: str,
                             metrics: Dict,
                             processing_time: float,
                             prompt_number: int = 1):
        """
        Save correction result in the specified format.
        
        Args:
            filename: Original filename
            incorrect_text: Original incorrect text
            correct_text: Corrected text
            etalon_text: Etalon text
            metrics: Dictionary of metrics
            processing_time: Processing time in seconds
            prompt_number: Prompt number used
        """
        output_file = self.correction_output_dir / filename
        
        content = f"""=== INCORRECT_TEXT ===
{incorrect_text}

=== CORRECT_TEXT ===
{correct_text}

=== ETALON_TEXT ===
{etalon_text}

=== METRICS ===
WER_0: {metrics.get('wer_original', 0):.3f}
WER: {metrics.get('wer_corrected', 0):.3f}
delta_WER: {metrics.get('delta_wer', 0):.3f}
LevRating_0: {metrics.get('lev_original', 0):.3f}
LevRating: {metrics.get('lev_corrected', 0):.3f}
delta_LEV: {metrics.get('delta_lev', 0):.3f}
PROMPT: {prompt_number}
Time: {processing_time:.3f} s, time per prompt: {processing_time/prompt_number:.3f} s.
"""
        
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(content)
            self.logger.info(f"Correction result saved to {output_file}")
        except Exception as e:
            self.logger.error(f"Error saving correction result: {e}")
    
    def save_summary_result(self,
                          filename: str,
                          correct_text: str,
                          summary_text: str,
                          etalon_summary: str,
                          metrics: Dict,
                          processing_time: float,
                          prompt_number: int = 1):
        """
        Save summary result in the specified format.
        
        Args:
            filename: Original filename
            correct_text: Corrected text
            summary_text: Generated summary
            etalon_summary: Etalon summary
            metrics: Dictionary of metrics
            processing_time: Processing time in seconds
            prompt_number: Prompt number used
        """
        output_file = self.summary_output_dir / filename
        
        content = f"""=== CORRECT_TEXT ===
{correct_text}

=== SUMMARY_TEXT ===
{summary_text}

=== ETALON_SUMMARY ===
{etalon_summary}

=== METRICS ===
METEOR: {metrics.get('meteor', 0):.3f}
BLEU: {metrics.get('bleu', 0):.3f}
ROUGE-1: {metrics.get('rouge1', 0):.3f}
ROUGE-2: {metrics.get('rouge2', 0):.3f}
ROUGE-L: {metrics.get('rougel', 0):.3f}
PROMPT: {prompt_number}
Time: {processing_time:.3f} s, time per prompt: {processing_time/prompt_number:.3f} s.
"""
        
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(content)
            self.logger.info(f"Summary result saved to {output_file}")
        except Exception as e:
            self.logger.error(f"Error saving summary result: {e}")
    
    def get_statistics(self) -> Dict[str, int]:
        """
        Get statistics about available files.
        
        Returns:
            Dictionary with file counts
        """
        stats = {
            'incorrect_files': 0,
            'etalon_files': 0,
            'summary_etalon_files': 0,
            'complete_pairs': 0
        }
        
        if self.incorrect_dir.exists():
            stats['incorrect_files'] = len(list(self.incorrect_dir.glob("*.txt")))
            stats['incorrect_files'] += len(list(self.incorrect_dir.glob("*.md")))
            stats['incorrect_files'] += len(list(self.incorrect_dir.glob("*.text")))
        
        if self.etalon_dir.exists():
            stats['etalon_files'] = len(list(self.etalon_dir.glob("*.txt")))
            stats['etalon_files'] += len(list(self.etalon_dir.glob("*.md")))
            stats['etalon_files'] += len(list(self.etalon_dir.glob("*.text")))
        
        if self.summary_etalon_dir.exists():
            stats['summary_etalon_files'] = len(list(self.summary_etalon_dir.glob("*.txt")))
            stats['summary_etalon_files'] += len(list(self.summary_etalon_dir.glob("*.md")))
            stats['summary_etalon_files'] += len(list(self.summary_etalon_dir.glob("*.text")))
        
        file_pairs = self.get_file_pairs()
        stats['complete_pairs'] = len(file_pairs)
        
        return stats
