#!/usr/bin/env python3
"""
Script for processing files from inputs/incorrect/ to correction/ and summary/
with best variant selection based on metrics
"""

import asyncio
import sys
import time
from pathlib import Path
from typing import Dict, Any, Optional, List

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.orchestrator import Orchestrator
from src.utils.file_processor import FileProcessor
from src.metrics.enhanced_metrics_calculator import EnhancedMetricsCalculator
from src.utils.lm_studio_client import LMStudioClient
import config

class FileProcessor:
    """
    Enhanced file processor for complete pipeline
    """
    
    def __init__(self):
        """Initialize processor"""
        self.inputs_dir = Path("inputs")
        self.incorrect_dir = self.inputs_dir / "incorrect"
        self.etalon_dir = self.inputs_dir / "etalon"
        self.summary_etalon_dir = self.inputs_dir / "summary_etalon"
        
        self.correction_output_dir = Path("data/correction")
        self.summary_output_dir = Path("data/summary")
        
        # Ensure output directories exist
        self.correction_output_dir.mkdir(parents=True, exist_ok=True)
        self.summary_output_dir.mkdir(parents=True, exist_ok=True)
        
        self.orchestrator = None
        self.enhanced_metrics = EnhancedMetricsCalculator()
    
    async def initialize(self):
        """Initialize components"""
        print("Initializing file processor...")
        
        # Initialize orchestrator
        self.orchestrator = Orchestrator()
        await self.orchestrator.initialize()
        
        # Override metrics calculator with enhanced version
        self.orchestrator.metrics_calculator = self.enhanced_metrics
        
        print("File processor initialized successfully")
    
    def get_input_files(self) -> List[Path]:
        """Get all files from inputs/incorrect/"""
        if not self.incorrect_dir.exists():
            print(f"Directory {self.incorrect_dir} does not exist")
            return []
        
        files = []
        for ext in ['*.txt', '*.md', '*.text']:
            files.extend(self.incorrect_dir.glob(ext))
        
        return sorted(files)
    
    def find_reference_files(self, filename: str) -> tuple[Optional[str], Optional[str]]:
        """Find reference files for given filename"""
        base_name = Path(filename).stem
        
        # Find etalon file
        etalon_file = None
        for ext in ['.txt', '.md', '.text']:
            candidate = self.etalon_dir / f"{base_name}{ext}"
            if candidate.exists():
                etalon_file = candidate
                break
            
            candidate = self.etalon_dir / f"{base_name}_etalon{ext}"
            if candidate.exists():
                etalon_file = candidate
                break
        
        # Find summary etalon file
        summary_etalon_file = None
        for ext in ['.txt', '.md', '.text']:
            candidate = self.summary_etalon_dir / f"{base_name}{ext}"
            if candidate.exists():
                summary_etalon_file = candidate
                break
            
            candidate = self.summary_etalon_dir / f"{base_name}_summary{ext}"
            if candidate.exists():
                summary_etalon_file = candidate
                break
        
        return (etalon_file, summary_etalon_file)
    
    def read_file(self, file_path: Path) -> str:
        """Read file content"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read().strip()
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            return ""
    
    async def process_correction(self, filename: str, input_text: str, etalon_text: str) -> tuple[str, Dict[str, Any]]:
        """Process text correction and return best result"""
        print(f"  Processing correction...")
        
        # Use orchestrator for correction
        result = await self.orchestrator.process_text(
            input_text=input_text,
            reference_text=etalon_text if etalon_text else None,
            domain="general",
            enable_correction=True,
            enable_summarization=False
        )
        
        corrected_text = result.get("corrected_text", input_text)
        metrics = result.get("metrics_correction", {})
        
        return corrected_text, metrics
    
    async def process_summarization(self, filename: str, corrected_text: str, summary_etalon: str) -> tuple[str, Dict[str, Any]]:
        """Process text summarization and return best result"""
        print(f"  Processing summarization...")
        
        # Use orchestrator for summarization
        result = await self.orchestrator.process_text(
            input_text=corrected_text,
            reference_summary=summary_etalon if summary_etalon else None,
            domain="general",
            enable_correction=False,
            enable_summarization=True
        )
        
        summary_text = result.get("summary", "")
        metrics = result.get("summary_metrics", {})
        
        return summary_text, metrics
    
    def save_correction_result(self, filename: str, input_text: str, corrected_text: str, 
                              etalon_text: str, metrics: Dict[str, Any], processing_time: float):
        """Save correction result with metrics"""
        output_file = self.correction_output_dir / filename
        
        # Calculate additional metrics if etalon is available
        if etalon_text:
            # Calculate original vs etalon metrics
            original_metrics = self.enhanced_metrics.calculate_correction_metrics(
                input_text, input_text, etalon_text
            )
            
            content = f"""=== INCORRECT_TEXT ===
{input_text}

=== CORRECT_TEXT ===
{corrected_text}

=== ETALON_TEXT ===
{etalon_text}

=== METRICS ===
WER_0: {original_metrics.get('wer_original', 0):.3f}
WER: {metrics.get('wer_corrected', 0):.3f}
delta_WER: {original_metrics.get('wer_original', 0) - metrics.get('wer_corrected', 0):.3f}
LevRating_0: {original_metrics.get('lev_original', 0):.3f}
LevRating: {metrics.get('lev_corrected', 0):.3f}
delta_LEV: {metrics.get('lev_corrected', 0) - original_metrics.get('lev_original', 0):.3f}
Perplexity: {metrics.get('perplexity', 0):.3f}
CorScore: {metrics.get('cor_score', 0):.3f}
PROMPT: {metrics.get('prompt_used', 1)}
Time: {processing_time:.3f} s, time per prompt: {processing_time/max(1, metrics.get('prompt_used', 1)):.3f} s.
"""
        else:
            content = f"""=== INCORRECT_TEXT ===
{input_text}

=== CORRECT_TEXT ===
{corrected_text}

=== ETALON_TEXT ===
{etalon_text}

=== METRICS ===
WER: {metrics.get('wer_corrected', 0):.3f}
LevRating: {metrics.get('lev_corrected', 0):.3f}
Perplexity: {metrics.get('perplexity', 0):.3f}
CorScore: {metrics.get('cor_score', 0):.3f}
PROMPT: {metrics.get('prompt_used', 1)}
Time: {processing_time:.3f} s, time per prompt: {processing_time/max(1, metrics.get('prompt_used', 1)):.3f} s.
"""
        
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"  Correction saved to {output_file}")
        except Exception as e:
            print(f"  Error saving correction: {e}")
    
    def save_summary_result(self, filename: str, corrected_text: str, summary_text: str,
                          summary_etalon: str, metrics: Dict[str, Any], processing_time: float):
        """Save summary result with metrics"""
        output_file = self.summary_output_dir / filename
        
        content = f"""=== CORRECT_TEXT ===
{corrected_text}

=== SUMMARY_TEXT ===
{summary_text}

=== ETALON_SUMMARY ===
{summary_etalon}

=== METRICS ===
METEOR: {metrics.get('meteor', 0):.3f}
BLEU: {metrics.get('bleu', 0):.3f}
ROUGE-1: {metrics.get('rouge1', 0):.3f}
ROUGE-2: {metrics.get('rouge2', 0):.3f}
ROUGE-L: {metrics.get('rougel', 0):.3f}
G-Eval: {metrics.get('geval', 0):.3f}
LLM-Judge: {metrics.get('llm_judge', 0):.1f}
BertScore: {metrics.get('bert_score', 0):.3f}
SumScore: {metrics.get('sum_score', 0):.3f}
PROMPT: {metrics.get('prompt_used', 1)}
Time: {processing_time:.3f} s, time per prompt: {processing_time/max(1, metrics.get('prompt_used', 1)):.3f} s.
"""
        
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"  Summary saved to {output_file}")
        except Exception as e:
            print(f"  Error saving summary: {e}")
    
    async def process_all_files(self):
        """Process all files from inputs/incorrect/"""
        input_files = self.get_input_files()
        
        if not input_files:
            print("No files found in inputs/incorrect/")
            return
        
        print(f"Found {len(input_files)} files to process")
        print(f"Output directories: {self.correction_output_dir}, {self.summary_output_dir}")
        
        for i, input_file in enumerate(input_files, 1):
            print(f"\n[{i}/{len(input_files)}] Processing: {input_file.name}")
            
            try:
                await self.process_single_file(input_file)
            except Exception as e:
                print(f"  Error processing {input_file.name}: {e}")
                continue
        
        print(f"\nProcessing completed!")
        print(f"Results in:")
        print(f"  Corrections: {self.correction_output_dir}")
        print(f"  Summaries: {self.summary_output_dir}")
    
    async def process_single_file(self, input_file: Path):
        """Process a single file through the complete pipeline"""
        start_time = time.time()
        
        # Read input file
        input_text = self.read_file(input_file)
        if not input_text:
            print(f"  Skipping empty file: {input_file.name}")
            return
        
        # Find reference files
        etalon_file, summary_etalon_file = self.find_reference_files(input_file.name)
        
        etalon_text = self.read_file(etalon_file) if etalon_file else ""
        summary_etalon = self.read_file(summary_etalon_file) if summary_etalon_file else ""
        
        print(f"  References: etalon={'found' if etalon_file else 'not found'}, "
              f"summary_etalon={'found' if summary_etalon_file else 'not found'}")
        
        # Step 1: Correction
        correction_start = time.time()
        corrected_text, correction_metrics = await self.process_correction(
            input_file.name, input_text, etalon_text
        )
        correction_time = time.time() - correction_start
        
        # Save correction result
        self.save_correction_result(
            input_file.name, input_text, corrected_text, 
            etalon_text, correction_metrics, correction_time
        )
        
        # Step 2: Summarization
        summary_start = time.time()
        summary_text, summary_metrics = await self.process_summarization(
            input_file.name, corrected_text, summary_etalon
        )
        summary_time = time.time() - summary_start
        
        # Save summary result
        self.save_summary_result(
            input_file.name, corrected_text, summary_text,
            summary_etalon, summary_metrics, summary_time
        )
        
        total_time = time.time() - start_time
        
        # Print summary
        print(f"  Results:")
        print(f"    Summary G-Eval: {summary_metrics.get('geval', 0):.3f}")
        print(f"    Summary LLM-Judge: {summary_metrics.get('llm_judge', 0):.1f}")
        print(f"    Summary BertScore: {summary_metrics.get('bert_score', 0):.3f}")
        print(f"    Summary SumScore: {summary_metrics.get('sum_score', 0):.3f}")
        print(f"    Total time: {total_time:.2f}s")
    
    async def cleanup(self):
        """Cleanup resources"""
        if self.orchestrator:
            await self.orchestrator.cleanup()

async def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Process files from inputs/incorrect/ to correction/ and summary/"
    )
    
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Show only file statistics"
    )
    
    args = parser.parse_args()
    
    processor = FileProcessor()
    
    try:
        await processor.initialize()
        
        # Show statistics only
        if args.stats:
            input_files = processor.get_input_files()
            print(f"Input files in inputs/incorrect/: {len(input_files)}")
            for f in input_files:
                print(f"  {f.name}")
            return
        
        # Process all files
        await processor.process_all_files()
        
    except Exception as e:
        print(f"Error: {e}")
    finally:
        await processor.cleanup()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nProcessing interrupted by user")
    except Exception as e:
        print(f"Unexpected error: {e}")
        sys.exit(1)
