#!/usr/bin/env python3
"""
Batch processor for handling files from INPUT directories with custom output format
"""

import asyncio
import sys
import time
from pathlib import Path
from typing import Dict, Any, Optional

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.orchestrator import Orchestrator
from src.utils.file_processor import FileProcessor
from src.metrics.enhanced_metrics_calculator import EnhancedMetricsCalculator
from src.utils.lm_studio_client import LMStudioClient
import config

class BatchProcessor:
    """
    Batch processor for handling multiple files with custom output format
    """
    
    def __init__(self):
        """Initialize batch processor"""
        self.file_processor = FileProcessor()
        self.orchestrator = None
        self.enhanced_metrics = EnhancedMetricsCalculator()
        
    async def initialize(self):
        """Initialize components"""
        print("Initializing batch processor...")
        
        # Initialize orchestrator
        self.orchestrator = Orchestrator()
        await self.orchestrator.initialize()
        
        # Override metrics calculator with enhanced version
        self.orchestrator.metrics_calculator = self.enhanced_metrics
        
        print("Batch processor initialized successfully")
    
    async def process_all_files(self, 
                             enable_correction: bool = True,
                             enable_summarization: bool = True):
        """
        Process all files from INPUT directories.
        
        Args:
            enable_correction: Enable text correction
            enable_summarization: Enable text summarization
        """
        # Get file statistics
        stats = self.file_processor.get_statistics()
        print(f"\nFile Statistics:")
        print(f"  Incorrect files: {stats['incorrect_files']}")
        print(f"  Etalon files: {stats['etalon_files']}")
        print(f"  Summary etalon files: {stats['summary_etalon_files']}")
        print(f"  Complete pairs: {stats['complete_pairs']}")
        
        # Get file pairs
        file_pairs = self.file_processor.get_file_pairs()
        
        if not file_pairs:
            print("No files found to process")
            return
        
        print(f"\nProcessing {len(file_pairs)} files...")
        
        # Process each file
        for i, (incorrect_file, etalon_file, summary_etalon_file) in enumerate(file_pairs, 1):
            print(f"\n[{i}/{len(file_pairs)}] Processing: {incorrect_file.name}")
            
            try:
                await self.process_single_file(
                    incorrect_file=incorrect_file,
                    etalon_file=etalon_file,
                    summary_etalon_file=summary_etalon_file,
                    enable_correction=enable_correction,
                    enable_summarization=enable_summarization
                )
            except Exception as e:
                print(f"Error processing file {incorrect_file.name}: {e}")
                continue
        
        print(f"\nBatch processing completed!")
        print(f"Results saved in:")
        print(f"  Corrections: {self.file_processor.correction_output_dir}")
        print(f"  Summaries: {self.file_processor.summary_output_dir}")
    
    async def process_single_file(self,
                                 incorrect_file: Path,
                                 etalon_file: Optional[Path],
                                 summary_etalon_file: Optional[Path],
                                 enable_correction: bool = True,
                                 enable_summarization: bool = True):
        """
        Process a single file.
        
        Args:
            incorrect_file: Path to incorrect file
            etalon_file: Path to etalon file
            summary_etalon_file: Path to summary etalon file
            enable_correction: Enable correction
            enable_summarization: Enable summarization
        """
        # Read files
        incorrect_text = self.file_processor.read_file_content(incorrect_file)
        etalon_text = self.file_processor.read_file_content(etalon_file) if etalon_file else ""
        summary_etalon = self.file_processor.read_file_content(summary_etalon_file) if summary_etalon_file else ""
        
        if not incorrect_text:
            print(f"  Skipping empty file: {incorrect_file.name}")
            return
        
        # Track processing time
        start_time = time.time()
        
        # Process correction
        corrected_text = ""
        correction_metrics = {}
        correction_time = 0
        
        if enable_correction:
            correction_start = time.time()
            
            # Use orchestrator for correction
            correction_result = await self.orchestrator.process_text(
                input_text=incorrect_text,
                reference_text=etalon_text if etalon_text else None,
                domain="general",
                enable_correction=True,
                enable_summarization=False
            )
            
            corrected_text = correction_result.get("corrected_text", "")
            correction_metrics = correction_result.get("metrics_correction", {})
            correction_time = time.time() - correction_start
            
            # Save correction result
            self.file_processor.save_correction_result(
                filename=incorrect_file.stem,
                incorrect_text=incorrect_text,
                correct_text=corrected_text,
                etalon_text=etalon_text,
                metrics=correction_metrics,
                processing_time=correction_time,
                prompt_number=correction_result.get("correction_attempts", 1)
            )
        
        # Process summarization
        summary_text = ""
        summary_metrics = {}
        summary_time = 0
        
        if enable_summarization and corrected_text:
            summary_start = time.time()
            
            # Use orchestrator for summarization
            summary_result = await self.orchestrator.process_text(
                input_text=corrected_text,
                reference_summary=summary_etalon if summary_etalon else None,
                domain="general",
                enable_correction=False,
                enable_summarization=True
            )
            
            summary_text = summary_result.get("summary", "")
            summary_metrics = summary_result.get("summary_metrics", {})
            summary_time = time.time() - summary_start
            
            # Save summary result
            self.file_processor.save_summary_result(
                filename=incorrect_file.stem,
                correct_text=corrected_text,
                summary_text=summary_text,
                etalon_summary=summary_etalon,
                metrics=summary_metrics,
                processing_time=summary_time,
                prompt_number=summary_result.get("summary_attempts", 1)
            )
        
        total_time = time.time() - start_time
        
        # Print summary
        print(f"  Correction: {'Done' if enable_correction else 'Skipped'}")
        if enable_correction and correction_metrics:
            print(f"    WER: {correction_metrics.get('wer_corrected', 0):.3f}")
            print(f"    LevRating: {correction_metrics.get('lev_corrected', 0):.3f}")
            print(f"    CorScore: {correction_metrics.get('cor_score', 0):.3f}")
        
        print(f"  Summarization: {'Done' if enable_summarization else 'Skipped'}")
        if enable_summarization and summary_metrics:
            print(f"    METEOR: {summary_metrics.get('meteor', 0):.3f}")
            print(f"    BLEU: {summary_metrics.get('bleu', 0):.3f}")
            print(f"    SumScore: {summary_metrics.get('sum_score', 0):.3f}")
        
        print(f"  Total time: {total_time:.2f}s")
    
    async def cleanup(self):
        """Cleanup resources"""
        if self.orchestrator:
            await self.orchestrator.cleanup()

async def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Batch processor for CorSumAgentsAI",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        "--input-dir",
        type=str,
        default="inputs",
        help="Input directory with files (default: inputs)"
    )
    
    parser.add_argument(
        "--no-correction",
        action="store_true",
        help="Skip correction phase"
    )
    
    parser.add_argument(
        "--no-summarization",
        action="store_true",
        help="Skip summarization phase"
    )
    
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Show only file statistics"
    )
    
    args = parser.parse_args()
    
    # Initialize processor
    processor = BatchProcessor()
    
    # Override input directory if specified
    if args.input_dir != "inputs":
        processor.file_processor = FileProcessor(args.input_dir)
    
    try:
        await processor.initialize()
        
        # Show statistics only
        if args.stats:
            stats = processor.file_processor.get_statistics()
            print("File Statistics:")
            print(f"  Incorrect files: {stats['incorrect_files']}")
            print(f"  Etalon files: {stats['etalon_files']}")
            print(f"  Summary etalon files: {stats['summary_etalon_files']}")
            print(f"  Complete pairs: {stats['complete_pairs']}")
            return
        
        # Process all files
        await processor.process_all_files(
            enable_correction=not args.no_correction,
            enable_summarization=not args.no_summarization
        )
        
    except Exception as e:
        print(f"Error: {e}")
    finally:
        await processor.cleanup()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nBatch processing interrupted by user")
    except Exception as e:
        print(f"Unexpected error: {e}")
        sys.exit(1)
