#!/usr/bin/env python3
"""
Main file for batch processing of files CorSumAgentsAI
"""

import asyncio
import sys
import json
import logging
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime
import re
import os
import threading
import webbrowser
import time

# Disable progress bars and verbose output from ML libraries
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["TQDM_DISABLE"] = "1"
os.environ["PROGRESS_BAR_DISABLE"] = "1"
os.environ["DISABLE_PROGRESS_BAR"] = "1"
os.environ["HUGGINGFACE_HUB_DISABLE_TELEMETRY"] = "1"

# Suppress tqdm output completely
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="tqdm")
warnings.filterwarnings("ignore", category=UserWarning)

# Redirect tqdm to null
import sys
from io import StringIO
class TqdmSilencer:
    def write(self, text):
        pass
    def flush(self):
        pass

# Replace tqdm's output
try:
    import tqdm
    tqdm.tqdm = lambda *args, **kwargs: args[0] if args else iter([])
except ImportError:
    pass

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.orchestrator import Orchestrator
from src.utils.metrics_table_manager import MetricsTableManager
from src.utils.realtime_metrics import realtime_store
import config
import threading
import webbrowser
import time

def setup_logging():
    """Setup logging with date-based folders"""
    from datetime import datetime
    
    # Create date string in DDMMYY format
    current_date = datetime.now()
    date_str = current_date.strftime("%d%m%y")  # DDMMYY format
    
    # Create logs directory structure
    logs_dir = Path("logs")
    date_dir = logs_dir / date_str
    
    # Create directories if they don't exist
    date_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logging configuration
    log_file = date_dir / f"corsum_agents_{date_str}.log"
    
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, config.LOG_LEVEL.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    # Suppress httpx INFO logs
    logging.getLogger('httpx').setLevel(logging.WARNING)
    logging.getLogger('transformers').setLevel(logging.WARNING)
    logging.getLogger('tokenizers').setLevel(logging.WARNING)
    logging.getLogger('huggingface_hub').setLevel(logging.WARNING)
    
    # Create logger for main module
    logger = logging.getLogger(__name__)
    logger.info(f"Logging initialized. Log file: {log_file}")
    
    return logger

def start_web_monitor():
    """Start web monitor in background thread"""
    def run_monitor():
        try:
            # Import Flask app
            from web_monitor import app
            
            print("Starting Web Monitor on http://127.0.0.1:5000")
            print("Web Monitor will be available for monitoring metrics in real-time")
            app.run(debug=False, host='127.0.0.1', port=5000, use_reloader=False)
        except Exception as e:
            print(f"Error starting web monitor: {e}")
            import traceback
            traceback.print_exc()
    
    # Start monitor in background thread
    print("Initializing Web Monitor...")
    monitor_thread = threading.Thread(target=run_monitor, daemon=True)
    monitor_thread.start()
    print("Web Monitor thread started")
    
    # Open browser after a short delay
    def open_browser():
        time.sleep(3)
        try:
            webbrowser.open('http://127.0.0.1:5000')
            print("Web Monitor opened in browser at http://127.0.0.1:5000")
        except Exception as e:
            print(f"Could not open browser automatically: {e}")
            print("Please manually open http://127.0.0.1:5000 in your browser")
    
    browser_thread = threading.Thread(target=open_browser, daemon=True)
    browser_thread.start()

def extract_and_store_metrics(state: Dict[str, Any], filename: str):
    """Extract metrics from state and store in realtime store"""
    try:
        # Extract correction metrics
        correction_metrics = {}
        if "best_correction_variant" in state:
            best_variant = state["best_correction_variant"]
            if isinstance(best_variant, dict) and "metrics" in best_variant:
                metrics = best_variant["metrics"]
                correction_metrics = {
                    'delta_WER': metrics.get('delta_wer', 0),
                    'Lev_Rating': metrics.get('lev_rating', 0),
                    'delta_Lev': metrics.get('delta_lev', 0),
                    'Perplexity': metrics.get('perplexity', 0),
                    'CorScore': metrics.get('cor_score', 0),
                    'best_prompt_cor': 1  # Default to basic
                }
                
                # Get prompt type
                prompt_type = best_variant.get('prompt_type', 'basic')
                prompt_mapping = {'basic': 1, 'saved': 2, 'few-shot': 3, 'cot': 4, 'aggregator': 5}
                correction_metrics['best_prompt_cor'] = prompt_mapping.get(prompt_type, 1)
        
        # Extract summarization metrics
        summarization_metrics = {}
        if "best_summary_variant" in state:
            best_variant = state["best_summary_variant"]
            if isinstance(best_variant, dict) and "metrics" in best_variant:
                metrics = best_variant["metrics"]
                summarization_metrics = {
                    'G_Eval': metrics.get('geval', metrics.get('g_eval', 0)),
                    'METEOR': metrics.get('meteor', 0),
                    'LLM_Judge': metrics.get('llm_judge', metrics.get('llm_judge', 0)),
                    'BertScore': metrics.get('bert_score', metrics.get('bertscore', 0)),
                    'SumScore': metrics.get('sum_score', metrics.get('sumscore', 0)),
                    'best_prompt_sum': 1  # Default to basic
                }
                
                # Get prompt type
                prompt_type = best_variant.get('prompt_type', 'basic')
                prompt_mapping = {'basic': 1, 'saved': 2, 'few-shot': 3, 'cot': 4, 'aggregator': 5}
                summarization_metrics['best_prompt_sum'] = prompt_mapping.get(prompt_type, 1)
        
        # Combine all metrics
        all_metrics = {
            'filename': filename,
            **correction_metrics,
            **summarization_metrics
        }
        
        # Store in realtime store
        realtime_store.add_metrics(filename, all_metrics)
        print(f"Stored realtime metrics for {filename}")
        
    except Exception as e:
        print(f"Error extracting/storing metrics: {e}")

def print_separator(title: str = ""):
    """Print separator"""
    print("\n" + "="*60)
    if title:
        print(f" {title}")
        print("="*60)

def save_results_to_file(results: dict, output_file: str):
    """Save results to file"""
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2, default=str)
        print(f"\n Results saved to file: {output_file}")
    except Exception as e:
        print(f"\n Error saving results: {e}")

def save_correction_results(results: dict, output_file: str):
    """Save correction results in text format with === separators"""
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            # Incorrect text
            f.write("=== INCORRECT_TEXT ===\n")
            f.write(format_text_sentences(results.get('input_text', '')) + "\n\n")
            
            # Corrected text
            f.write("=== CORRECT_TEXT ===\n")
            f.write(format_text_sentences(results.get('corrected_text', '')) + "\n\n")
            
            # Reference text
            f.write("=== ETALON_TEXT ===\n")
            f.write(format_text_sentences(results.get('reference_text', '')) + "\n\n")
            
            # Best prompt
            f.write("=== PROMPT ===\n")
            best_prompt = results.get('best_correction_prompt', 'N/A')
            f.write(f"{best_prompt}\n\n")
            
            # Metrics
            f.write("=== METRICS ===\n")
            m = results.get('metrics_correction', {})
            
            # WER metrics
            wer_orig = m.get('wer_original', 'N/A')
            wer_corr = m.get('wer_corrected', 'N/A')
            delta_wer = None
            if isinstance(wer_orig, (int, float)) and isinstance(wer_corr, (int, float)):
                delta_wer = wer_orig - wer_corr  # WER(before) - WER(after)
            f.write(f"WER_0: {wer_orig:.4f}\n" if isinstance(wer_orig, (int, float)) else f"WER_0: {wer_orig}\n")
            f.write(f"WER: {wer_corr:.4f}\n" if isinstance(wer_corr, (int, float)) else f"WER: {wer_corr}\n")
            if delta_wer is not None:
                f.write(f"delta_WER: {delta_wer:+.4f}\n")
            
            # LevRating metrics
            lev_orig = m.get('lev_original', 'N/A')
            lev_corr = m.get('lev_corrected', 'N/A')
            delta_lev = None
            if isinstance(lev_orig, (int, float)) and isinstance(lev_corr, (int, float)):
                delta_lev = lev_corr - lev_orig  # LevRating(after) - LevRating(before)
            f.write(f"LevRating_0: {lev_orig:.4f}\n" if isinstance(lev_orig, (int, float)) else f"LevRating_0: {lev_orig}\n")
            f.write(f"LevRating: {lev_corr:.4f}\n" if isinstance(lev_corr, (int, float)) else f"LevRating: {lev_corr}\n")
            if delta_lev is not None:
                f.write(f"delta_LEV: {delta_lev:+.4f}\n")
            
            # Perplexity and CorScore
            perplexity = m.get('perplexity', 'N/A')
            cor_score = m.get('cor_score', 'N/A')
            f.write(f"Perplexity: {perplexity:.4f}\n" if isinstance(perplexity, (int, float)) else f"Perplexity: {perplexity}\n")
            f.write(f"CorScore: {cor_score:.4f}\n" if isinstance(cor_score, (int, float)) else f"CorScore: {cor_score}\n")
            
            # Additional info
            correction_time = results.get('correction_time', 'N/A')
            if correction_time is None:
                correction_time = 'N/A'
            f.write(f"PROMPT: {results.get('correction_attempts', 1)}\n")
            f.write(f"TIME: {correction_time}\n")
            
        print(f"\n Correction results saved to file: {output_file}")
    except Exception as e:
        print(f"\n Error saving correction results: {e}")

def save_summary_results(results: dict, output_file: str):
    """Save summary results in text format with === separators"""
    try:
        # Extract reference_summary from results or state
        reference_summary = results.get('reference_summary') or results.get('state', {}).get('reference_summary', '')
        
        with open(output_file, 'w', encoding='utf-8') as f:
            # Corrected text
            f.write("=== CORRECT_TEXT ===\n")
            f.write(format_text_sentences(results.get('corrected_text', '')) + "\n\n")
            
            # Summary text
            f.write("=== SUMMARY_TEXT ===\n")
            f.write(format_text_sentences(results.get('summary', '')) + "\n\n")
            
            # Reference summary
            f.write("=== ETALON_SUMMARY ===\n")
            f.write(format_text_sentences(reference_summary) + "\n\n")
            
            # Best prompt
            f.write("=== PROMPT ===\n")
            best_prompt = results.get('best_summary_prompt', 'N/A')
            f.write(f"{best_prompt}\n\n")
            
            # Metrics
            f.write("=== METRICS ===\n")
            m = results.get('summary_metrics', {})
            
            # Summary metrics
            meteor = m.get('meteor', 'N/A')
            geval = m.get('geval', 'N/A')
            llm = m.get('llm_judge', 'N/A')
            bert = m.get('bert_score', 'N/A')
            sum_score = m.get('sum_score', 'N/A')
            
            f.write(f"METEOR: {meteor:.4f}\n" if isinstance(meteor, (int, float)) else f"METEOR: {meteor}\n")
            f.write(f"G-Eval: {geval:.4f}\n" if isinstance(geval, (int, float)) else f"G-Eval: {geval}\n")
            f.write(f"LLM-Judge: {llm:.4f}\n" if isinstance(llm, (int, float)) else f"LLM-Judge: {llm}\n")
            f.write(f"BertScore: {bert:.4f}\n" if isinstance(bert, (int, float)) else f"BertScore: {bert}\n")
            f.write(f"SumScore: {sum_score:.4f}\n" if isinstance(sum_score, (int, float)) else f"SumScore: {sum_score}\n")
            
            # Additional info
            summary_time = results.get('summary_time', 'N/A')
            if summary_time is None:
                summary_time = 'N/A'
            f.write(f"TIME: {summary_time}\n")
            
        print(f"\n Summary results saved to file: {output_file}")
    except Exception as e:
        print(f"\n Error saving summary results: {e}")

def print_variant_table(variants: list, variant_type: str):
    """Print variants in table format for comparison"""
    if not variants:
        return
    
    print(f"\n {variant_type.upper()} VARIANTS TABLE:")
    print("=" * 120)
    
    if variant_type == "correction":
        headers = ["Prompt", "Temperature", "WER", "deltaWER", "LevRating", "deltaLev", "Perplexity", "CorScore"]
        print(f"{'Prompt':<20} {'Temp':<8} {'WER':<8} {'deltaWER':<10} {'LevRating':<10} {'deltaLev':<10} {'Perplexity':<12} {'CorScore':<8}")
        print("-" * 120)
        
        for i, variant in enumerate(variants):
            prompt_type = variant.get("prompt_type", f"Variant{i+1}")
            temp = variant.get("temperature", 0.0)
            metrics = variant.get("metrics", {})
            
            wer = metrics.get("wer_corrected", 0.0)
            delta_wer = metrics.get("delta_wer", 0.0)
            lev = metrics.get("lev_corrected", 0.0)
            delta_lev = metrics.get("delta_lev", 0.0)
            perp = metrics.get("perplexity", 0.0)
            cor = metrics.get("cor_score", 0.0)
            
            print(f"{prompt_type:<20} {temp:<8.1f} {wer:<8.4f} {delta_wer:<+10.4f} {lev:<10.4f} {delta_lev:<+10.4f} {perp:<12.4f} {cor:<8.4f}")
    
    elif variant_type == "summarization":
        headers = ["Prompt", "Temperature", "G-Eval", "LLM-Judge", "METEOR", "BertScore", "SumScore"]
        print(f"{'Prompt':<20} {'Temp':<8} {'G-Eval':<8} {'LLM-Judge':<10} {'METEOR':<8} {'BertScore':<10} {'SumScore':<8}")
        print("-" * 120)
        
        for i, variant in enumerate(variants):
            prompt_type = variant.get("prompt_type", f"Variant{i+1}")
            temp = variant.get("temperature", 0.0)
            metrics = variant.get("metrics", {})
            
            geval = metrics.get("geval", 0.0)
            llm = metrics.get("llm_judge", 0.0)
            meteor = metrics.get("meteor", 0.0)
            bert = metrics.get("bert_score", 0.0)
            sum_score = metrics.get("sum_score", 0.0)
            
            print(f"{prompt_type:<20} {temp:<8.1f} {geval:<8.4f} {llm:<10.4f} {meteor:<8.4f} {bert:<10.4f} {sum_score:<8.4f}")
    
    print("=" * 120)

def print_batch_results(results: dict, file_type: str):
    """Print results in tree format"""
    print("\n RESULTS:")
    
    # Note: Variant tables are already printed by orchestrator after each ensemble
    # No need to duplicate them here
    
    # Correction
    if "corrected_text" in results:
        print(" CORRECTION")
        orig_text = results.get('input_text', 'N/A')
        if orig_text is None:
            orig_text = 'N/A'
        print(f"  # Original text: {orig_text[:50]}...")
        
        corr_text = results.get('corrected_text', 'N/A')
        if corr_text is None:
            corr_text = 'N/A'
        print(f"    Corrected text: {corr_text[:50]}...")
        
        # Show best prompt
        best_prompt = results.get('best_correction_prompt', 'N/A')
        if best_prompt != 'N/A':
            print(f"    Best prompt: {best_prompt[:100]}...")
        
        if "metrics_correction" in results:
            m = results["metrics_correction"]
            print("    Metrics:")
            wer_orig = m.get('wer_original', 'N/A')
            wer = m.get('wer_corrected', 'N/A')
            lev_orig = m.get('lev_original', 'N/A')
            lev = m.get('lev_corrected', 'N/A')
            perp = m.get('perplexity', 'N/A')
            cor = m.get('cor_score', 'N/A')
            
            # Calculate deltas (improvements)
            delta_wer = None
            delta_lev = None
            if isinstance(wer_orig, (int, float)) and isinstance(wer, (int, float)):
                delta_wer = wer_orig - wer  # WER(before) - WER(after), should be > 0 for improvement
            if isinstance(lev_orig, (int, float)) and isinstance(lev, (int, float)):
                delta_lev = lev - lev_orig  # LevRating(after) - LevRating(before), should be > 0 for improvement
            
            # Highlight key metrics with emphasis
            if isinstance(cor, (int, float)):
                cor_quality = "EXCELLENT" if cor > 0.8 else "GOOD" if cor > 0.5 else "POOR"
                print(f"       CorScore: {cor:.4f} [{cor_quality}]")
            else:
                print(f"       CorScore: {cor}")
                
            if delta_wer is not None:
                wer_improvement = "IMPROVED" if delta_wer > 0 else "WORSENED" if delta_wer < 0 else "SAME"
                print(f"       WER: {wer:.4f} (DeltaWER: {delta_wer:+.4f}) [{wer_improvement}]")
            else:
                print(f"       WER: {wer}")
                
            if delta_lev is not None:
                lev_improvement = "IMPROVED" if delta_lev > 0 else "WORSENED" if delta_lev < 0 else "SAME"
                print(f"       LevRating: {lev:.4f} (DeltaLev: {delta_lev:+.4f}) [{lev_improvement}]")
            else:
                print(f"       LevRating: {lev}")
                
            print(f"       Perplexity: {perp:.4f}" if isinstance(perp, (int, float)) else f"       Perplexity: {perp}")
            
            # Calculate overall correction quality based on LevRating and improvements
            lev_rating = lev if isinstance(lev, (int, float)) else 0
            has_improvements = (delta_wer or 0) > 0 or (delta_lev or 0) > 0
            
            if has_improvements and lev_rating > 0.95:
                quality = "ОТЛИЧНО"
            elif has_improvements and 0.85 <= lev_rating <= 0.95:
                quality = "ХОРОШО"
            elif has_improvements and 0.75 <= lev_rating <= 0.85:
                quality = "УДОВЛЕТВОРИТЕЛЬНО"
            elif has_improvements and lev_rating < 0.75:
                quality = "ТРЕБУЕТСЯ УЛУЧШЕНИЕ"
            elif not has_improvements and lev_rating > 0.75:
                quality = "ТРЕБУЕТСЯ УЛУЧШЕНИЕ"
            else:  # not has_improvements and lev_rating < 0.75
                quality = "ПЛОХО"
                
            print(f"        Quality: {quality}")
            
            # Show correction time
            correction_time = results.get('correction_time', 'N/A')
            if correction_time != 'N/A':
                print(f"        Processing time: {correction_time}")
            
            # Show aggregation results
            aggregator_used = results.get('aggregator_used', 'N/A')
            aggregation_reason = results.get('aggregation_reason', 'N/A')
            if aggregator_used != 'N/A':
                print(f"       Aggregation: {'Used' if aggregator_used else 'Not used'}")
                if aggregation_reason != 'N/A':
                    print(f"       Reason: {aggregation_reason}")
    
    # Summarization
    if "summary" in results:
        print(" SUMMARIZATION")
        summ_text = results.get('summary', 'N/A')
        if summ_text is None:
            summ_text = 'N/A'
        print(f"    Summary text: {summ_text[:50]}...")
        
        # Show best prompt
        best_prompt = results.get('best_summary_prompt', 'N/A')
        if best_prompt != 'N/A':
            print(f"    Best prompt: {best_prompt[:100]}...")
        
        if "summary_metrics" in results:
            m = results["summary_metrics"]
            print("    Metrics:")
            geval = m.get('geval', 'N/A')
            llm = m.get('llm_judge', 'N/A')
            meteor = m.get('meteor', 'N/A')
            bert = m.get('bert_score', 'N/A')
            sum_score = m.get('sum_score', 'N/A')
            quality = results.get('summary_quality', 'N/A')
            
            # Highlight SumScore with quality assessment
            if isinstance(sum_score, (int, float)):
                sum_quality = "EXCELLENT" if sum_score > 0.8 else "GOOD" if sum_score > 0.6 else "BAD"
                print(f"        SumScore: {sum_score:.4f} [{sum_quality}]")
            else:
                print(f"        SumScore: {sum_score}")
            
            # Show individual metrics with emphasis
            print(f"        G-Eval: {geval:.4f}" if isinstance(geval, (int, float)) else f"        G-Eval: {geval}")
            print(f"        LLM-Judge: {llm:.4f}" if isinstance(llm, (int, float)) else f"        LLM-Judge: {llm}")
            print(f"        METEOR: {meteor:.4f}" if isinstance(meteor, (int, float)) else f"        METEOR: {meteor}")
            print(f"        BertScore: {bert:.4f}" if isinstance(bert, (int, float)) else f"        BertScore: {bert}")
            print(f"        Quality: {quality}")
            
            # Show summarization time
            summary_time = results.get('summary_time', 'N/A')
            if summary_time != 'N/A':
                print(f"        Processing time: {summary_time}")
            
            # Show aggregation results
            aggregator_used = results.get('aggregator_used', 'N/A')
            aggregation_reason = results.get('aggregation_reason', 'N/A')
            if aggregator_used != 'N/A':
                print(f"        Aggregation: {'Used' if aggregator_used else 'Not used'}")
                if aggregation_reason != 'N/A':
                    print(f"        Reason: {aggregation_reason}")
    
    print()  # Empty line for separation

async def process_text_file(input_file: str, 
                         reference_file: Optional[str] = None,
                         reference_summary_file: Optional[str] = None,
                         domain: str = "general",
                         output_file: Optional[str] = None,
                         enable_correction: bool = True,
                         enable_summarization: bool = True) -> dict:
    """Process text from file"""
    
    # Read main text
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            input_text = f.read().strip()
    except Exception as e:
        print(f" Error reading file {input_file}: {e}")
        return None
    
    # Read reference text
    reference_text = None
    if reference_file:
        try:
            with open(reference_file, 'r', encoding='utf-8') as f:
                reference_text = f.read().strip()
        except Exception as e:
            print(f" Error reading reference file {reference_file}: {e}")
    
    # Read reference summary
    reference_summary = None
    if reference_summary_file:
        try:
            with open(reference_summary_file, 'r', encoding='utf-8') as f:
                reference_summary = f.read().strip()
        except Exception as e:
            print(f" Error reading reference summary file {reference_summary_file}: {e}")
    
    # Create orchestrator and process
    orchestrator = Orchestrator()
    
    results = await orchestrator.process_text(
        input_text=input_text,
        reference_text=reference_text,
        reference_summary=reference_summary,
        domain=domain,
        enable_correction=enable_correction,
        enable_summarization=enable_summarization
    )
    
    # Save results to separate folders
    if output_file and results:
        from pathlib import Path
        import json
        
        # Get filename without extension
        filename = Path(output_file).stem
        
        # Create directories if they don't exist
        Path("data/correction").mkdir(parents=True, exist_ok=True)
        Path("data/summary").mkdir(parents=True, exist_ok=True)
        Path("data/correction_metrics").mkdir(parents=True, exist_ok=True)
        Path("data/summary_metrics").mkdir(parents=True, exist_ok=True)
        
        # Save correction results
        if "corrected_text" in results and "input_text" in results:
            # Save corrected text to correction folder
            correction_file = f"data/correction/{filename}.txt"
            save_correction_results(results, correction_file)
            
            # Save correction metrics to correction_metrics folder
            correction_metrics_file = f"data/correction_metrics/{filename}.json"
            
            # Calculate processing time for correction
            start_time = results.get("start_time")
            end_time = results.get("end_time")
            processing_time_seconds = 0
            if start_time and end_time:
                try:
                    from datetime import datetime
                    start_dt = datetime.fromisoformat(start_time)
                    end_dt = datetime.fromisoformat(end_time)
                    processing_time_seconds = int((end_dt - start_dt).total_seconds())
                except:
                    processing_time_seconds = 0
            
            correction_metrics = {
                "original_text": results.get("original_text", ""),
                "corrected_text": results.get("corrected_text", ""),
                "reference_text": results.get("reference_text", ""),
                "metrics": results.get("metrics_correction", {}),
                "processing_time_seconds": processing_time_seconds,
                "best_prompt": results.get("best_correction_prompt", results.get("best_prompt_cor", "basic")),
                "temperature": results.get("correction_temperature", 0.7)
            }
            with open(correction_metrics_file, 'w', encoding='utf-8') as f:
                json.dump(correction_metrics, f, ensure_ascii=False, indent=2)
        
        # Save summary results
        if "summary" in results:
            # Extract reference_summary from state (orchestrator returns state)
            reference_summary = results.get("reference_summary") or results.get("state", {}).get("reference_summary", "")
            
            # Save summary text to summary folder (TXT file with text and brief metrics)
            summary_file = f"data/summary/{filename}.txt"
            save_summary_results(results, summary_file)
            
            # Save detailed summary metrics to summary_metrics folder (JSON with LLM explanations)
            summary_metrics_file = f"data/summary_metrics/{filename}.json"
            
            # Calculate processing time
            start_time = results.get("start_time")
            end_time = results.get("end_time")
            processing_time_seconds = 0
            if start_time and end_time:
                try:
                    from datetime import datetime
                    start_dt = datetime.fromisoformat(start_time)
                    end_dt = datetime.fromisoformat(end_time)
                    processing_time_seconds = int((end_dt - start_dt).total_seconds())
                except:
                    processing_time_seconds = 0
            
            summary_metrics = {
                "corrected_text": results.get("corrected_text", ""),
                "summary": results.get("summary", ""),
                "reference_summary": reference_summary,
                "metrics": results.get("summary_metrics", {}),
                "processing_time_seconds": processing_time_seconds,
                "best_prompt": results.get("best_summary_prompt", results.get("best_prompt_sum", "")),  # Изменено
                "temperature": results.get("summarization_temperature", 0.7)
            }
            with open(summary_metrics_file, 'w', encoding='utf-8') as f:
                json.dump(summary_metrics, f, ensure_ascii=False, indent=2)
        
        # Also save combined results to original output file
        save_results_to_file(results, output_file)
    
    return results

async def process_batch_files(domain: str = "general", 
                           enable_correction: bool = True, 
                           enable_summarization: bool = True,
                           metrics_manager: Optional[MetricsTableManager] = None) -> Optional[dict]:
    """
    Batch processing of files from inputs/ directory
    
    Args:
        domain: Text domain
        enable_correction: Enable correction
        enable_summarization: Enable summarization
        
    Returns:
        Dictionary with results of processing all files
    """
    ROOT_DIR = Path(__file__).parent
    DATA_DIR = ROOT_DIR / "data"
    CORRECTION_METRICS_DIR = DATA_DIR / "correction_metrics"
    SUMMARY_METRICS_DIR = DATA_DIR / "summary_metrics"
    LOGS_DIR = ROOT_DIR / "logs"
    INPUT_DIR = ROOT_DIR / "inputs"  # correct path for inputs folder
    inputs_dir = INPUT_DIR
    etalon_dir = inputs_dir / "etalon"
    incorrect_dir = inputs_dir / "incorrect"
    summary_etalon_dir = inputs_dir / "summary_etalon"
    
    results = {}
    all_files = []
    
    # Collect only incorrect files for processing
    if incorrect_dir.exists():
        incorrect_files = list(incorrect_dir.glob("*.txt"))
        for file_path in incorrect_files:
            all_files.append(("incorrect", file_path))
    
    print_separator("BATCH FILE PROCESSING")
    print(f" Found files: {len(all_files)}")
    print(f" Domain: {domain}")
    print(f" Correction: {'Enabled' if enable_correction else 'Disabled'}")
    print(f" Summarization: {'Enabled' if enable_summarization else 'Disabled'}")
    
    # Check ПРОПУСК_ОБРАБОТАННЫХ setting
    skip_processed = config.ПРОПУСК_ОБРАБОТАННЫХ
    print(f" Skip processed files: {'YES' if skip_processed else 'NO'}")
    
    if skip_processed == 1:
        # Check which files are already processed and process only missing ones
        print(" MODE: Skip already processed files, process only missing ones")
        
        # Check which files already have metrics
        processed_files = set()
        correction_metrics_dir = Path("data/correction_metrics")
        if correction_metrics_dir.exists():
            for json_file in correction_metrics_dir.glob("*.json"):
                processed_files.add(json_file.stem)
        
        print(f" Already processed files: {len(processed_files)}")
        for filename in processed_files:
            print(f"   - {filename}")
        
        # Filter out already processed files
        files_to_process = []
        for file_type, file_path in all_files:
            if file_path.stem not in processed_files:
                files_to_process.append((file_type, file_path))
        
        if not files_to_process:
            print(" All files already processed, loading existing metrics only...")
            import time
            time.sleep(3)  # Give web monitor time to load existing files
            print("Web monitor should now show all existing metrics files")
            return {}
        else:
            print(f" Processing {len(files_to_process)} missing files:")
            for file_type, file_path in files_to_process:
                print(f"   - {file_path.name}")
            
            # Update all_files to only include missing files
            all_files = files_to_process
    else:
        # Reprocess all files, clear existing data
        print(" MODE: Reprocess all files, clear existing data")
        # Clear existing metrics files
        import shutil
        data_dirs = ["data/correction_metrics", "data/summary_metrics", "data/correction", "data/summary"]
        for data_dir in data_dirs:
            if Path(data_dir).exists():
                shutil.rmtree(data_dir)
                Path(data_dir).mkdir(parents=True, exist_ok=True)
                print(f" Cleared and recreated: {data_dir}")
    
    # Process all files with numbering
    for i, (file_type, file_path) in enumerate(all_files, 1):
        print_separator(f"FILE #{i}: {file_path.name}")
        print(f" Type: {'Reference' if file_type == 'etalon' else 'Incorrect'}")
        print(f" Path: {file_path}")
        
        # Find corresponding files
        reference_file = None
        summary_file = None
        
        if file_type == "incorrect" and etalon_dir.exists():
            reference_candidates = list(etalon_dir.glob(f"{file_path.stem}.txt"))
            print(f" DEBUG: Looking for reference: {file_path.stem}.txt")
            print(f" DEBUG: Found candidates: {len(reference_candidates)}")
            if reference_candidates:
                reference_file = str(reference_candidates[0])
                print(f" Found reference: {Path(reference_file).name}")
            else:
                print(f" DEBUG: No reference found for {file_path.name}")
        
        if summary_etalon_dir.exists():
            summary_candidates = list(summary_etalon_dir.glob(f"{file_path.stem}.txt"))
            print(f" DEBUG: Looking for summary: {file_path.stem}.txt")
            print(f" DEBUG: Found summary candidates: {len(summary_candidates)}")
            if summary_candidates:
                summary_file = str(summary_candidates[0])
                print(f" Found reference summary: {Path(summary_file).name}")
            else:
                print(f" DEBUG: No reference summary found for {file_path.name}")
        
        print(" Processing...")
        
        # Only enable correction for incorrect files, not for etalon files
        file_enable_correction = enable_correction and (file_type == "incorrect")
        
        file_results = await process_text_file(
            input_file=str(file_path),
            reference_file=reference_file,
            reference_summary_file=summary_file,
            domain=domain,
            output_file=f"data/{file_path.stem}.json",
            enable_correction=file_enable_correction,
            enable_summarization=enable_summarization
        )
        
        if file_results:
            results[file_path.name] = file_results
            print_batch_results(file_results, file_type)
            print(f" File {file_path.name} processed successfully")
            
            # Update metrics in realtime store and table
            if file_results:
                try:
                    # Extract state from results for metrics
                    state = file_results.get("state", {})
                    
                    # Store in realtime store for web monitor
                    extract_and_store_metrics(state, file_path.name)
                    
                    # Update metrics table if manager is available
                    if metrics_manager:
                        metrics_manager.update_metrics_table(state, file_path.name)
                        print(f" Metrics table updated for {file_path.name}")
                except Exception as e:
                    print(f" Warning: Failed to update metrics: {e}")
        else:
            print(f" Error processing file {file_path.name}")
    
    return results

async def main():
    """Main function - batch processing only"""
    print_separator("CorSumAgentsAI - Batch File Processing System")
    
    # Initialize system start time and start web monitor
    from datetime import datetime
    system_start_time = datetime.now()
    print(f"System start time: {system_start_time}")
    
    # Store start time in realtime store
    try:
        from src.utils.realtime_metrics import realtime_store
        realtime_store.add_metrics("_system_start", {
            "system_start_time": system_start_time.isoformat(),
            "timestamp": system_start_time.isoformat()
        })
    except Exception as e:
        print(f"Warning: Could not store system start time: {e}")
    
    print("Starting Web Monitor...")
    start_web_monitor()
    
    # Reset web monitor time after a short delay to ensure it's running
    import time
    time.sleep(2)  # Wait for web monitor to start
    
    try:
        import urllib.request
        import json
        
        data = json.dumps({}).encode('utf-8')
        req = urllib.request.Request(
            'http://127.0.0.1:5000/api/reset_time',
            data=data,
            headers={'Content-Type': 'application/json'}
        )
        
        with urllib.request.urlopen(req, timeout=5) as response:
            if response.getcode() == 200:
                print("Web monitor time reset successfully")
            else:
                print(f"Warning: Failed to reset web monitor time: {response.getcode()}")
    except Exception as e:
        print(f"Warning: Could not reset web monitor time: {e}")
    
    # Initialize metrics table manager
    metrics_manager = MetricsTableManager()
    
    # Process all files from inputs/ directory
    results = await process_batch_files(
        domain="general",
        enable_correction=True,
        enable_summarization=True,
        metrics_manager=metrics_manager
    )
    
    # Output summary
    if results:
        print_separator("BATCH PROCESSING SUMMARY")
        for fname, res in results.items():
            status = " " if res else " "
            print(f"{status} {fname}")
    else:
        print("\n No files processed")
    
    # Wait for web monitor to process the last file
    print("\n Waiting for web monitor to process final files...")
    import time
    time.sleep(5)  # Give web monitor time to process the last file
    print("Web monitor should now show all processed files")

def format_text_sentences(text: str) -> str:
    """Format text with each sentence on new line"""
    if not text:
        return text
    
    # Split by sentence endings (., !, ?) followed by space or end
    import re
    sentences = re.split(r'([.!?]+)\s*', text)
    
    formatted_text = ""
    for i in range(0, len(sentences), 2):
        if i < len(sentences) - 1:
            sentence = sentences[i] + sentences[i+1]
        else:
            sentence = sentences[i]
        
        sentence = sentence.strip()
        if sentence:
            formatted_text += sentence + "\n"
    
    return formatted_text.strip()

if __name__ == "__main__":
    try:
        # Setup logging with date-based folders
        logger = setup_logging()
        
        import asyncio
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n Program interrupted by user")
    except Exception as e:
        print(f"\n Unexpected error: {e}")
        import sys
        sys.exit(1)
