# Batch Processing Guide

## Overview

The batch processor allows you to process multiple files from the `inputs/` directory structure with custom output formatting for correction and summarization metrics.

## Directory Structure

```
inputs/
|-- incorrect/           # Files with errors to correct
|   |-- sample1.txt
|   |-- sample2.txt
|   `-- ...
|-- etalon/             # Correct reference files (optional)
|   |-- sample1.txt
|   |-- sample2.txt
|   `-- ...
`-- summary_etalon/     # Reference summaries (optional)
    |-- sample1.txt
    |-- sample2.txt
    `-- ...
```

## Usage

### Basic Usage

```bash
# Process all files with correction and summarization
python batch_processor.py

# Show only file statistics
python batch_processor.py --stats

# Process only correction (skip summarization)
python batch_processor.py --no-summarization

# Process only summarization (skip correction)
python batch_processor.py --no-correction

# Use custom input directory
python batch_processor.py --input-dir /path/to/inputs
```

### Examples

```bash
# Check what files are available
python batch_processor.py --stats

### Slow processing

```bashbashbashtextpython
python batch_processor.py
```

# Only correction phase
```bash
python batch_processor.py --no-summarization
```

# Only summarization phase
```bash
python batch_processor.py --no-correction
```

## Output Format

### Correction Results (`data/correction/`)

Each corrected file generates output in the format:

```
=== INCORRECT_TEXT ===
[original text with errors]

=== CORRECT_TEXT ===
[corrected text]

=== ETALON_TEXT ===
[reference text if available]

=== METRICS ===
WER_0: 0.259
WER: 0.079
delta_WER: 0.180
LevRating_0: 0.955
LevRating: 0.985
delta_LEV: 0.030
PROMPT: 1
Time: 46.795 s, time per prompt: 9.359 s.
```

### Summary Results (`data/summary/`)

Each summary generates output in the format:

```
=== CORRECT_TEXT ===
[corrected input text]

=== SUMMARY_TEXT ===
[generated summary]

=== ETALON_SUMMARY ===
[reference summary if available]

=== METRICS ===
METEOR: 0.403
BLEU: 0.185
ROUGE-1: 0.422
ROUGE-2: 0.219
ROUGE-L: 0.410
PROMPT: 5
Time: 16.83 s, time per prompt: 3.366 s.
```

## Metrics Explained

### Correction Metrics

- **WER_0**: Word Error Rate of original text vs reference
- **WER**: Word Error Rate of corrected text vs reference
- **delta_WER**: Improvement in WER (WER_0 - WER)
- **LevRating_0**: Levenshtein similarity of original vs reference
- **LevRating**: Levenshtein similarity of corrected vs reference
- **delta_LEV**: Improvement in Levenshtein similarity

### Summary Metrics

- **METEOR**: Metric for Evaluation of Translation with Explicit ORdering
- **BLEU**: Bilingual Evaluation Understudy
- **ROUGE-1**: Recall-Oriented Understudy for Gisting Evaluation (unigram)
- **ROUGE-2**: ROUGE with bigrams
- **ROUGE-L**: ROUGE with longest common subsequence

## File Naming

The processor automatically matches files across directories:

- `inputs/incorrect/sample1.txt` matches with:
  - `inputs/etalon/sample1.txt` (reference for correction)
  - `inputs/summary_etalon/sample1.txt` (reference for summary)

Supported naming patterns for reference files:
- Same name as incorrect file
- `{name}_etalon.{ext}`
- `{name}_correct.{ext}`

## Requirements

1. **LM Studio** must be running with a model loaded
2. **Input files** should be placed in `inputs/incorrect/`
3. **Reference files** are optional but recommended for accurate metrics
4. **Dependencies** must be installed: `pip install -r requirements.txt`
5. **Troubleshooting**

### No Files Found

```bash
No files found to process
```

- Check that files exist in `inputs/incorrect/`
- Verify file extensions (.txt, .md, .text)

### LM Studio Connection Error

```bash
Error: Could not establish connection to LM Studio
```

- Ensure LM Studio is running
- Check that server is on port 1234

### Memory Issues

```bash
CUDA out of memory
```

- Use a smaller model in LM Studio
- Process files one at a time
- Reduce batch size in configuration

## Performance Tips

1. **GPU Acceleration**: Use CUDA-enabled GPU for faster processing
2. **Model Selection**: Choose appropriate model size for your hardware
3. **Batch Processing**: Process multiple files sequentially for efficiency
4. **Reference Files**: Provide reference files for accurate metrics

## Integration with Main System

The batch processor uses the same core components as the main system:
- Same agents and metrics
- Same configuration options
- Compatible output formats

You can use both systems interchangeably based on your needs:
- Use `main.py` for single file processing
- Use `batch_processor.py` for multiple files
