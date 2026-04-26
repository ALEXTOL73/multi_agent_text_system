#!/usr/bin/env python3
"""
Web Monitor for CorSumAgentsAI Metrics – финальная версия с диагностикой
Пути: inputs (с s), data/correction_metrics, data/summary_metrics
"""

import json
import logging
from pathlib import Path
from datetime import datetime
from flask import Flask, jsonify, render_template_string, send_file
import io

logging.getLogger('werkzeug').setLevel(logging.ERROR)
app = Flask(__name__)

# ---------- ПУТИ (исправлено: inputs, а не input) ----------
ROOT_DIR = Path(__file__).parent
DATA_DIR = ROOT_DIR / "data"
CORRECTION_METRICS_DIR = DATA_DIR / "correction_metrics"
SUMMARY_METRICS_DIR = DATA_DIR / "summary_metrics"
INPUT_DIR = ROOT_DIR / "inputs"          # ← здесь s на конце


# ---------- Глобальные переменные ----------
system_start_time = datetime.now()
monitoring_start_time = None  # Время начала обработки
monitoring_active = True
processing_complete = False

def reset_monitor_time():
    """Сбрасывает время монитора до текущего момента"""
    global system_start_time
    system_start_time = datetime.now()
    print(f"Monitor time reset to: {system_start_time}")

def start_monitoring_time():
    """Начинает время мониторинга при первой метрике"""
    global monitoring_start_time, processing_complete
    if monitoring_start_time is None:
        monitoring_start_time = datetime.now()
        processing_complete = False
        print(f"Monitoring started at: {monitoring_start_time}")

def stop_monitoring_time():
    """Останавливает время мониторинга после последнего файла"""
    global processing_complete
    processing_complete = True
    print(f"Monitoring stopped - processing complete")

def load_metrics_from_files():
    """Загружает метрики из JSON-файлов (поддерживает оба формата)."""
    metrics_dict = {}
    
    # Проверяем наличие папки
    if not CORRECTION_METRICS_DIR.exists():
        return metrics_dict
    
    json_files = list(CORRECTION_METRICS_DIR.glob("*.json"))
    
    for file in json_files:
        test_id = file.stem
        try:
            with open(file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Пробуем извлечь метрики из разных возможных ключей
            # В примере 1_0049... часть метрик лежит в "metrics_correction", часть в "summary_metrics"
            m = data.get('metrics_correction', data.get('metrics', {}))
            
            metrics = {
                'delta_WER': m.get('delta_WER', m.get('delta_wer', 0)),
                'delta_Lev': m.get('delta_LEV', m.get('delta_lev', 0)),
                'LevRating_after': m.get('LevRating', m.get('lev_corrected', 0)),
                'Perplexity': m.get('perplexity', 0),
                'CorScore': m.get('CorScore', m.get('cor_score', 0)),
                'processing_time_seconds': data.get('processing_time_seconds', 0),
                'timestamp': datetime.fromtimestamp(file.stat().st_mtime).isoformat()
            }
            
            # Метрики суммаризации – если есть в том же файле
            if 'summary_metrics' in data and data['summary_metrics']:
                sm = data['summary_metrics']
                metrics.update({
                    'G_Eval': sm.get('geval', 0),
                    'METEOR': sm.get('meteor', 0),
                    'LLM_Judge': sm.get('llm_judge', 0),
                    'BertScore': sm.get('bert_score', 0),
                    'SumScore': sm.get('sum_score', 0)
                })
            elif 'metrics' in data and any(key in data['metrics'] for key in ['geval', 'meteor', 'llm_judge']):
                # Альтернативный формат: все метрики в одном словаре metrics
                metrics.update({
                    'G_Eval': data['metrics'].get('geval', 0),
                    'METEOR': data['metrics'].get('meteor', 0),
                    'LLM_Judge': data['metrics'].get('llm_judge', 0),
                    'BertScore': data['metrics'].get('bert_score', 0),
                    'SumScore': data['metrics'].get('sum_score', 0)
                })
            else:
                # Добавим значения по умолчанию для summary метрик
                metrics.update({
                    'G_Eval': 0,
                    'METEOR': 0,
                    'LLM_Judge': 0,
                    'BertScore': 0,
                    'SumScore': 0
                })
            
            metrics_dict[test_id] = metrics
        except Exception as e:
            pass
    
    # Если есть отдельные файлы суммаризации в summary_metrics
    if SUMMARY_METRICS_DIR.exists():
        for file in SUMMARY_METRICS_DIR.glob("*.json"):
            test_id = file.stem
            try:
                with open(file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                m = data.get('metrics', {})
                if m:  # Проверяем что метрики не пустые
                    summary = {
                        'G_Eval': m.get('geval', 0),
                        'METEOR': m.get('meteor', 0),
                        'LLM_Judge': m.get('llm_judge', 0),
                        'BertScore': m.get('bert_score', 0),
                        'SumScore': m.get('sum_score', 0)
                    }
                    if test_id in metrics_dict:
                        metrics_dict[test_id].update(summary)
                    else:
                        # Если нет correction метрик, создаем новую запись с summary метриками
                        metrics_dict[test_id] = {
                            'delta_WER': 0,
                            'delta_Lev': 0,
                            'LevRating_after': 0,
                            'Perplexity': 0,
                            'CorScore': 0,
                            'timestamp': datetime.fromtimestamp(file.stat().st_mtime).isoformat(),
                            **summary
                        }
            except Exception as e:
                pass
    
    # Sort metrics by document number
    def extract_doc_number(test_id):
        """Extract document number from test_id for sorting."""
        try:
            import re
            match = re.search(r'^(\d+)_', test_id)
            if match:
                return int(match.group(1))
            return 0
        except Exception:
            return 0
    
    # Sort dictionary by document number
    sorted_items = sorted(metrics_dict.items(), key=lambda x: extract_doc_number(x[0]))
    return dict(sorted_items)

def get_total_input_files():
    """Считает количество входных .txt файлов в inputs/incorrect/ или inputs/"""
    total = 0
    incorrect_dir = INPUT_DIR / "incorrect"
    if incorrect_dir.exists():
        total = len(list(incorrect_dir.glob("*.txt")))
    else:
        total = len(list(INPUT_DIR.glob("*.txt")))
    return total

# ---------- Flask маршруты ----------
@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)  # HTML такой же, как в вашем файле (я его не меняю)

# Для экономии места – HTML-шаблон можно оставить как в вашем файле, он рабочий.
# Я продублирую его здесь, чтобы код был самодостаточным.

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="ru">
<head>
    <meta charset="UTF-8">
    <title>CorSumAgentsAI - Мониторинг</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; background: #f4f4f4; }
        .header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 20px; }
        .runtime-display { font-size: 32px; font-weight: bold; color: #2196f3; font-family: monospace; }
        .status-bar { display: flex; gap: 20px; flex-wrap: wrap; margin-bottom: 20px; }
        .status-card { background: white; padding: 15px; border-radius: 8px; flex: 1; min-width: 150px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        .progress-bar { width: 100%; height: 30px; background: #ddd; border-radius: 15px; overflow: hidden; margin-top: 10px; }
        .progress-fill { height: 100%; background: #4caf50; width: 0%; text-align: center; line-height: 30px; color: white; }
        .charts-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(400px, 1fr)); gap: 20px; margin: 20px 0; }
        .chart-card { background: white; border-radius: 8px; padding: 15px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        .table-container { overflow-x: auto; margin: 20px 0; background: white; border-radius: 8px; padding: 15px; }
        table { width: 100%; border-collapse: collapse; }
        th, td { padding: 8px; text-align: left; border-bottom: 1px solid #ddd; font-size: 12px; }
        th { background: #f2f2f2; }
        button { padding: 10px 20px; background: #28a745; color: white; border: none; border-radius: 4px; cursor: pointer; margin-left: 10px; }
        button:hover { background: #218838; }
    </style>
</head>
<body>
    <div class="header">
        <h1>📊 CorSumAgentsAI - Мониторинг метрик</h1>
        <div style="display: flex; align-items: center;">
            <div class="runtime-display" id="runtime">00:00:00</div>
            <button onclick="exportExcel()">📎 Экспорт Excel</button>
            <button onclick="refreshData()">🔄 Обновить</button>
        </div>
    </div>
    <div class="status-bar">
        <div class="status-card"><h3>Статус</h3><p id="status">-</p></div>
        <div class="status-card"><h3>Всего файлов</h3><p id="total">0</p></div>
        <div class="status-card"><h3>Обработано</h3><p id="processed">0</p></div>
        <div class="status-card"><h3>Осталось</h3><p id="remaining">0</p></div>
        <div class="status-card"><h3>Успешно/Неуспешно</h3><p id="success-rate">-</p></div>
        <div class="status-card"><h3>Прогресс</h3><div class="progress-bar"><div class="progress-fill" id="progress-fill">0%</div></div></div>
    </div>
    <div class="charts-grid">
        <div class="chart-card"><h3>ΔWER</h3><canvas id="chart-delta-wer"></canvas></div>
        <div class="chart-card"><h3>LevRating</h3><canvas id="chart-levrating"></canvas></div>
        <div class="chart-card"><h3>Perplexity</h3><canvas id="chart-perplexity"></canvas></div>
        <div class="chart-card"><h3>CorScore</h3><canvas id="chart-corscore"></canvas></div>
        <div class="chart-card"><h3>G-Eval</h3><canvas id="chart-geval"></canvas></div>
        <div class="chart-card"><h3>METEOR</h3><canvas id="chart-meteor"></canvas></div>
        <div class="chart-card"><h3>LLM-Judge</h3><canvas id="chart-llmjudge"></canvas></div>
        <div class="chart-card"><h3>BertScore</h3><canvas id="chart-bertscore"></canvas></div>
        <div class="chart-card"><h3>SumScore</h3><canvas id="chart-sumscore"></canvas></div>
    </div>
    <div class="table-container">
        <table id="metrics-table">
            <thead><tr><th>Название файла</th><th>ΔWER</th><th>ΔLev</th><th>LevRating</th><th>Perplexity</th><th>CorScore</th><th>G-Eval</th><th>METEOR</th><th>LLM-Judge</th><th>BertScore</th><th>SumScore</th><th>Время (с)</th></tr></thead>
            <tbody></tbody>
        </table>
    </div>
    <script>
        let charts = {};
        let startTime = null;
        let runtimeInterval = null;
        let isProcessingComplete = false;
        
        function initializeRuntime() {
            if (!startTime) {
                startTime = new Date();
                runtimeInterval = setInterval(updateRuntime, 1000);
            }
        }
        
        function updateRuntime() {
            if (!startTime) return;
            
            const elapsed = Math.floor((new Date() - startTime) / 1000);
            const hours = String(Math.floor(elapsed / 3600)).padStart(2,'0');
            const minutes = String(Math.floor((elapsed % 3600) / 60)).padStart(2,'0');
            const seconds = String(elapsed % 60).padStart(2,'0');
            document.getElementById('runtime').innerText = `${hours}:${minutes}:${seconds}`;
        }
        
        async function refreshData() {
            try {
                const res = await fetch('/api/status');
                const data = await res.json();
                document.getElementById('status').innerText = data.status;
                document.getElementById('total').innerText = data.total_files;
                document.getElementById('processed').innerText = data.processed_files;
                document.getElementById('remaining').innerText = data.remaining_files;
                
                // Calculate success rate based on thresholds
                if (data.metrics_list && data.metrics_list.length > 0) {
                    const thresholds = {
                        delta_WER: 0.1,
                        LevRating_after: 0.8,
                        CorScore: 0.5,
                        G_Eval: 0.5,
                        METEOR: 0.25,
                        LLM_Judge: 6,
                        BertScore: 0.7,
                        SumScore: 0.6
                    };
                    
                    let successful = 0;
                    let unsuccessful = 0;
                    
                    for (let m of data.metrics_list) {
                        let isSuccessful = true;
                        
                        // Check if any metric is below threshold (for delta_WER, lower is worse - means less improvement)
                        if ((m.delta_WER || 0) < thresholds.delta_WER) isSuccessful = false;
                        if ((m.LevRating_after || 0) < thresholds.LevRating_after) isSuccessful = false;
                        if ((m.CorScore || 0) < thresholds.CorScore) isSuccessful = false;
                        if ((m.G_Eval || 0) < thresholds.G_Eval) isSuccessful = false;
                        if ((m.METEOR || 0) < thresholds.METEOR) isSuccessful = false;
                        if ((m.LLM_Judge || 0) < thresholds.LLM_Judge) isSuccessful = false;
                        if ((m.BertScore || 0) < thresholds.BertScore) isSuccessful = false;
                        if ((m.SumScore || 0) < thresholds.SumScore) isSuccessful = false;
                        
                        if (isSuccessful) {
                            successful++;
                        } else {
                            unsuccessful++;
                        }
                    }
                    
                    document.getElementById('success-rate').innerText = `${successful}/${unsuccessful}`;
                } else {
                    document.getElementById('success-rate').innerText = '-';
                }
                const progress = data.progress_percentage || 0;
                document.getElementById('progress-fill').style.width = progress + '%';
                document.getElementById('progress-fill').innerText = progress.toFixed(1) + '%';
                
                if (data.metrics_list && data.metrics_list.length) {
                    updateCharts(data.metrics_list);
                    updateTable(data.metrics_list);
                }
            } catch(e) { console.error(e); }
        }
        
        function updateCharts(metricsList) {
            const labels = metricsList.map(m => m.id);
            function makeChart(id, data, label, color, minY, maxY, threshold = null) {
                const ctx = document.getElementById(id).getContext('2d');
                if (charts[id]) charts[id].destroy();
                
                const datasets = [{
                    label: label,
                    data: data,
                    borderColor: color,
                    fill: false
                }];
                
                // Add threshold line if provided
                if (threshold !== null) {
                    datasets.push({
                        label: `Порог: ${threshold}`,
                        data: Array(data.length).fill(threshold),
                        borderColor: 'red',
                        borderDash: [5, 5],
                        fill: false,
                        pointRadius: 0
                    });
                }
                
                charts[id] = new Chart(ctx, {
                    type: 'line',
                    data: { labels, datasets },
                    options: { 
                        responsive: true,
                        scales: { y: { min: minY, max: maxY } },
                        plugins: {
                            legend: {
                                display: true,
                                position: 'top'
                            }
                        }
                    }
                });
            }
            makeChart('chart-delta-wer', metricsList.map(m => m.delta_WER), 'ΔWER', '#f44336', -0.5, 1, 0.1);
            makeChart('chart-levrating', metricsList.map(m => m.LevRating_after), 'LevRating', '#4caf50', 0, 1, 0.7);
            makeChart('chart-perplexity', metricsList.map(m => m.Perplexity), 'Perplexity', '#9e9e9e', 0, 50, 10);
            makeChart('chart-corscore', metricsList.map(m => m.CorScore), 'CorScore', '#ff9800', -1, 1, 0);
            makeChart('chart-geval', metricsList.map(m => m.G_Eval), 'G-Eval', '#2196f3', 0, 1, 0.5);
            makeChart('chart-meteor', metricsList.map(m => m.METEOR), 'METEOR', '#9c27b0', 0, 1, 0.25);
            makeChart('chart-llmjudge', metricsList.map(m => m.LLM_Judge), 'LLM-Judge', '#ff5722', 0, 10, 6);
            makeChart('chart-sumscore', metricsList.map(m => m.SumScore), 'SumScore', '#009688', 0, 1, 0.6);
            makeChart('chart-bertscore', metricsList.map(m => m.BertScore), 'BertScore', '#795548', 0, 1, 0.7);
        }
        
        function updateTable(metricsList) {
            const tbody = document.querySelector('#metrics-table tbody');
            tbody.innerHTML = '';
            
            // Define thresholds for red highlighting
            const thresholds = {
                delta_WER: 0.1,      // Higher is worse
                delta_Lev: 0,        // Lower is worse
                LevRating_after: 0.7, // Lower is worse
                Perplexity: 10,      // Higher is worse
                CorScore: 0.5,       // Lower is worse
                G_Eval: 0.5,         // Lower is worse
                METEOR: 0.25,         // Lower is worse
                LLM_Judge: 6,        // Lower is worse
                BertScore: 0.7,     // Lower is worse
                SumScore: 0.6        // Lower is worse
            };
            
            // Add data rows
            for (let m of metricsList) {
                const row = tbody.insertRow();
                row.insertCell(0).innerText = m.id;
                
                // Helper function to create colored cell
                function createColoredCell(value, threshold, isHigherWorse = false) {
                    const cell = row.insertCell();
                    cell.innerText = value;
                    if ((isHigherWorse && value > threshold) || (!isHigherWorse && value < threshold)) {
                        cell.style.color = 'red';
                        cell.style.fontWeight = 'bold';
                    }
                    return cell;
                }
                
                createColoredCell((m.delta_WER || 0).toFixed(4), thresholds.delta_WER, false);
                createColoredCell((m.delta_Lev || 0).toFixed(4), thresholds.delta_Lev, false);
                createColoredCell((m.LevRating_after || 0).toFixed(4), thresholds.LevRating_after, false);
                createColoredCell((m.Perplexity || 0).toFixed(4), thresholds.Perplexity, true);
                createColoredCell((m.CorScore || 0).toFixed(4), thresholds.CorScore, false);
                createColoredCell((m.G_Eval || 0).toFixed(4), thresholds.G_Eval, false);
                createColoredCell((m.METEOR || 0).toFixed(4), thresholds.METEOR, false);
                createColoredCell((m.LLM_Judge || 0).toFixed(1), thresholds.LLM_Judge, false);
                createColoredCell((m.BertScore || 0).toFixed(4), thresholds.BertScore, false);
                createColoredCell((m.SumScore || 0).toFixed(4), thresholds.SumScore, false);
                
                // Add processing time column
                const timeCell = row.insertCell();
                const processingTime = m.processing_time_seconds || 0;
                timeCell.innerText = processingTime + 's';
            }
            
            // Add average row if there are metrics
            if (metricsList.length > 0) {
                const avgRow = tbody.insertRow();
                avgRow.style.fontWeight = 'bold';
                avgRow.style.backgroundColor = '#f0f0f0';
                
                // Calculate averages
                const avgDeltaWER = metricsList.reduce((sum, m) => sum + (m.delta_WER || 0), 0) / metricsList.length;
                const avgDeltaLev = metricsList.reduce((sum, m) => sum + (m.delta_Lev || 0), 0) / metricsList.length;
                const avgLevRating = metricsList.reduce((sum, m) => sum + (m.LevRating_after || 0), 0) / metricsList.length;
                const avgPerplexity = metricsList.reduce((sum, m) => sum + (m.Perplexity || 0), 0) / metricsList.length;
                const avgCorScore = metricsList.reduce((sum, m) => sum + (m.CorScore || 0), 0) / metricsList.length;
                const avgGEval = metricsList.reduce((sum, m) => sum + (m.G_Eval || 0), 0) / metricsList.length;
                const avgMeteor = metricsList.reduce((sum, m) => sum + (m.METEOR || 0), 0) / metricsList.length;
                const avgLLMJudge = metricsList.reduce((sum, m) => sum + (m.LLM_Judge || 0), 0) / metricsList.length;
                const avgBertScore = metricsList.reduce((sum, m) => sum + (m.BertScore || 0), 0) / metricsList.length;
                const avgSumScore = metricsList.reduce((sum, m) => sum + (m.SumScore || 0), 0) / metricsList.length;
                const avgProcessingTime = metricsList.reduce((sum, m) => sum + (m.processing_time_seconds || 0), 0) / metricsList.length;
                
                // Helper function to create colored average cell
                function createColoredAvgCell(value, threshold, isHigherWorse = false) {
                    const cell = avgRow.insertCell();
                    cell.innerText = value;
                    if ((isHigherWorse && value > threshold) || (!isHigherWorse && value < threshold)) {
                        cell.style.color = 'red';
                    }
                    return cell;
                }
                
                avgRow.insertCell(0).innerText = 'Среднее';
                createColoredAvgCell(avgDeltaWER.toFixed(4), thresholds.delta_WER, false);
                createColoredAvgCell(avgDeltaLev.toFixed(4), thresholds.delta_Lev, false);
                createColoredAvgCell(avgLevRating.toFixed(4), thresholds.LevRating_after, false);
                createColoredAvgCell(avgPerplexity.toFixed(4), thresholds.Perplexity, true);
                createColoredAvgCell(avgCorScore.toFixed(4), thresholds.CorScore, false);
                createColoredAvgCell(avgGEval.toFixed(4), thresholds.G_Eval, false);
                createColoredAvgCell(avgMeteor.toFixed(4), thresholds.METEOR, false);
                createColoredAvgCell(avgLLMJudge.toFixed(1), thresholds.LLM_Judge, false);
                createColoredAvgCell(avgBertScore.toFixed(4), thresholds.BertScore, false);
                createColoredAvgCell(avgSumScore.toFixed(4), thresholds.SumScore, false);
                
                // Add average processing time
                const avgTimeCell = avgRow.insertCell();
                avgTimeCell.innerText = Math.round(avgProcessingTime) + 's';
            }
        }
        
        async function exportExcel() {
            const response = await fetch('/api/export_excel');
            const blob = await response.blob();
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement('a');
            const now = new Date();
            const dateStr = `${String(now.getDate()).padStart(2,'0')}${String(now.getMonth()+1).padStart(2,'0')}${String(now.getFullYear()).slice(2)}`;
            a.download = `Table_metrics_${dateStr}.xlsx`;
            a.href = url;
            a.click();
            window.URL.revokeObjectURL(url);
        }
        
        // Initialize runtime timer when page loads
        initializeRuntime();
        
        refreshData();
        setInterval(refreshData, 15000);
    </script>
</body>
</html>
"""

@app.route('/api/status')
def api_status():
    metrics = load_metrics_from_files()
    processed = len(metrics)
    total = get_total_input_files()
    remaining = max(0, total - processed)
    progress = (processed / total * 100) if total > 0 else 0
    
    metrics_list = []
    for test_id, m in metrics.items():
        metrics_list.append({
            'id': test_id,
            'delta_WER': m.get('delta_WER', 0),
            'delta_Lev': m.get('delta_Lev', 0),
            'LevRating_after': m.get('LevRating_after', 0),
            'Perplexity': m.get('Perplexity', 0),
            'CorScore': m.get('CorScore', 0),
            'G_Eval': m.get('G_Eval', 0),
            'METEOR': m.get('METEOR', 0),
            'LLM_Judge': m.get('LLM_Judge', 0),
            'SumScore': m.get('SumScore', 0),
            'BertScore': m.get('BertScore', 0),
            'processing_time_seconds': m.get('processing_time_seconds', 0)
        })
    
    # Additional sorting by document number for charts
    def extract_doc_number(test_id):
        """Extract document number from test_id for sorting."""
        try:
            import re
            match = re.search(r'^(\d+)_', test_id)
            if match:
                return int(match.group(1))
            return 0
        except Exception:
            return 0
    
    metrics_list.sort(key=lambda x: extract_doc_number(x['id']))
    
    # Используем время мониторинга или системное время как запасной вариант
    if monitoring_start_time and not processing_complete:
        elapsed = (datetime.now() - monitoring_start_time).total_seconds()
        status = 'running'
    elif processing_complete:
        elapsed = (monitoring_start_time and (datetime.now() - monitoring_start_time).total_seconds()) or 0
        status = 'completed'
    else:
        elapsed = (datetime.now() - system_start_time).total_seconds()
        status = 'idle'
        
    return jsonify({
        'status': status,
        'total_files': total,
        'processed_files': processed,
        'remaining_files': remaining,
        'progress_percentage': progress,
        'metrics_list': metrics_list,
        'server_elapsed': elapsed,
        'last_update': datetime.now().isoformat()
    })

@app.route('/api/export_excel')
def export_excel():
    metrics = load_metrics_from_files()
    try:
        import pandas as pd
        data = []
        for test_id, m in metrics.items():
            data.append({
                'filename': test_id,
                'delta_WER': m.get('delta_WER', 0),
                'LevRating_after': m.get('LevRating', m.get('lev_corrected', 0)),
                'delta_Lev': m.get('delta_Lev', 0),
                'Perplexity': m.get('Perplexity', 0),
                'CorScore': m.get('CorScore', m.get('cor_score', 0)),
                'G_Eval': m.get('G_Eval', 0),
                'METEOR': m.get('METEOR', 0),
                'LLM_Judge': m.get('LLM_Judge', 0),
                'BertScore': m.get('BertScore', 0),
                'SumScore': m.get('SumScore', 0)
            })
        df = pd.DataFrame(data)
        
        # Sort by document number before saving
        def extract_doc_number(filename):
            """Extract document number from filename for sorting."""
            try:
                import re
                match = re.search(r'^(\d+)_', filename)
                if match:
                    return int(match.group(1))
                return 0
            except Exception:
                return 0
        
        df["doc_number"] = df["filename"].apply(extract_doc_number)
        df = df.sort_values("doc_number", ascending=True)
        df = df.drop(columns=["doc_number"])
        
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name='Sheet1', index=False)
        output.seek(0)
        return send_file(output, mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                         as_attachment=True, download_name=f'Table_metrics_{datetime.now().strftime("%d%m%y")}.xlsx')
    except ImportError:
        return jsonify({'error': 'pandas or openpyxl not installed'}), 500

@app.route('/api/reset_time', methods=['POST'])
def reset_time():
    reset_monitor_time()
    return jsonify({'success': True})

@app.route('/api/start_monitoring', methods=['POST'])
def start_monitoring():
    start_monitoring_time()
    return jsonify({'success': True})

@app.route('/api/stop_monitoring', methods=['POST'])
def stop_monitoring():
    stop_monitoring_time()
    return jsonify({'success': True})

if __name__ == '__main__':
    print("\n🌐 WebMonitor запущен на http://127.0.0.1:5000")
    app.run(debug=False, host='127.0.0.1', port=5000, use_reloader=False)