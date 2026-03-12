import json
import os
import sys
from collections import Counter
from typing import Dict, List, Any

def analyze_performance(results_path: str):
    if not os.path.exists(results_path):
        print(f"Error: File not found: {results_path}")
        return

    with open(results_path, 'r') as f:
        data = json.load(f)

    total_docs = len(data)
    if total_docs == 0:
        print("No documents found in results.")
        return

    # Global Metrics
    decisions = [d.get('decision') for d in data]
    decision_counts = Counter(decisions)
    
    confidences = [d.get('confidence', 0) for d in data]
    avg_confidence = sum(confidences) / total_docs if total_docs > 0 else 0

    # Field-Level Metrics
    fields = ['bank_name', 'customer_name', 'loan_amount', 'disbursed_amount', 'branch_id']
    field_stats = {}
    for field in fields:
        extracted = [d for d in data if d.get(field) is not None]
        count = len(extracted)
        rate = (count / total_docs) * 100
        field_stats[field] = {
            'count': count,
            'rate': rate
        }

    # Bank-Specific Metrics
    bank_data = {}
    for d in data:
        bank = d.get('bank_name') or 'Unknown'
        if bank not in bank_data:
            bank_data[bank] = {'count': 0, 'confidences': [], 'decisions': []}
        bank_data[bank]['count'] += 1
        bank_data[bank]['confidences'].append(d.get('confidence', 0))
        bank_data[bank]['decisions'].append(d.get('decision'))

    # Failure Mode Categorization
    no_text = [d for d in data if not d.get('full_text')]
    low_conf_rejects = [d for d in data if d.get('decision') == 'reject' and d.get('confidence', 0) < 0.3]

    # --- REPORTING ---
    print("="*60)
    print(f"OCR PERFORMANCE REPORT: {os.path.basename(results_path)}")
    print("="*60)
    
    print(f"\nGLOBAL METRICS")
    print(f"{'-'*20}")
    print(f"Total Documents: {total_docs}")
    for dec, count in decision_counts.items():
        print(f"Decision {dec:10}: {count:3} ({count/total_docs*100:5.1f}%)")
    print(f"Average Confidence: {avg_confidence:.3f}")

    print(f"\nFIELD EXTRACTION RATES")
    print(f"{'-'*20}")
    for field, stats in field_stats.items():
        print(f"{field:18}: {stats['count']:3} ({stats['rate']:5.1f}%)")

    print(f"\nBANK-WISE ANALYSIS (Top 10)")
    print(f"{'-'*20}")
    # Sort banks by count
    sorted_banks = sorted(bank_data.items(), key=lambda x: x[1]['count'], reverse=True)[:10]
    print(f"{'Bank Name':25} | {'Docs':4} | {'Avg Conf':8} | {'Accept %':8}")
    for bank, stats in sorted_banks:
        avg_c = sum(stats['confidences']) / len(stats['confidences'])
        acc_p = (stats['decisions'].count('accept') / len(stats['decisions'])) * 100
        print(f"{bank[:25]:25} | {stats['count']:4} | {avg_c:8.3f} | {acc_p:7.1f}%")

    print(f"\nFAILURE CATEGORIES")
    print(f"{'-'*20}")
    print(f"No Text Detected (OCR Failure): {len(no_text):3} ({len(no_text)/total_docs*100:5.1f}%)")
    print(f"Low Confidence Rejects        : {len(low_conf_rejects):3} ({len(low_conf_rejects)/total_docs*100:5.1f}%)")
    
    print("\n" + "="*60)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python analyze_performance.py <results_json_path>")
    else:
        analyze_performance(sys.argv[1])
