import os
from pathlib import Path    

def write_header(eval_path):
    if not os.path.exists(eval_path):
        os.makedirs(os.path.dirname(eval_path), exist_ok=True)
    with open(eval_path, 'w') as f:
        f.write(f"\n\n=== Begin Evaluation ===\n")
        f.write('========================================\n\n\n')


def write_domain_header(eval_path, domain):
    with open(eval_path, 'a') as f:
        f.write(f"=== Domain: {domain} ===\n")
        f.write('========================================\n\n\n')

def write_topic_header(eval_path, topic):
    with open(eval_path, 'a') as f:
        f.write(f"=== TOPIC: {topic} ===\n")
        f.write('========================================\n\n\n')


def write_domain_result(eval_path, domain, results):
    with open(eval_path, 'a') as f:
        f.write(f"=== {domain} Result ===\n")
        f.write('========================================\n\n\n')
    if len(results) == 0:
        return
    final_result = {}
    results_count = len(results)
    for result in results:
        for key, value in result.items():
            if key not in final_result:
                final_result[key] = value
            else:
                final_result[key] += value
    for key in final_result:
        final_result[key] /= results_count
        
    with open(eval_path, 'a') as f:
        for key, value in final_result.items():
            f.write(f"{key}: {value}\n")

        f.write('Domain End\n\n\n')

def write_result(eval_path, description, results, reasons):
    with open(eval_path, 'a') as f:
        f.write(f"=== {description} Result ===\n")
        f.write('========================================\n')
        for key, value in results.items():
            f.write(f"{key}: {value}\n")
            if key in reasons:
                f.write(f"  Reason: {reasons[key]}\n")
        f.write('\n')
