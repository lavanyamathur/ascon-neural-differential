# bit_sweep_compare.py
import subprocess, re, csv

MODELS = ["gohr_cnn", "mlp"]
WORDS = [0, 4]
BITS = [0, 1, 2, 3, 5, 8, 13, 21, 34, 50, 63]

results = []
for model in MODELS:
    for w in WORDS:
        for b in BITS:
            proc = subprocess.run(
                ["python", "saliency_analysis.py", "--model", model,
                 "--delta_word", str(w), "--delta_bit", str(b)],
                capture_output=True, text=True
            )
            out = proc.stdout + proc.stderr
            m = re.search(r"using (\d+) correctly-classified", out)
            count = int(m.group(1)) if m else 0
            print(f"model={model} word={w} bit={b} -> correct={count}")
            results.append((model, w, b, count))

with open("bit_sweep_compare_results.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["model", "word", "bit", "correct_count"])
    writer.writerows(results)

print("Saved to bit_sweep_compare_results.csv")