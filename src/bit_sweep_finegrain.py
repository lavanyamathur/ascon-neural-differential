# bit_sweep_finegrain.py
import subprocess, re, csv

MODELS = ["gohr_cnn", "mlp"]
WORDS = list(range(5))
BITS = list(range(15, 30)) + list(range(38, 48))  # dense around 22, and around 42/44

results = []
total = len(MODELS) * len(WORDS) * len(BITS)
done = 0
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
            done += 1
            print(f"[{done}/{total}] model={model} word={w} bit={b} -> correct={count}")
            results.append((model, w, b, count))

with open("bit_sweep_finegrain_results.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["model", "word", "bit", "correct_count"])
    writer.writerows(results)

print("Saved to bit_sweep_finegrain_results.csv")