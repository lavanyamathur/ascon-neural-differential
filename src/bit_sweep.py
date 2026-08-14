import subprocess, re, csv

WORDS = [0, 4]
BITS = [0, 1, 2, 3, 5, 8, 13, 21, 34, 50, 63]

results = []
for w in WORDS:
    for b in BITS:
        proc = subprocess.run(
            ["python", "saliency_analysis.py", "--model", "gohr_cnn",
             "--delta_word", str(w), "--delta_bit", str(b)],
            capture_output=True, text=True
        )
        out = proc.stdout + proc.stderr
        m = re.search(r"using (\d+) correctly-classified", out)
        count = int(m.group(1)) if m else 0
        print(f"word={w} bit={b} -> correct={count}")
        results.append((w, b, count))

with open("bit_sweep_results.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["word", "bit", "correct_count"])
    writer.writerows(results)

print("Saved to bit_sweep_results.csv")