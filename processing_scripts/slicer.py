import os
import sys
import subprocess
from tqdm import tqdm


if len(sys.argv) < 3:
    print("Ошибка: Скрипт требует два аргумента.")
    print("Использование: python slicer.py <путь_к_wav> <путь_к_timings.list>")
    sys.exit(1)

INPUT_WAV = sys.argv[1]
TIMINGS_FILE = sys.argv[2]

if not os.path.exists(INPUT_WAV):
    print(f"Ошибка: Входной WAV-файл не найден по пути: {INPUT_WAV}")
    sys.exit(1)

if not os.path.exists(TIMINGS_FILE):
    print(f"Ошибка: Файл с таймингами не найден по пути: {TIMINGS_FILE}")
    sys.exit(1)


TRESHOLD_TIME = 1.1
OUTPUT_DIR = "." 

timings = []
with open(TIMINGS_FILE, 'r') as f:
    for line in f:
        timings.append(line.split())

result = []
for entry in timings:
    speaker, start, end = entry
    if (float(end) - float(start)) >= TRESHOLD_TIME:
        result.append(entry)

timings = result


result = []
if timings:
    for entry in timings:
        speaker, start, end = entry
        if result and result[-1][0] == speaker:
            result[-1][2] = end
        else:
            result.append([speaker, start, end])

timings = result

i = 1
print(f"Начинаю нарезку {INPUT_WAV} на {len(timings)} клипов...")
for r in tqdm(timings, desc="Slicing audio"):
    speaker, start, end = r
    output_file = os.path.join(OUTPUT_DIR, f"clip-{i:09}-{speaker}.wav")
    
    ffmpeg_command = [
        "ffmpeg",
        "-i", INPUT_WAV,
        "-ss", start,
        "-to", end,
        "-c", "copy",
        output_file
    ]
    
    try:
        subprocess.run(ffmpeg_command, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        print(f"Ошибка при обработке клипа {i}: {e}")
        print(f"FFMPEG STDERR: {e.stderr}")

    i += 1

print("Нарезка завершена.")
