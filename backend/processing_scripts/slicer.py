import os, sys, subprocess
from tqdm import tqdm

TRESHOLD_TIME=1.1
OUTPUT_DIR = "./"

timings=[]
with open('timings.list','r') as f:
    for line in f:
        timings.append(line.split())

result = []
for entry in timings:
    speaker, start, end = entry
    if (float(end)-float(start))>=TRESHOLD_TIME:
        result.append(entry)
timings=result

result=[]
for entry in timings:
    speaker, start, end = entry
    if result and result[-1][0] == speaker:
        result[-1][2] = end
    else:
        result.append([speaker, start, end])
timings=result

i=1
for r in tqdm(timings):
    speaker,start,end = r
    output_file = os.path.join(OUTPUT_DIR, f"clip-{i:09}-{speaker}.wav")
    ffmpeg_command = [
        "ffmpeg",
        "-i", sys.argv[1],
        "-ss", start,
        "-to", end,
        "-c", "copy",
        output_file
    ]
    try:
        subprocess.run(ffmpeg_command, check=True,stderr=subprocess.DEVNULL,stdout=subprocess.DEVNULL)
    except subprocess.CalledProcessError as e:
        print(f"Error processing clip {i}: {e}")
    i+=1
