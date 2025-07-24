from pyannote.audio import Pipeline
import sys, os, torch

if 'HF_TOKEN' not in os.environ: 
    raise ValueError("Переменная окружения HF_TOKEN не найдена! Добавьте ее в .env файл.")

pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-3.1",
     use_auth_token=os.environ['HF_TOKEN'])

pipeline.to(torch.device("cuda"))

diarization = pipeline(sys.argv[1])

for turn, _, speaker in diarization.itertracks(yield_label=True):
    print(f"{speaker} {turn.start:.1f} {turn.end:.1f}")
