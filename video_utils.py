import pandas as pd
import yt_dlp
from yt_dlp.utils import download_range_func
import os 

MAX_RETRIES = 3

def download_clip(clip: pd.Series, verbose: bool = False):
    """
    Download video from url and save it to output_path
    """
    name = f"{clip.clean_text}_{str(clip.url).split('=')[-1]}"
    path = f"clips/{name}.mkv"
    yt_opts = {
        'verbose': verbose,
        'download_ranges': download_range_func(None, [(clip.start_time, clip.end_time)]),
        'force_keyframes_at_cuts': True,
        'outtmpl': path,
    }
    if os.path.exists(path):
        print(f"Video already exists: {path}")
        return name
    try:
        with yt_dlp.YoutubeDL(yt_opts) as ydl:
            ydl.download([clip.url])
            return name
    except Exception as e:
        print(f"Error downloading video: {e}")
        return None

def delete_clip(clip_name: str):
    """
    Delete video from clips directory
    """
    try:
        os.remove(f"clips/{clip_name}.mkv")
    except Exception as e:
        print(f"Error deleting video: {e}")