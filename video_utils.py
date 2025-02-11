import pandas as pd
import yt_dlp
from yt_dlp.utils import download_range_func

MAX_RETRIES = 3

def download_clip(clip: pd.Series):
    """
    Download video from url and save it to output_path
    """
    yt_opts = {
        'verbose': True,
        'download_ranges': download_range_func(None, [(clip.start_time, clip.end_time)]),
        'force_keyframes_at_cuts': True,
        'outtmpl': f"clips/{clip.clean_text}_{str(clip.url).split('=')[-1]}.mkv",
    }
    try:
        with yt_dlp.YoutubeDL(yt_opts) as ydl:
            ydl.download([clip.url])
            return True
    except Exception as e:
        print(f"Error downloading video: {e}")
        return False
