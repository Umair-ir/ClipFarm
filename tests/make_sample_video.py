"""Generate a synthetic ~3 minute "stream" for testing (macOS only: uses `say`).

Two TTS voices hold a conversation with dull stretches, one funny story and one
heated argument, over an FFmpeg test pattern. Real footage isn't needed to
exercise the whole pipeline.

    python tests/make_sample_video.py [OUT_PATH]
"""
from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT = ROOT / "data" / "samples" / "sample_stream.mp4"

A, B = "Samantha", "Daniel"
SCRIPT = [
    (A, "Okay, we are live. Let me just fix the audio levels real quick. Give me one second."),
    (A, "Alright, can everyone hear me? Chat, let me know if the mic is too quiet."),
    (B, "You sound fine. Your camera is a little dark though."),
    (A, "Yeah, I know. I'll fix the lighting after the stream. Anyway, welcome in everybody."),
    (A, "Thank you for the follows. We're going to play some ranked later today."),
    (B, "Before we start, tell them what happened at the airport yesterday."),
    (A, "Oh no. Okay, so I'm at the airport, right, and I'm running super late for my flight."),
    (A, "I sprint to the gate, I'm sweating, I hand the lady my boarding pass, completely out of breath."),
    (A, "And she looks at it, looks at me, and goes: sir, this flight left yesterday."),
    (B, "Yesterday! You showed up a whole day late and still ran!"),
    (A, "I ran so hard! I was so proud of that sprint! I did a little victory jump at the gate!"),
    (B, "That is the funniest thing I have ever heard. I can't breathe."),
    (A, "The guy behind me started clapping. Slow clapping. The entire line joined in."),
    (A, "Anyway. Let me check the settings for the game. One moment."),
    (A, "Okay, I need to update the graphics drivers. This is going to take a minute."),
    (B, "While that downloads, I need to say something. Pineapple on pizza is a crime."),
    (A, "Absolutely not. Pineapple on pizza is the best topping ever invented. I will die on this hill."),
    (B, "Fruit does not belong on pizza. Tomatoes are a fruit and that's where it stops."),
    (A, "That is the worst argument I have ever heard. You literally put ketchup on pasta!"),
    (B, "That was one time! And you promised you would never bring that up on stream!"),
    (A, "Chat, he put ketchup on pasta. Chat is going crazy right now. Ninety percent agree with me."),
    (B, "Chat is wrong. I'm leaving. I'm actually leaving the call."),
    (A, "Okay, the download is at fifty percent. Let me read some messages while we wait."),
    (A, "Someone asked what keyboard I use. It's just a normal mechanical one, nothing special."),
    (A, "And someone asked about the schedule. I stream Monday, Wednesday and Friday."),
    (A, "Okay, the drivers are installed. Let's queue up for ranked. Wish me luck."),
]


def main(out: Path = DEFAULT_OUT) -> Path:
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        tmpd = Path(tmp)
        parts = []
        for i, (voice, line) in enumerate(SCRIPT):
            aiff = tmpd / f"{i:03d}.aiff"
            subprocess.run(["say", "-v", voice, "-r", "185", "-o", str(aiff), line], check=True)
            parts.append(aiff)
        # Concat with a short gap between lines.
        gap = tmpd / "gap.aiff"
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i",
                        "anullsrc=r=22050:cl=mono", "-t", "0.6", str(gap)], check=True)
        listing = tmpd / "list.txt"
        listing.write_text("".join(f"file '{p}'\nfile '{gap}'\n" for p in parts))
        audio = tmpd / "audio.wav"
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0",
                        "-i", str(listing), "-ar", "48000", str(audio)], check=True)
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error",
                        "-f", "lavfi", "-i", "testsrc2=size=1920x1080:rate=30",
                        "-i", str(audio), "-shortest",
                        "-c:v", "libx264", "-preset", "veryfast", "-crf", "28", "-pix_fmt", "yuv420p",
                        "-c:a", "aac", "-b:a", "128k", str(out)], check=True)
    print(out)
    return out


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_OUT)
