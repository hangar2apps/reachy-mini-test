import threading
import time
import numpy as np
import requests
from datetime import datetime
import tempfile
import os

from reachy_mini import ReachyMini, ReachyMiniApp
from reachy_mini.utils import create_head_pose

# TTS setup - using gTTS to generate audio, then play through Reachy's speaker
try:
    from gtts import gTTS
    import soundfile as sf
    import scipy.signal
    TTS_AVAILABLE = True
except ImportError:
    TTS_AVAILABLE = False
    print("gTTS/soundfile not installed - TTS disabled. Run: pip install gtts soundfile scipy")


# El Reno, Oklahoma coordinates
LATITUDE = 35.5326
LONGITUDE = -97.9550


def get_weather() -> dict:
    """Fetch current weather from Open-Meteo API."""
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": LATITUDE,
        "longitude": LONGITUDE,
        "current": ["temperature_2m", "weather_code", "wind_speed_10m"],
        "temperature_unit": "fahrenheit",
        "wind_speed_unit": "mph",
        "timezone": "America/Chicago",
    }
    
    try:
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        current = data.get("current", {})
        return {
            "temp": current.get("temperature_2m"),
            "wind": current.get("wind_speed_10m"),
            "code": current.get("weather_code"),
        }
    except Exception as e:
        print(f"Weather fetch failed: {e}")
        return {}


def weather_code_to_text(code: int) -> str:
    """Convert WMO weather code to human-readable text."""
    codes = {
        0: "clear skies",
        1: "mainly clear",
        2: "partly cloudy",
        3: "overcast",
        45: "foggy",
        48: "foggy with frost",
        51: "light drizzle",
        53: "moderate drizzle",
        55: "dense drizzle",
        61: "slight rain",
        63: "moderate rain",
        65: "heavy rain",
        71: "slight snow",
        73: "moderate snow",
        75: "heavy snow",
        77: "snow grains",
        80: "slight rain showers",
        81: "moderate rain showers",
        82: "violent rain showers",
        85: "slight snow showers",
        86: "heavy snow showers",
        95: "thunderstorm",
        96: "thunderstorm with slight hail",
        99: "thunderstorm with heavy hail",
    }
    return codes.get(code, "unknown conditions")


def build_greeting(weather: dict) -> str:
    """Build the morning greeting text."""
    now = datetime.now()
    day_name = now.strftime("%A")
    
    greeting = f"Good morning! Happy {day_name}!"
    
    if weather:
        temp = weather.get("temp")
        wind = weather.get("wind")
        code = weather.get("code")
        
        if temp is not None:
            greeting += f" It's currently {int(temp)} degrees"
            if code is not None:
                greeting += f" with {weather_code_to_text(code)}"
            greeting += " in El Reno."
        
        if wind is not None and wind > 15:
            greeting += f" It's a bit windy at {int(wind)} miles per hour."
    
    greeting += " Have a great day!"
    return greeting


class MorningRoutine(ReachyMiniApp):
    """Morning wake-up routine with weather report."""
    
    custom_app_url: str | None = None

    def run(self, reachy_mini: ReachyMini, stop_event: threading.Event):
        """Run the morning routine."""
        
        # --- Phase 1: Wake up from sleep position ---
        print("Starting wake-up sequence...")
        
        # Start in sleep pose (head down, antennas flat)
        reachy_mini.goto_target(
            head=create_head_pose(z=-20, mm=True),
            antennas=[0, 0],
            duration=0.5,
            method="minjerk"
        )
        time.sleep(0.6)
        
        if stop_event.is_set():
            return
        
        # Slowly raise head (waking up)
        reachy_mini.goto_target(
            head=create_head_pose(z=0, mm=True),
            duration=1.5,
            method="minjerk"
        )
        time.sleep(1.6)
        
        if stop_event.is_set():
            return
        
        # --- Phase 2: Stretch / look around ---
        print("Stretching...")
        
        # Look left
        reachy_mini.goto_target(
            head=create_head_pose(yaw=30, degrees=True),
            duration=0.8,
            method="minjerk"
        )
        time.sleep(0.9)
        
        if stop_event.is_set():
            return
        
        # Look right
        reachy_mini.goto_target(
            head=create_head_pose(yaw=-30, degrees=True),
            duration=1.0,
            method="minjerk"
        )
        time.sleep(1.1)
        
        if stop_event.is_set():
            return
        
        # Back to center + look up (stretch)
        reachy_mini.goto_target(
            head=create_head_pose(z=15, mm=True),
            antennas=[0.5, 0.5],  # antennas out
            duration=1.0,
            method="minjerk"
        )
        time.sleep(1.1)
        
        if stop_event.is_set():
            return
        
        # --- Phase 3: Happy antenna wiggle ---
        print("Wiggling antennas...")
        
        for _ in range(3):
            if stop_event.is_set():
                return
            reachy_mini.goto_target(antennas=[0.7, -0.3], duration=0.2)
            time.sleep(0.25)
            reachy_mini.goto_target(antennas=[-0.3, 0.7], duration=0.2)
            time.sleep(0.25)
        
        # Settle antennas
        reachy_mini.goto_target(antennas=[0.3, 0.3], duration=0.3)
        time.sleep(0.4)
        
        if stop_event.is_set():
            return
        
        # --- Phase 4: Return to neutral + fetch weather ---
        reachy_mini.goto_target(
            head=create_head_pose(),  # neutral
            antennas=[0, 0],
            duration=0.8,
            method="minjerk"
        )
        
        print("Fetching weather...")
        weather = get_weather()
        greeting = build_greeting(weather)
        print(f"Greeting: {greeting}")
        
        time.sleep(0.9)
        
        if stop_event.is_set():
            return
        
        # --- Phase 5: Speak the greeting ---
        if TTS_AVAILABLE and reachy_mini.media is not None:
            try:
                # Generate TTS audio file
                with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
                    tts = gTTS(text=greeting, lang='en')
                    tts.save(tmp.name)
                    tmp_path = tmp.name
                
                # Convert mp3 to wav
                from pydub import AudioSegment
                audio = AudioSegment.from_mp3(tmp_path)
                wav_path = tmp_path.replace(".mp3", ".wav")
                audio.export(wav_path, format="wav")
                data, samplerate_in = sf.read(wav_path, dtype="float32")
                os.unlink(wav_path)
                os.unlink(tmp_path)
                
                # Resample to target rate
                target_rate = reachy_mini.media.get_output_audio_samplerate()
                if samplerate_in != target_rate:
                    data = scipy.signal.resample(
                        data,
                        int(len(data) * (target_rate / samplerate_in))
                    )
                
                # Ensure correct shape (samples, channels)
                if data.ndim == 1:
                    data = data.reshape(-1, 1)
                
                # Start playback, push audio, wait, stop
                reachy_mini.media.start_playing()
                reachy_mini.media.push_audio_sample(data.astype(np.float32))
                
                duration = len(data) / target_rate
                time.sleep(duration + 1)
                
                reachy_mini.media.stop_playing()
                
            except Exception as e:
                print(f"TTS error: {e}")
                print(f"Would say: {greeting}")
        else:
            print(f"[TTS not available] Would say: {greeting}")
        
        # Small nod while speaking
        reachy_mini.goto_target(
            head=create_head_pose(z=5, mm=True),
            duration=0.5
        )
        time.sleep(3)  # wait for speech
        
        reachy_mini.goto_target(
            head=create_head_pose(),
            duration=0.5
        )
        
        print("Morning routine complete!")


# For direct execution (testing on Pi or Mac)
if __name__ == "__main__":
    import sys
    
    # Check if running remotely (from Mac)
    localhost_only = "--remote" not in sys.argv
    
    print(f"Running {'locally on Pi' if localhost_only else 'remotely from Mac'}...")
    
    # If running locally on Pi, ensure daemon is started
    if localhost_only:
        try:
            import requests as req
            print("Starting Rodrigo...")
            req.post("http://localhost:8000/api/daemon/start?wake_up=true", timeout=5)
            # Wait for daemon to be ready
            import time
            for _ in range(10):
                try:
                    status = req.get("http://localhost:8000/api/daemon/status", timeout=2).json()
                    if status.get("state") == "running":
                        print("Rodrigo is ready!")
                        break
                except:
                    pass
                time.sleep(1)
        except Exception as e:
            print(f"Warning: Could not start daemon via API: {e}")
    
    # Use "gstreamer" backend on Pi (wireless version), "no_media" for remote testing
    media_backend = "gstreamer" if localhost_only else "no_media"
    
    with ReachyMini(localhost_only=localhost_only, media_backend=media_backend) as mini:
        stop = threading.Event()
        app = MorningRoutine()
        app.run(mini, stop)