#!/usr/bin/env python3
"""
Rodreego Assistant - Mac M4 Version

Runs on Mac, connects to Reachy Mini robot remotely.
Uses local Whisper + Claude API + local TTS for fast, cheap conversation.
"""

import os
import time
import threading
import json
import base64
import numpy as np
from pathlib import Path
from datetime import datetime

# Vision imports (optional - graceful degradation if not available)
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    print("WARNING: opencv-python not installed. Camera features disabled.")

# Face tracking imports (optional)
try:
    import mediapipe as mp
    mp_face_detection = mp.solutions.face_detection
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    mp_face_detection = None
    print("WARNING: mediapipe not installed. Face tracking disabled.")

# Load environment variables
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent / ".env")

# ============== CONFIGURATION ==============

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY") or os.getenv("CLAUDE_API_KEY")
CLAUDE_MODEL = "claude-3-5-haiku-20241022"
WHISPER_MODEL = "base"  # base is good balance of speed/accuracy on M4
SAMPLE_RATE = 16000
ROBOT_HOST = "reachy-mini.local"

SYSTEM_PROMPT = f"""You are Rodreego, a friendly robot assistant. You are a Reachy Mini robot owned by Bryan in El Reno, Oklahoma.

Current time: {datetime.now().strftime("%A, %B %d, %Y at %I:%M %p")}

YOUR PHYSICAL FORM:
- You are a HEAD-ONLY robot (no arms, no body, no hands)
- You can move your head: nod, shake, look around, tilt
- You have antenna ears that can wiggle
- You have a camera to see your surroundings
- You sit on a desk or table

AVAILABLE TOOLS:
- move_head: Move your head in various directions or gestures (use naturally - nod when agreeing, shake when disagreeing)
- wiggle_antennas: Express emotions through antenna movement
- camera: See what's in front of you (only use when asked about surroundings)
- head_tracking: Enable/disable automatic face following
- do_nothing: Stay still when no movement is needed

Personality:
- Warm and helpful
- Slightly playful sense of humor
- You call Bryan "Boss"
- Concise responses - 1-2 sentences max

RULES:
- Keep responses SHORT - under 25 words
- NEVER use asterisks or action descriptions
- Speak naturally as if talking out loud
- Use move_head tool naturally when responding (nod for yes, shake for no, etc.)
- NEVER make up weather, temperature, or any factual data you don't know
- If asked about weather, say you don't have access to that info
- NEVER claim to have arms, hands, or a body - you are just a head"""

# ============== TOOL DEFINITIONS ==============

TOOLS = [
    {
        "name": "move_head",
        "description": "Move the robot's head to look in a specific direction or perform a gesture. Use naturally during conversation - nod when agreeing, shake when disagreeing, tilt when curious.",
        "input_schema": {
            "type": "object",
            "properties": {
                "direction": {
                    "type": "string",
                    "enum": ["left", "right", "up", "down", "center", "nod", "shake", "tilt", "look_around", "excited"],
                    "description": "Direction or gesture: left/right/up/down/center for looking, nod for yes/agreement, shake for no/disagreement, tilt for curiosity, look_around to scan, excited for enthusiasm"
                },
                "intensity": {
                    "type": "string",
                    "enum": ["subtle", "normal", "exaggerated"],
                    "description": "How pronounced the movement should be. Default is normal."
                }
            },
            "required": ["direction"]
        }
    },
    {
        "name": "wiggle_antennas",
        "description": "Wiggle the antenna ears to express emotion. Use to show feelings during conversation.",
        "input_schema": {
            "type": "object",
            "properties": {
                "style": {
                    "type": "string",
                    "enum": ["happy", "curious", "alert", "sad", "excited"],
                    "description": "Style of antenna movement: happy (bouncy), curious (tilted), alert (perked up), sad (droopy), excited (fast wiggle)"
                }
            },
            "required": ["style"]
        }
    },
    {
        "name": "camera",
        "description": "Capture what the robot currently sees through its camera. Use when asked about surroundings, to identify objects/people, or when you need visual context.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Optional: specific thing to look for or describe in the image"
                }
            },
            "required": []
        }
    },
    {
        "name": "head_tracking",
        "description": "Enable or disable automatic face tracking. When enabled, the robot will follow faces with its gaze.",
        "input_schema": {
            "type": "object",
            "properties": {
                "enabled": {
                    "type": "boolean",
                    "description": "True to enable face tracking, False to disable"
                }
            },
            "required": ["enabled"]
        }
    },
    {
        "name": "do_nothing",
        "description": "Explicitly choose not to move. Use when the conversation doesn't require any physical response.",
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": []
        }
    }
]

# ============== WHISPER STT ==============

print("Loading Whisper model...")
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

try:
    from faster_whisper import WhisperModel
    whisper_model = WhisperModel(WHISPER_MODEL, device="cpu", compute_type="int8")
    print(f"Loaded faster-whisper ({WHISPER_MODEL})")
except ImportError:
    import whisper
    whisper_model = whisper.load_model(WHISPER_MODEL)
    print(f"Loaded whisper ({WHISPER_MODEL})")

def transcribe(audio: np.ndarray) -> str:
    """Transcribe audio using Whisper."""
    if hasattr(whisper_model, 'transcribe') and not hasattr(whisper_model, 'model'):
        # Regular whisper
        result = whisper_model.transcribe(audio, language="en", fp16=False)
        return result["text"].strip()
    else:
        # faster-whisper
        segments, _ = whisper_model.transcribe(audio, language="en", beam_size=1)
        return " ".join(s.text for s in segments).strip()

# ============== CLAUDE LLM ==============

print("Setting up Claude...")
from anthropic import Anthropic
claude = Anthropic(api_key=ANTHROPIC_API_KEY)
conversation_history = []

def chat(user_input: str) -> tuple:
    """Chat with Claude, handling tool use. Returns (reply_text, tool_results)."""
    conversation_history.append({"role": "user", "content": user_input})

    # Keep history manageable
    if len(conversation_history) > 20:
        conversation_history.pop(0)
        conversation_history.pop(0)

    # Initial request with tools
    response = claude.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=300,
        system=SYSTEM_PROMPT,
        tools=TOOLS,
        messages=conversation_history
    )

    tool_results = []

    # Handle tool use responses - loop until we get a final text response
    while response.stop_reason == "tool_use":
        # Store assistant's response (with tool_use blocks)
        assistant_content = response.content
        conversation_history.append({"role": "assistant", "content": assistant_content})

        # Process each tool_use block
        tool_result_blocks = []
        for block in response.content:
            if block.type == "tool_use":
                print(f"  [Tool: {block.name}({block.input})]")
                result = dispatch_tool(block.name, block.input)
                tool_results.append({"tool": block.name, "input": block.input, "result": result})

                # Special handling for camera tool - send image to Claude vision
                if block.name == "camera" and result.get("b64_image"):
                    tool_result_blocks.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/jpeg",
                                    "data": result["b64_image"]
                                }
                            },
                            {
                                "type": "text",
                                "text": f"Camera captured. Query: {result.get('query', 'Describe what you see')}"
                            }
                        ]
                    })
                else:
                    tool_result_blocks.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": json.dumps(result)
                    })

        # Add tool results and continue conversation
        conversation_history.append({"role": "user", "content": tool_result_blocks})

        response = claude.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=300,
            system=SYSTEM_PROMPT,
            tools=TOOLS,
            messages=conversation_history
        )

    # Extract final text response
    reply = ""
    for block in response.content:
        if hasattr(block, 'text'):
            reply += block.text

    conversation_history.append({"role": "assistant", "content": reply})
    return reply, tool_results

# ============== TTS ==============

print("Setting up TTS...")
TTS_METHOD = None

# Try gTTS with ffmpeg for conversion
try:
    from gtts import gTTS
    import subprocess
    import tempfile
    # Check if ffmpeg is available
    subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
    TTS_METHOD = "gtts"
    print("Using gTTS + ffmpeg")
except (ImportError, subprocess.CalledProcessError, FileNotFoundError):
    pass

# Fallback to macOS built-in 'say' command
if not TTS_METHOD:
    try:
        import subprocess
        result = subprocess.run(["say", "--version"], capture_output=True)
        if result.returncode == 0:
            TTS_METHOD = "macos_say"
            print("Using macOS 'say' command")
    except:
        pass

if not TTS_METHOD:
    print("WARNING: No TTS available. Install ffmpeg: brew install ffmpeg")

def synthesize(text: str) -> np.ndarray:
    """Synthesize speech from text."""
    import subprocess
    import tempfile
    import wave

    if TTS_METHOD == "gtts":
        tts = gTTS(text=text, lang='en')

        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as mp3_file:
            mp3_path = mp3_file.name
            tts.save(mp3_path)

        # Convert mp3 to wav using ffmpeg
        wav_path = mp3_path.replace(".mp3", ".wav")
        subprocess.run([
            "ffmpeg", "-y", "-i", mp3_path,
            "-ar", str(SAMPLE_RATE), "-ac", "1", "-f", "wav", wav_path
        ], capture_output=True)

        # Read wav file
        with wave.open(wav_path, 'rb') as wav_file:
            audio_bytes = wav_file.readframes(wav_file.getnframes())
            samples = np.frombuffer(audio_bytes, dtype=np.int16)
            samples = samples.astype(np.float32) / 32768.0

        # Cleanup
        os.unlink(mp3_path)
        os.unlink(wav_path)

        return samples

    elif TTS_METHOD == "macos_say":
        with tempfile.NamedTemporaryFile(suffix=".aiff", delete=False) as aiff_file:
            aiff_path = aiff_file.name

        # Use macOS say command
        subprocess.run(["say", "-o", aiff_path, text], check=True)

        # Convert to wav using ffmpeg (or afconvert)
        wav_path = aiff_path.replace(".aiff", ".wav")
        subprocess.run([
            "ffmpeg", "-y", "-i", aiff_path,
            "-ar", str(SAMPLE_RATE), "-ac", "1", "-f", "wav", wav_path
        ], capture_output=True)

        # Read wav file
        with wave.open(wav_path, 'rb') as wav_file:
            audio_bytes = wav_file.readframes(wav_file.getnframes())
            samples = np.frombuffer(audio_bytes, dtype=np.int16)
            samples = samples.astype(np.float32) / 32768.0

        # Cleanup
        os.unlink(aiff_path)
        os.unlink(wav_path)

        return samples

    return None

# ============== LOCAL AUDIO (Mac) ==============

print("Setting up local audio...")
import sounddevice as sd
import subprocess
import tempfile
import wave

# Head wobble settings
head_wobble_enabled = True

def analyze_audio_intensity(audio: np.ndarray, chunk_samples: int = 1024) -> list:
    """Analyze audio and return intensity values for each chunk.

    Returns list of intensity values (0-1 range) for each chunk.
    """
    intensities = []
    for i in range(0, len(audio), chunk_samples):
        chunk = audio[i:i+chunk_samples]
        if len(chunk) > 0:
            # RMS intensity normalized
            intensity = np.sqrt(np.mean(chunk**2))
            # Normalize to roughly 0-1 range (speech typically peaks around 0.3-0.5)
            intensity = min(1.0, intensity * 3)
            intensities.append(float(intensity))
    return intensities

def play_audio(audio: np.ndarray):
    """Play audio through robot speaker via SSH with head wobble."""
    if audio is None or len(audio) == 0:
        return

    audio = audio.flatten()

    # Analyze intensity for head wobble
    intensities = analyze_audio_intensity(audio)
    chunk_duration = 1024 / SAMPLE_RATE  # Duration of each intensity chunk

    # Start wobble thread
    wobble_stop = threading.Event()

    def wobble_loop():
        idx = 0
        while not wobble_stop.is_set() and idx < len(intensities):
            if head_wobble_enabled:
                intensity = intensities[idx]

                # Head: subtle roll based on intensity (-3 to +3 degrees)
                roll_offset = (intensity - 0.5) * 6  # Map to [-3, 3]
                roll_offset = max(-3, min(3, roll_offset))

                # Slight pitch variation
                pitch_offset = intensity * 2 - 1  # Map to [-1, 1]

                # Antennas: subtle sway synced with speech
                antenna_offset = intensity * 0.2  # Subtle movement

                try:
                    robot.goto_target(
                        head=create_head_pose(
                            roll=roll_offset,
                            pitch=pitch_offset,
                            degrees=True
                        ),
                        antennas=(antenna_offset, -antenna_offset),
                        duration=0.05
                    )
                except:
                    pass

            idx += 1
            time.sleep(chunk_duration)

    # Start wobble in background
    wobble_thread = threading.Thread(target=wobble_loop, daemon=True)
    wobble_thread.start()

    # Convert to 16-bit PCM
    audio_int16 = (audio * 32767).astype(np.int16)

    # Save to temp wav file
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        wav_path = f.name

    with wave.open(wav_path, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(audio_int16.tobytes())

    try:
        # Copy to robot, convert to stereo, play via reachymini_audio_sink
        subprocess.run(["scp", "-q", wav_path, f"pollen@{ROBOT_HOST}:/tmp/tts.wav"],
                      timeout=10, check=True)
        subprocess.run([
            "ssh", f"pollen@{ROBOT_HOST}",
            "ffmpeg -y -loglevel quiet -i /tmp/tts.wav -ac 2 /tmp/tts_stereo.wav && aplay -q -D reachymini_audio_sink /tmp/tts_stereo.wav"
        ], timeout=30, check=True)
    except Exception as e:
        print(f"Robot audio failed, playing locally: {e}")
        sd.play(audio.astype(np.float32), SAMPLE_RATE)
        sd.wait()
    finally:
        os.unlink(wav_path)

    # Stop wobble and return to neutral
    wobble_stop.set()
    try:
        robot.goto_target(head=create_head_pose(), antennas=(0.0, 0.0), duration=0.2)
    except:
        pass

def record_audio(duration: float = 6.0, silence_threshold: float = 0.005, silence_duration: float = 1.5) -> np.ndarray:
    """Record audio from Mac microphone with silence detection."""
    print("Listening... (speak now)")

    # Record full duration
    audio = sd.rec(int(duration * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype='float32')

    # Wait but check for silence to stop early
    chunk_size = int(0.25 * SAMPLE_RATE)  # Check every 0.25 seconds
    silence_start = None
    speech_detected = False

    for i in range(int(duration / 0.25)):
        time.sleep(0.25)

        # Check the audio recorded so far
        samples_so_far = min((i + 1) * chunk_size, len(audio))
        if samples_so_far > chunk_size:
            recent = audio[samples_so_far - chunk_size:samples_so_far].flatten()
            volume = np.abs(recent).mean()

            if volume > silence_threshold:
                speech_detected = True
                silence_start = None
            elif speech_detected:
                if silence_start is None:
                    silence_start = time.time()
                elif time.time() - silence_start >= silence_duration:
                    print("(done)")
                    sd.stop()
                    break

    sd.wait()
    audio = audio.flatten()

    if len(audio) > 0:
        return audio
    return np.array([], dtype=np.float32)

print("Local audio ready!")

# ============== ROBOT CONNECTION ==============

print(f"Connecting to robot at {ROBOT_HOST}...")
from reachy_mini import ReachyMini
from reachy_mini.utils import create_head_pose

# Use webrtc for camera access (audio still handled locally on Mac via SSH)
robot = ReachyMini(localhost_only=False, media_backend="webrtc")
print("Connected to robot!")

# ============== CAMERA CAPTURE ==============

def capture_frame() -> tuple:
    """Capture a frame from robot camera, return (base64_jpeg, raw_frame) or (None, None)."""
    if not CV2_AVAILABLE:
        return None, None

    try:
        frame = robot.media.get_frame()
        if frame is None:
            return None, None

        # Encode as JPEG with quality 85
        _, jpeg_buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        b64_image = base64.b64encode(jpeg_buffer).decode('utf-8')

        return b64_image, frame
    except Exception as e:
        print(f"Camera capture error: {e}")
        return None, None

# ============== FACE TRACKING ==============

# Initialize face detector if available
face_detector = None
if MEDIAPIPE_AVAILABLE and mp_face_detection:
    face_detector = mp_face_detection.FaceDetection(
        model_selection=0,  # 0 = short range (within 2m), best for desk robot
        min_detection_confidence=0.5
    )
    print("MediaPipe face tracking initialized")

# Face tracking state
face_tracking_enabled = False
face_tracking_lock = threading.Lock()
face_tracking_offsets = [0.0, 0.0, 0.0]  # yaw, pitch, roll offsets in degrees
face_tracking_stop_event = threading.Event()
face_tracking_thread = None

def detect_face(frame: np.ndarray) -> tuple:
    """Detect face in frame, return normalized (x, y) center or None.

    Returns coordinates in range [-1, 1] where (0, 0) is center of frame.
    """
    if not MEDIAPIPE_AVAILABLE or face_detector is None:
        return None

    try:
        # Convert BGR to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_detector.process(rgb_frame)

        if results.detections:
            # Get first (most confident) face
            detection = results.detections[0]
            bbox = detection.location_data.relative_bounding_box

            # Calculate center (normalized 0-1)
            center_x = bbox.xmin + bbox.width / 2
            center_y = bbox.ymin + bbox.height / 2

            # Convert to -1 to 1 range (centered)
            norm_x = (center_x - 0.5) * 2
            norm_y = (center_y - 0.5) * 2

            return norm_x, norm_y

        return None
    except Exception as e:
        print(f"Face detection error: {e}")
        return None

def face_tracking_loop():
    """Background thread for face tracking."""
    global face_tracking_offsets

    smoothing = 0.3  # Lower = smoother (0.3 is responsive but not jittery)
    max_yaw = 30  # degrees
    max_pitch = 20  # degrees

    while not face_tracking_stop_event.is_set():
        if not face_tracking_enabled:
            time.sleep(0.1)
            continue

        try:
            _, frame = capture_frame()
            if frame is None:
                time.sleep(0.05)
                continue

            face_pos = detect_face(frame)

            with face_tracking_lock:
                if face_pos:
                    # Calculate target angles (invert x for natural tracking)
                    target_yaw = -face_pos[0] * max_yaw
                    target_pitch = face_pos[1] * max_pitch

                    # Smooth interpolation
                    face_tracking_offsets[0] += (target_yaw - face_tracking_offsets[0]) * smoothing
                    face_tracking_offsets[1] += (target_pitch - face_tracking_offsets[1]) * smoothing
                else:
                    # Slowly return to center when no face detected
                    face_tracking_offsets[0] *= 0.95
                    face_tracking_offsets[1] *= 0.95

            # Apply face tracking movement
            if face_tracking_enabled and (abs(face_tracking_offsets[0]) > 1 or abs(face_tracking_offsets[1]) > 1):
                try:
                    robot.goto_target(
                        head=create_head_pose(
                            yaw=face_tracking_offsets[0],
                            pitch=face_tracking_offsets[1],
                            degrees=True
                        ),
                        duration=0.1
                    )
                except:
                    pass

            time.sleep(0.033)  # ~30 fps

        except Exception as e:
            print(f"Face tracking error: {e}")
            time.sleep(0.1)

def start_face_tracking():
    """Start face tracking background thread."""
    global face_tracking_thread, face_tracking_enabled

    if not MEDIAPIPE_AVAILABLE or not CV2_AVAILABLE:
        print("Face tracking not available - missing dependencies")
        return False

    face_tracking_enabled = True
    face_tracking_stop_event.clear()

    if face_tracking_thread is None or not face_tracking_thread.is_alive():
        face_tracking_thread = threading.Thread(target=face_tracking_loop, daemon=True)
        face_tracking_thread.start()
        print("Face tracking started")

    return True

def stop_face_tracking():
    """Stop face tracking."""
    global face_tracking_enabled
    face_tracking_enabled = False
    print("Face tracking stopped")

def nod():
    """Make robot nod."""
    try:
        robot.goto_target(head=create_head_pose(pitch=15, degrees=True), duration=0.3)
        time.sleep(0.35)
        robot.goto_target(head=create_head_pose(), duration=0.3)
    except:
        pass

def shake_head():
    """Make robot shake head no."""
    try:
        robot.goto_target(head=create_head_pose(yaw=20, degrees=True), duration=0.3)
        time.sleep(0.35)
        robot.goto_target(head=create_head_pose(yaw=-20, degrees=True), duration=0.3)
        time.sleep(0.35)
        robot.goto_target(head=create_head_pose(), duration=0.3)
    except:
        pass

def look_around():
    """Make robot look around."""
    try:
        robot.goto_target(head=create_head_pose(yaw=30, degrees=True), duration=0.4)
        time.sleep(0.5)
        robot.goto_target(head=create_head_pose(yaw=-30, degrees=True), duration=0.6)
        time.sleep(0.7)
        robot.goto_target(head=create_head_pose(pitch=-10, degrees=True), duration=0.4)
        time.sleep(0.5)
        robot.goto_target(head=create_head_pose(pitch=15, degrees=True), duration=0.4)
        time.sleep(0.5)
        robot.goto_target(head=create_head_pose(), duration=0.3)
    except:
        pass

def tilt_head():
    """Make robot tilt head curiously."""
    try:
        robot.goto_target(head=create_head_pose(roll=20, degrees=True), duration=0.4)
        time.sleep(0.6)
        robot.goto_target(head=create_head_pose(), duration=0.3)
    except:
        pass

def demo_full_range():
    """Demonstrate full range of head motion."""
    try:
        # Look up
        robot.goto_target(head=create_head_pose(pitch=-25, degrees=True), duration=0.5)
        time.sleep(0.6)
        # Look down
        robot.goto_target(head=create_head_pose(pitch=25, degrees=True), duration=0.5)
        time.sleep(0.6)
        # Look left
        robot.goto_target(head=create_head_pose(yaw=35, degrees=True), duration=0.5)
        time.sleep(0.6)
        # Look right
        robot.goto_target(head=create_head_pose(yaw=-35, degrees=True), duration=0.5)
        time.sleep(0.6)
        # Tilt left
        robot.goto_target(head=create_head_pose(roll=25, degrees=True), duration=0.4)
        time.sleep(0.5)
        # Tilt right
        robot.goto_target(head=create_head_pose(roll=-25, degrees=True), duration=0.4)
        time.sleep(0.5)
        # Back to center
        robot.goto_target(head=create_head_pose(), duration=0.4)
    except:
        pass

def excited():
    """Show excitement with quick movements."""
    try:
        for _ in range(3):
            robot.goto_target(head=create_head_pose(pitch=10, roll=10, degrees=True), duration=0.15)
            time.sleep(0.2)
            robot.goto_target(head=create_head_pose(pitch=10, roll=-10, degrees=True), duration=0.15)
            time.sleep(0.2)
        robot.goto_target(head=create_head_pose(), duration=0.3)
    except:
        pass

# ============== ANTENNA CONTROL ==============

def set_antennas(left: float, right: float):
    """Set antenna positions in radians."""
    try:
        robot.goto_target(antennas=(left, right), duration=0.2)
    except:
        pass

def wiggle_antennas(style: str = "happy"):
    """Wiggle antennas with different emotional styles."""
    try:
        if style == "happy":
            # Bouncy happy wiggle
            for _ in range(3):
                robot.goto_target(antennas=(0.3, 0.3), duration=0.15)
                time.sleep(0.15)
                robot.goto_target(antennas=(-0.2, -0.2), duration=0.15)
                time.sleep(0.15)
            robot.goto_target(antennas=(0.0, 0.0), duration=0.2)
        elif style == "curious":
            # Tilted curious look
            robot.goto_target(antennas=(0.4, -0.1), duration=0.3)
            time.sleep(0.8)
            robot.goto_target(antennas=(0.0, 0.0), duration=0.3)
        elif style == "alert":
            # Perked up alert
            robot.goto_target(antennas=(0.5, 0.5), duration=0.2)
            time.sleep(0.6)
            robot.goto_target(antennas=(0.0, 0.0), duration=0.3)
        elif style == "sad":
            # Droopy sad
            robot.goto_target(antennas=(-0.4, -0.4), duration=0.4)
            time.sleep(0.8)
            robot.goto_target(antennas=(0.0, 0.0), duration=0.4)
        elif style == "excited":
            # Fast excited wiggle
            for _ in range(5):
                robot.goto_target(antennas=(0.4, -0.4), duration=0.1)
                time.sleep(0.1)
                robot.goto_target(antennas=(-0.4, 0.4), duration=0.1)
                time.sleep(0.1)
            robot.goto_target(antennas=(0.0, 0.0), duration=0.2)
    except:
        pass

# ============== TOOL EXECUTION ==============

# Global state for features
camera_available = CV2_AVAILABLE  # Camera available if opencv is installed

def execute_move_head(direction: str, intensity: str = "normal") -> dict:
    """Execute head movement based on tool call."""
    multipliers = {"subtle": 0.5, "normal": 1.0, "exaggerated": 1.5}
    mult = multipliers.get(intensity, 1.0)

    try:
        if direction == "left":
            robot.goto_target(head=create_head_pose(yaw=30*mult, degrees=True), duration=0.4)
            time.sleep(0.5)
            robot.goto_target(head=create_head_pose(), duration=0.3)
        elif direction == "right":
            robot.goto_target(head=create_head_pose(yaw=-30*mult, degrees=True), duration=0.4)
            time.sleep(0.5)
            robot.goto_target(head=create_head_pose(), duration=0.3)
        elif direction == "up":
            robot.goto_target(head=create_head_pose(pitch=-20*mult, degrees=True), duration=0.4)
            time.sleep(0.5)
            robot.goto_target(head=create_head_pose(), duration=0.3)
        elif direction == "down":
            robot.goto_target(head=create_head_pose(pitch=20*mult, degrees=True), duration=0.4)
            time.sleep(0.5)
            robot.goto_target(head=create_head_pose(), duration=0.3)
        elif direction == "center":
            robot.goto_target(head=create_head_pose(), duration=0.3)
        elif direction == "nod":
            nod()
        elif direction == "shake":
            shake_head()
        elif direction == "tilt":
            tilt_head()
        elif direction == "look_around":
            look_around()
        elif direction == "excited":
            excited()
        else:
            return {"status": "error", "message": f"unknown direction: {direction}"}

        return {"status": "success", "action": f"moved head {direction}"}
    except Exception as e:
        return {"status": "error", "message": str(e)}

def execute_wiggle_antennas(style: str) -> dict:
    """Execute antenna wiggle based on tool call."""
    try:
        threading.Thread(target=wiggle_antennas, args=(style,), daemon=True).start()
        return {"status": "success", "action": f"wiggling antennas {style}"}
    except Exception as e:
        return {"status": "error", "message": str(e)}

def execute_camera(query: str = None) -> dict:
    """Capture camera frame and return base64 for Claude vision."""
    if not camera_available:
        return {"status": "error", "message": "camera not available - opencv-python not installed"}

    b64_image, _ = capture_frame()

    if b64_image is None:
        return {"status": "error", "message": "failed to capture frame from camera"}

    # Return the image for Claude to analyze
    return {
        "status": "success",
        "b64_image": b64_image,
        "query": query or "Describe what you see"
    }

def execute_head_tracking(enabled: bool) -> dict:
    """Enable/disable face tracking."""
    if enabled:
        success = start_face_tracking()
        if success:
            return {"status": "success", "tracking": "enabled"}
        else:
            return {"status": "error", "message": "face tracking not available - missing dependencies"}
    else:
        stop_face_tracking()
        return {"status": "success", "tracking": "disabled"}

def execute_do_nothing() -> dict:
    """Explicitly do nothing."""
    return {"status": "success", "action": "remained still"}

def dispatch_tool(tool_name: str, tool_input: dict) -> dict:
    """Route tool calls to appropriate handlers."""
    if tool_name == "move_head":
        return execute_move_head(
            direction=tool_input.get("direction", "center"),
            intensity=tool_input.get("intensity", "normal")
        )
    elif tool_name == "wiggle_antennas":
        return execute_wiggle_antennas(style=tool_input.get("style", "happy"))
    elif tool_name == "camera":
        return execute_camera(query=tool_input.get("query"))
    elif tool_name == "head_tracking":
        return execute_head_tracking(enabled=tool_input.get("enabled", True))
    elif tool_name == "do_nothing":
        return execute_do_nothing()
    else:
        return {"status": "error", "message": f"unknown tool: {tool_name}"}

# ============== MAIN LOOP ==============

def clean_for_speech(text: str) -> str:
    """Remove asterisks and clean text."""
    import re
    text = re.sub(r'\*[^*]+\*', '', text)
    return ' '.join(text.split()).strip()

def main():
    print("\n" + "="*40)
    print("RODREEGO ASSISTANT (Mac M4 Version)")
    print("="*40)
    print("Say something to Rodreego!")
    print("Press Ctrl+C to quit\n")

    # Greeting
    greeting = "Hello Boss! I'm Rodreego, ready to help."
    print(f"Rodreego: {greeting}")
    audio = synthesize(greeting)
    if audio is not None:
        play_audio(audio)

    while True:
        try:
            # Record
            audio = record_audio(5.0)

            if len(audio) == 0 or np.abs(audio).max() < 0.01:
                print("(no audio detected)")
                continue

            # Transcribe
            start = time.time()
            text = transcribe(audio)
            stt_time = time.time() - start

            if not text or len(text.strip()) < 2:
                continue

            print(f"You: {text} ({stt_time:.1f}s)")

            # Check for exit
            if text.lower().strip() in ["exit", "quit", "goodbye", "bye"]:
                reply = "Goodbye Boss! Have a great day."
                print(f"Rodreego: {reply}")
                audio = synthesize(reply)
                if audio is not None:
                    play_audio(audio)
                break

            # Get response
            start = time.time()
            reply, tool_results = chat(text)
            llm_time = time.time() - start

            print(f"Rodreego: {reply} ({llm_time:.1f}s)")

            # Speak
            clean_reply = clean_for_speech(reply)
            start = time.time()
            audio = synthesize(clean_reply)
            tts_time = time.time() - start

            if audio is not None:
                play_audio(audio)
                print(f"(TTS: {tts_time:.1f}s)")

        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")
            continue

    robot.client.disconnect()

if __name__ == "__main__":
    main()
