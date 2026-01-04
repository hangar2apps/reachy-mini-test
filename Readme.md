# Morning Routine

A Reachy Mini app that performs a wake-up animation and announces the weather for El Reno, Oklahoma.

## Features

- **Wake-up animation**: Head raises from sleep, looks around, stretches
- **Antenna wiggle**: Happy greeting gesture
- **Weather report**: Fetches current conditions from Open-Meteo (free, no API key)
- **TTS greeting**: Speaks the day and weather

## Dev Setup (Mac)

```bash
# Install uv (recommended by Pollen)
brew install uv

# Create project directory
mkdir -p ~/dev/reachy-projects && cd ~/dev/reachy-projects

# Create and activate venv
uv venv .venv
source .venv/bin/activate

# Install dependencies
uv pip install reachy-mini requests numpy

# Clone/copy this app into the directory
```

## Usage

### Test from Mac (remote)

```bash
cd ~/dev/reachy-projects/morning_routine
source ../.venv/bin/activate
python -m morning_routine.main --remote
```

### From the Dashboard

1. Copy to Pi: `scp -r morning_routine pollen@reachy-mini.local:/home/pollen/my-apps/`
2. SSH in: `ssh pollen@reachy-mini.local` (password: root)
3. Install: `cd /home/pollen/my-apps/morning_routine && pip install -e .`
4. Run from `http://reachy-mini.local:8000/`

### Direct execution on Pi

```bash
# On the Pi
python -m morning_routine.main
```

### Cron job (8am daily)

```bash
ssh pollen@reachy-mini.local
crontab -e
# Add:
0 8 * * * /home/pollen/.local/bin/python -m morning_routine.main >> /tmp/morning.log 2>&1
```

## Configuration

Edit `main.py` to change:
- `LATITUDE` / `LONGITUDE` for different location
- Weather units (currently Fahrenheit / mph)
- Animation timings and movements