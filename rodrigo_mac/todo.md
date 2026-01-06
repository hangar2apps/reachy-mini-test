# Rodreego Personal Assistant - Feature Roadmap

## Completed Features

| Feature | Description | Status |
|---------|-------------|--------|
| Tool System | Claude function calling instead of keyword triggers | ✅ Done |
| Vision/Camera | Camera capture + Claude vision analysis | ✅ Done |
| Face Tracking | MediaPipe head tracking (follows faces) | ✅ Done |
| Head Wobble | Speech-reactive head + antenna movement | ✅ Done |
| Antenna Control | Emotional antenna expressions (happy, curious, alert, sad, excited) | ✅ Done |

## Skipped (for now)

| Feature | Description | Status |
|---------|-------------|--------|
| Web UI | Gradio interface with live transcripts | ⏭️ Skipped |
| Local Vision | Optional SmolVLM2 on-device | ⏭️ Skipped |

---

## Planned Features - Information Access

| Feature | Description | Priority |
|---------|-------------|----------|
| Weather | Current weather, forecast, alerts (OpenWeatherMap or WeatherAPI) | High |
| Calendar | Google Calendar integration - view/create events, reminders | High |
| Email | Gmail integration - read/summarize/compose emails | High |
| Web Search | Search the internet for current info (Tavily, SerpAPI, or Brave) | High |

## Planned Features - Smart Home & IoT

| Feature | Description | Priority |
|---------|-------------|----------|
| Home Assistant | Control lights, thermostat, locks, etc. | Medium |
| Music Control | Spotify/Apple Music - play, pause, queue songs | Medium |
| Smart Displays | Send info to nearby screens/tablets | Low |

## Planned Features - Productivity

| Feature | Description | Priority |
|---------|-------------|----------|
| Reminders | Set timed reminders, recurring tasks | High |
| Notes | Take voice notes, save to Notion/Obsidian | Medium |
| Timers/Alarms | "Set a timer for 10 minutes" | High |
| Todo Lists | Manage task lists, mark complete | Medium |

## Planned Features - Communication

| Feature | Description | Priority |
|---------|-------------|----------|
| SMS/iMessage | Send texts via shortcuts or API | Medium |
| Phone Calls | Initiate calls, announce callers | Low |
| Slack/Discord | Post messages, read notifications | Low |

## Planned Features - Context & Memory

| Feature | Description | Priority |
|---------|-------------|----------|
| Persistent Memory | Remember user preferences, past conversations | High |
| User Profiles | Recognize different users (face + voice) | Medium |
| Location Awareness | Know time of day, suggest contextual actions | Medium |
| Proactive Alerts | "You have a meeting in 15 minutes" | Medium |

## Planned Features - Entertainment

| Feature | Description | Priority |
|---------|-------------|----------|
| News Briefing | Daily news summary from preferred sources | Medium |
| Jokes/Trivia | Tell jokes, play trivia games | Low |
| Storytelling | Tell stories, read audiobooks | Low |
| Sports Scores | Check scores for favorite teams | Low |

## Planned Features - Robot Personality

| Feature | Description | Priority |
|---------|-------------|----------|
| Wake Word | "Hey Rodreego" activation | High |
| Idle Animations | Breathing, looking around when bored | Medium |
| Mood System | Personality changes based on interactions | Low |
| Voice Customization | Different TTS voices/styles | Low |

## Planned Features - Advanced

| Feature | Description | Priority |
|---------|-------------|----------|
| Multi-turn Planning | Complex task execution with multiple steps | Medium |
| File Access | Read/summarize documents on your computer | Medium |
| Code Execution | Run Python scripts, automate tasks | Low |
| Learning | Adapt responses based on corrections | Low |

---

## Implementation Notes

### Weather (recommended first)
```python
# OpenWeatherMap free tier: 1000 calls/day
# Add tool: get_weather(location: str) -> dict
# API key in .env: OPENWEATHER_API_KEY
```

### Calendar (Google Calendar API)
```python
# Requires OAuth2 setup with Google Cloud Console
# Scopes: calendar.readonly, calendar.events
# Add tools: get_events(date), create_event(title, time, duration)
```

### Web Search (Tavily recommended)
```python
# Tavily has generous free tier, optimized for AI
# Add tool: search_web(query: str) -> list[results]
# API key in .env: TAVILY_API_KEY
```

---

## Dependencies to Add
```
# For weather
requests

# For Google Calendar/Gmail
google-auth google-auth-oauthlib google-auth-httplib2 google-api-python-client

# For web search
tavily-python
```
