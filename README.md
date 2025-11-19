# Groundwater Commons Game - Flask Deployment Guide

## What You Get

**Fixed problems:**
- ✅ Real-time updates (no refresh button)
- ✅ Session persistence (reload = auto-rejoin with same name)
- ✅ Cleaner UI with dark theme
- ✅ All your game logic intact

## Option 1: Railway (Recommended - Easiest)

**Cost:** Free tier works fine for your class size

### Steps:

1. **Install Git** (if you don't have it):
   - Mac: `brew install git`
   - Windows: Download from git-scm.com
   - Linux: `sudo apt install git`

2. **Create a GitHub repository:**
   ```bash
   cd /path/to/your/files
   git init
   git add .
   git commit -m "Initial commit"
   gh repo create groundwater-game --public --source=. --push
   ```
   (Or manually create on github.com and push)

3. **Deploy to Railway:**
   - Go to railway.app
   - Click "New Project" → "Deploy from GitHub repo"
   - Select your repository
   - Railway auto-detects Flask and deploys
   - You'll get a URL like: `groundwater-game.railway.app`

**That's it.** Share the URL with students.

---

## Option 2: Render

**Cost:** Free tier (spins down after inactivity but wakes up fast)

### Steps:

1. Push code to GitHub (same as Railway step 2)

2. **Deploy to Render:**
   - Go to render.com
   - Click "New" → "Web Service"
   - Connect GitHub repo
   - Settings:
     - **Build Command:** `pip install -r requirements.txt`
     - **Start Command:** `gunicorn --worker-class eventlet -w 1 app:app`
   - Click "Create Web Service"

3. **Add gunicorn to requirements.txt:**
   Add this line: `gunicorn==21.2.0`

---

## Option 3: Local Testing (Before Deployment)

```bash
# Install dependencies
pip install -r requirements.txt

# Run the app
python app.py

# Open browser to: http://localhost:5000
```

**Test with multiple browser tabs** to simulate students joining.

---

## File Structure

```
groundwater-game/
├── app.py                 # Flask backend
├── templates/
│   └── index.html        # Frontend
├── requirements.txt      # Python dependencies
├── commons_game.db       # Created automatically
└── README.md            # This file
```

---

## How It Works

### For Students (Players):
1. Go to your deployed URL
2. Click "Join as Player"
3. Enter room code + name
4. **If they reload:** Just re-enter same name → auto-rejoins same spot

### For You (Host):
1. Go to "Host Controls" tab
2. Set parameters (or use defaults)
3. Click "Create Room" → get 5-digit code
4. Share code with students
5. When everyone's joined, click "Start Game"
6. Rounds auto-advance when all players submit (no refresh needed!)

---

## Parameters Explained

- **P**: Price per unit water
- **γ (gamma)**: Demand curve slope (quadratic cost)
- **c₀**: Base extraction cost
- **c₁**: Depth-dependent cost coefficient
- **S₀**: Initial aquifer stock
- **Smax**: Maximum aquifer capacity
- **R**: Natural recharge per round
- **qmax**: Maximum pumping per player per round
- **T**: Total rounds (default 8)
- **Players per Room**: 3-7 (default 6)

---

## Troubleshooting

**Problem:** Students can't connect
- Check Railway/Render logs for errors
- Ensure URL is `https://` not `http://`

**Problem:** Game feels slow
- Free tiers sleep after 15 min inactivity
- First request wakes them (10-30 sec)
- Upgrade to paid tier ($5/mo) for instant response

**Problem:** Database corrupted
- Delete `commons_game.db` file
- App creates fresh one on restart

**Problem:** Need to reset mid-class
- Host tab → "Create New Room"
- Old rooms stay in DB but are inactive

---

## Cost Breakdown

| Service | Free Tier | Paid Upgrade |
|---------|-----------|--------------|
| Railway | 500 hrs/mo | $5/mo unlimited |
| Render  | Spins down | $7/mo always-on |

**Recommendation:** Start free. Upgrade only if spin-down bugs you.

---

## Advanced: Custom Domain

If you want `groundwater.youruniversity.edu` instead of Railway URL:

1. Buy domain or use university domain
2. Railway: Settings → Add custom domain → Follow DNS instructions
3. Done

---

## Questions?

- Railway docs: docs.railway.app
- Render docs: render.com/docs
- Flask-SocketIO: flask-socketio.readthedocs.io

**Good luck with your class!**
