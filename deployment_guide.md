# Deployment Guide for Emotion Detection App

## Step-by-Step Deployment Instructions

### 1. Prepare Your Files

Ensure your folder structure looks like this:
```
YOURSURNAME_YOURMATNO_EMOTION_DETECTION_WEB_APP/
├── app.py
├── model.py
├── requirements.txt
├── link_to_my_web_app.txt
├── README.md
├── templates/
│   └── index.html
├── .gitignore
└── Procfile (for Heroku)
```

### 2. Create .gitignore

Create a `.gitignore` file:
```
__pycache__/
*.pyc
*.pyo
*.pyd
.Python
*.so
*.db
*.sqlite3
uploads/
venv/
env/
.env
.venv
*.log
.DS_Store
fer2013.csv
*.h5
*.weights
.deepface/
```

### 3. GitHub Setup

```bash
# Initialize git repository
git init

# Add all files
git add .

# Commit
git commit -m "Initial commit: Emotion Detection Web App"

# Create repository on GitHub (via website)
# Then link and push:
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
git branch -M main
git push -u origin main
```

### 4. Deploy to Render (Recommended for Free Hosting)

**Why Render?**
- Free tier available
- Easy deployment
- Good for AI/ML apps
- Persistent storage options

**Steps:**

1. Go to [render.com](https://render.com) and sign up

2. Click "New +" → "Web Service"

3. Connect your GitHub repository

4. Configure:
   ```
   Name: emotion-detection-app-yourname
   Environment: Python 3
   Build Command: pip install -r requirements.txt
   Start Command: gunicorn app:app --bind 0.0.0.0:$PORT
   ```

5. Add Environment Variables (if needed):
   ```
   PYTHON_VERSION=3.10.0
   ```

6. Click "Create Web Service"

7. Wait for deployment (5-10 minutes first time)

8. Copy your URL: `https://emotion-detection-app-yourname.onrender.com`

### 5. Deploy to Railway (Alternative)

**Steps:**

1. Go to [railway.app](https://railway.app)

2. Sign up with GitHub

3. Click "New Project" → "Deploy from GitHub repo"

4. Select your repository

5. Railway auto-detects Python

6. Add `Procfile` with:
   ```
   web: gunicorn app:app
   ```

7. Deploy automatically

8. Get URL from Railway dashboard

### 6. Deploy to Heroku (Alternative)

**Steps:**

1. Create account at [heroku.com](https://heroku.com)

2. Install Heroku CLI:
   ```bash
   # For Mac
   brew tap heroku/brew && brew install heroku
   
   # For Windows
   # Download from heroku.com
   ```

3. Create `Procfile`:
   ```
   web: gunicorn app:app
   ```

4. Deploy:
   ```bash
   heroku login
   heroku create your-emotion-detection-app
   git push heroku main
   heroku open
   ```

### 7. Deploy to PythonAnywhere

**Steps:**

1. Sign up at [pythonanywhere.com](https://pythonanywhere.com)

2. Go to "Web" tab → "Add a new web app"

3. Choose "Flask" and Python 3.10

4. Upload your files via "Files" tab

5. Edit WSGI configuration file:
   ```python
   import sys
   path = '/home/yourusername/emotion-detection'
   if path not in sys.path:
       sys.path.append(path)
   
   from app import app as application
   ```

6. Install packages in Bash console:
   ```bash
   pip install -r requirements.txt
   ```

7. Reload web app

### 8. Update link_to_my_web_app.txt

After deployment, create/update this file:

```
Platform: Render
Link: https://emotion-detection-app-yourname.onrender.com
Deployed on: 2024-11-01
Status: Active
```

### 9. Test Your Deployment

**Checklist:**
- ✅ Homepage loads correctly
- ✅ Can upload images
- ✅ Camera access works (HTTPS required)
- ✅ Emotion detection returns results
- ✅ Database saves records
- ✅ Records page displays data

### 10. Common Deployment Issues

#### Issue: Camera not working
**Solution**: Ensure your app uses HTTPS. Most free platforms provide HTTPS by default.

#### Issue: DeepFace model download fails
**Solution**: 
- Increase startup timeout in platform settings
- Pre-download models in build step
- Use lighter models

#### Issue: Out of memory
**Solution**:
- Use smaller model (VGGFace instead of ResNet50)
- Reduce image size before processing
- Upgrade to paid tier if necessary

#### Issue: Database not persisting
**Solution**:
- Use persistent storage options
- For Render: Add persistent disk
- For Heroku: Use PostgreSQL addon
- For development: SQLite is fine

### 11. Optimize for Free Tier

Add to `app.py` for better performance:

```python
import os

# Use lighter model for free tier
USE_LITE_MODEL = os.getenv('USE_LITE_MODEL', 'true') == 'true'

# In detect function:
if USE_LITE_MODEL:
    detector_backend = 'opencv'  # Faster, less accurate
else:
    detector_backend = 'retinaface'  # Slower, more accurate
```

### 12. Alternative: Deploy with Replit

1. Go to [replit.com](https://replit.com)
2. Create new Repl → Import from GitHub
3. Run automatically
4. Get shareable link

### 13. Final Submission Checklist

Before submitting:

- [ ] All files in correct folder structure
- [ ] GitHub repository is public
- [ ] App is deployed and accessible
- [ ] link_to_my_web_app.txt is updated
- [ ] Database is working
- [ ] README is complete
- [ ] Code is well-commented
- [ ] Requirements.txt is accurate
- [ ] Folder named correctly: SURNAME_MATNO_EMOTION_DETECTION_WEB_APP
- [ ] Everything tested thoroughly

### 14. Create Submission Package

```bash
# Navigate to parent directory
cd ..

# Create zip file
zip -r SURNAME_MATNO_EMOTION_DETECTION_WEB_APP.zip SURNAME_MATNO_EMOTION_DETECTION_WEB_APP/

# Verify zip contents
unzip -l SURNAME_MATNO_EMOTION_DETECTION_WEB_APP.zip
```

### 15. Email Submission

**To**: odunayo.osofuye@covenantuniversity.edu.ng

**Subject**: Emotion Detection Web App Submission - [YOUR NAME] - [MAT NO]

**Body**:
```
Dear Dr. Osofuye,

Please find attached my Emotion Detection Web App submission.

Details:
- Name: [Your Full Name]
- Matric No: [Your Matric Number]
- GitHub Repo: [Your GitHub URL]
- Hosted App: [Your Deployment URL]

The app includes:
✓ Image upload detection
✓ Live camera detection
✓ Database storage
✓ Model training script
✓ Complete documentation

Thank you.

Best regards,
[Your Name]
```

## Recommended Hosting Platforms Summary

| Platform | Pros | Cons | Best For |
|----------|------|------|----------|
| **Render** | Free tier, easy, ML-friendly | Cold starts | Recommended |
| **Railway** | Auto-deploy, modern | Limited free hours | Fast deployment |
| **Heroku** | Reliable, well-documented | Paid only now | Production |
| **PythonAnywhere** | Python-specific, simple | Limited resources | Beginners |
| **Replit** | Instant, no setup | Public code | Quick demos |

## Pro Tips

1. **Test locally first**: Always test thoroughly on your machine
2. **Use environment variables**: Don't hardcode sensitive data
3. **Monitor logs**: Check platform logs for errors
4. **Keep it simple**: Start with basic features, add complexity later
5. **Document everything**: Good documentation = good grades
6. **Version control**: Commit frequently with clear messages
7. **Test on mobile**: Ensure responsive design works
8. **Check HTTPS**: Required for camera access
9. **Optimize images**: Compress uploaded images to save bandwidth
10. **Add error handling**: Graceful failures improve user experience

---

**Need Help?**
- Check platform documentation
- Review error logs carefully
- Test each feature individually
- Ask questions early

**Good luck with your deployment! 🚀**
