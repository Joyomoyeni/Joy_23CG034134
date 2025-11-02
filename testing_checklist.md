# 🧪 Testing Checklist

## Pre-Submission Testing Guide

Use this checklist to ensure everything works perfectly before submission.

---

## 🖥️ Local Testing

### Installation Test
- [ ] Virtual environment created successfully
- [ ] All dependencies installed without errors
- [ ] No missing packages when running app
- [ ] Database file created automatically
- [ ] Uploads folder created automatically

### Application Startup
- [ ] `python app.py` runs without errors
- [ ] Server starts on port 5000
- [ ] Console shows "Database initialized successfully!"
- [ ] No warning messages
- [ ] Can access http://localhost:5000

---

## 🎨 Frontend Testing

### Page Load
- [ ] Homepage loads without errors
- [ ] All CSS styles applied correctly
- [ ] No broken images or links
- [ ] Responsive on mobile devices
- [ ] No console errors in browser DevTools

### Upload Section
- [ ] Name input field works
- [ ] File input accepts images
- [ ] File input rejects non-image files
- [ ] "Detect Emotion" button is clickable
- [ ] Loading spinner appears during processing
- [ ] Results display correctly
- [ ] All emotion bars render
- [ ] Confidence percentages shown
- [ ] Error messages display when needed

### Camera Section
- [ ] Name input field works
- [ ] "Start Camera" button works
- [ ] Browser asks for camera permission
- [ ] Video feed displays correctly
- [ ] "Capture & Detect" button enables after camera starts
- [ ] Loading spinner appears during processing
- [ ] Results display correctly
- [ ] "Stop Camera" works
- [ ] Error messages display when needed

### Records Section
- [ ] "Refresh Records" button works
- [ ] Table displays correctly
- [ ] Recent detections appear
- [ ] Emotion badges have correct colors
- [ ] Timestamps format correctly
- [ ] Empty state shows when no records

---

## 🔧 Backend Testing

### Upload Endpoint
- [ ] Accepts POST requests
- [ ] Validates file types
- [ ] Handles missing name gracefully
- [ ] Handles missing file gracefully
- [ ] Processes images correctly
- [ ] Returns JSON response
- [ ] Saves to database
- [ ] Stores uploaded file

### Camera Endpoint
- [ ] Accepts POST requests
- [ ] Handles base64 image data
- [ ] Validates name input
- [ ] Processes images correctly
- [ ] Returns JSON response
- [ ] Saves to database
- [ ] Stores captured image

### Records Endpoint
- [ ] Returns all records
- [ ] Orders by timestamp (newest first)
- [ ] Limits to 50 records
- [ ] Handles empty database
- [ ] Returns correct JSON format

### Health Endpoint
- [ ] Returns 200 status
- [ ] Shows current timestamp
- [ ] Indicates app is running

---

## 💾 Database Testing

### Table Creation
- [ ] `emotion_records` table exists
- [ ] All columns created correctly
- [ ] Primary key auto-increments
- [ ] Timestamp defaults work

### Data Storage
- [ ] Name saves correctly
- [ ] Image path saves
- [ ] Image data stores as BLOB
- [ ] Detected emotion saves
- [ ] Confidence score saves
- [ ] All emotions JSON string saves
- [ ] Source type (upload/camera) saves
- [ ] Timestamp auto-generates

### Data Retrieval
- [ ] Can query all records
- [ ] Can filter by name
- [ ] Can order by timestamp
- [ ] Can limit results

---

## 🤖 Emotion Detection Testing

### Test Different Emotions

Upload/capture images for each emotion:

- [ ] **Happy**: Clear smile, should detect >80% confidence
- [ ] **Sad**: Frown, downturned mouth
- [ ] **Angry**: Furrowed brows, intense expression
- [ ] **Surprise**: Raised eyebrows, open mouth
- [ ] **Fear**: Wide eyes, tense expression
- [ ] **Disgust**: Wrinkled nose, grimace
- [ ] **Neutral**: Relaxed, no strong expression

### Edge Cases

- [ ] Very small face
- [ ] Very large face
- [ ] Face at angle
- [ ] Multiple faces (uses first detected)
- [ ] No face visible (handles error)
- [ ] Blurry image
- [ ] Low light image
- [ ] Black and white image
- [ ] High resolution image
- [ ] Low resolution image

### Performance
- [ ] First detection (model download) < 60 seconds
- [ ] Subsequent detections < 5 seconds
- [ ] Multiple rapid requests handled
- [ ] No memory leaks over time

---

## 🌐 Deployment Testing

### Pre-Deployment
- [ ] requirements.txt is complete
- [ ] No hardcoded localhost URLs
- [ ] No debug mode in production
- [ ] .gitignore excludes sensitive files
- [ ] All file paths are relative
- [ ] Environment variables configured

### Post-Deployment
- [ ] Site loads on hosting URL
- [ ] HTTPS is enabled
- [ ] Camera permission request works
- [ ] Upload still works
- [ ] Database persists (if applicable)
- [ ] No CORS errors
- [ ] Mobile responsive
- [ ] Works in different browsers
- [ ] Loading times acceptable

### Hosting Platform Checks
- [ ] Build completed successfully
- [ ] No build errors in logs
- [ ] Runtime errors checked
- [ ] Enough memory allocated
- [ ] Disk space sufficient
- [ ] Cold start time acceptable

---

## 📱 Cross-Browser Testing

### Desktop Browsers
- [ ] Google Chrome (latest)
- [ ] Mozilla Firefox (latest)
- [ ] Microsoft Edge (latest)
- [ ] Safari (if on Mac)

### Mobile Browsers
- [ ] Chrome Mobile (Android)
- [ ] Safari Mobile (iOS)
- [ ] Samsung Internet
- [ ] Firefox Mobile

### Functionality per Browser
- [ ] File upload works
- [ ] Camera access works
- [ ] Results display correctly
- [ ] Styling consistent
- [ ] No JavaScript errors

---

## 🔒 Security Testing

### Input Validation
- [ ] Name field sanitized
- [ ] File type validation works
- [ ] File size limit enforced
- [ ] Base64 data validated
- [ ] SQL injection prevented
- [ ] XSS attacks prevented

### File Handling
- [ ] Uploaded files have safe names
- [ ] Files stored securely
- [ ] Old files cleaned up (optional)
- [ ] No path traversal possible
- [ ] File permissions correct

---

## 📊 Data Accuracy Testing

### Confidence Scores
- [ ] Scores between 0-100%
- [ ] Dominant emotion has highest score
- [ ] All 7 emotions have scores
- [ ] Scores sum to ~100%

### Database Accuracy
- [ ] Correct emotion stored
- [ ] Confidence matches display
- [ ] Timestamp accurate
- [ ] Source type correct
- [ ] Image data retrievable

---

## 🎯 Assignment Requirements Check

### Files Present
- [ ] app.py exists
- [ ] model.py exists
- [ ] requirements.txt exists
- [ ] link_to_my_web_app.txt exists
- [ ] templates/index.html exists
- [ ] README.md exists (optional but recommended)

### Functionality Requirements
- [ ] Uses pre-trained model (DeepFace)
- [ ] Detects from uploaded images
- [ ] Detects from live camera
- [ ] Stores results in database
- [ ] Shows user name
- [ ] Shows detected emotion
- [ ] Shows confidence/probability
- [ ] Works online and offline

### Folder Naming
- [ ] Folder name format: `SURNAME_MATNO_EMOTION_DETECTION_WEB_APP`
- [ ] All CAPS for surname
- [ ] Correct matric number

### Submission Package
- [ ] GitHub repository created
- [ ] Repository is public
- [ ] All files committed
- [ ] App deployed to hosting platform
- [ ] Hosting link works
- [ ] link_to_my_web_app.txt updated
- [ ] Folder zipped correctly
- [ ] Zip file named correctly

---

## 📸 Documentation Testing

### README Quality
- [ ] Installation instructions clear
- [ ] Usage examples provided
- [ ] Screenshots included (optional)
- [ ] Dependencies listed
- [ ] Contact information included

### Code Documentation
- [ ] Functions have docstrings
- [ ] Complex logic explained
- [ ] Variable names meaningful
- [ ] Comments where needed
- [ ] No commented-out code blocks

---

## 🎬 Demo Preparation

### Screenshots to Take
- [ ] Homepage
- [ ] Upload with results
- [ ] Camera with results
- [ ] Records table with data
- [ ] Database viewer (optional)

### Demo Script
- [ ] Can explain how it works
- [ ] Can demo upload feature
- [ ] Can demo camera feature
- [ ] Can show database records
- [ ] Can explain the model used

---

## ✅ Final Checklist

Before submission, confirm:

- [ ] App works perfectly locally
- [ ] App deployed and accessible online
- [ ] All features tested thoroughly
- [ ] Database storing data correctly
- [ ] No errors in browser console
- [ ] No errors in server logs
- [ ] Code is clean and documented
- [ ] GitHub repo is complete
- [ ] link_to_my_web_app.txt is accurate
- [ ] Folder structure is correct
- [ ] Zip file created correctly
- [ ] Email ready to send

---

## 🐛 Common Issues Found During Testing

### Issue: "No module named 'deepface'"
**Fix**: `pip install deepface`

### Issue: Camera not working locally
**Fix**: Use http://localhost:5000, camera works here

### Issue: Camera not working on deployed site
**Fix**: Ensure site uses HTTPS

### Issue: First detection very slow
**Fix**: Normal! DeepFace downloads models first time

### Issue: "Database is locked"
**Fix**: Close all connections, restart app

### Issue: Upload fails with large images
**Fix**: Compress image or increase MAX_CONTENT_LENGTH

### Issue: Detection gives wrong emotion
**Fix**: Ensure good lighting, clear face, front-facing

---

## 📧 Ready to Submit?

**Final 3-Step Check:**

1. **Test locally** ✓
   - Run through all tests above
   - Fix any issues found

2. **Test deployed version** ✓
   - Verify hosted site works
   - Test all features online

3. **Prepare submission** ✓
   - Update all documents
   - Create zip file
   - Send email

---

**You're all set! 🎉**

*Print this checklist and tick off each item as you test. This ensures nothing is missed before submission.*
