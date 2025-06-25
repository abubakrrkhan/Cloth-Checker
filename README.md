
# Cloth Checker - AI Fashion Advisor

An AI-powered fashion advisor application that analyzes clothing and provides personalized style recommendations based on skin tone and fashion preferences.

## ✨ Key Features

### 1. Skin Analysis
- Advanced AI detection of skin tone
- Undertone analysis (warm, cool, neutral)
- Personalized color recommendations
- Skin-matching makeup suggestions

### 2. Clothing Detection
- Automatic garment type identification
- Color and pattern analysis
- Style classification
- Fit analysis

### 3. Smart Recommendations
- Personalized color palettes
- Outfit coordination suggestions
- Shopping recommendations
- Seasonal style advice

### 4. Interactive Features
- Virtual wardrobe management
- AI Fashion Assistant chat
- Personal style profile
- Saved analyses history

## 🛠️ Technologies Used

### Backend
- Python 3.7+
- Flask web framework
- OpenCV for image processing
- TensorFlow/PyTorch for AI models
- SQLite database (MySQL optional)

### Frontend
- HTML5, CSS3, JavaScript
- Bootstrap for responsive design
- AJAX for dynamic interactions
- Webcam integration

### AI Components
- Custom machine learning models
- Computer vision algorithms
- Natural language processing
- Color theory algorithms

## 📦 Installation

1. Clone the repository:
```bash
git clone https://github.com/your-username/cloth-checker.git
cd cloth-checker
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
python app.py
```

4. Access the web interface:
   - Open browser and go to: http://localhost:5000

## 📁 Project Structure
cloth-checker/
├── app.py # Main Flask application
├── models.py # Database models
├── requirements.txt # Project dependencies
├── static/ # Static files (CSS, JS, images)
├── templates/ # HTML templates
├── modules/
│ ├── clothing_detector.py # Clothing detection
│ ├── color_recommender.py # Color recommendations
│ ├── skin_analyzer.py # Skin tone analysis
│ ├── shopping_recommender.py # Shopping suggestions
│ ├── outfit_recommender.py # Outfitrecommendations
│ └── fashion_assistant.py # AI chatbot logic
└── utils/ # Utility functions

## 💻 Usage Guide

### Camera Analysis
1. Navigate to Camera Analysis section
2. Ensure good lighting conditions
3. Position yourself/clothing item
4. Capture image for analysis
5. Review recommendations

### Image Upload
1. Select or drag-drop image
2. Wait for AI processing
3. View detailed analysis
4. Get personalized recommendations

### Virtual Wardrobe
1. Add items from recommendations
2. Upload existing clothing items
3. Create outfit combinations
4. Save favorite looks

### Fashion Assistant
1. Chat with AI assistant
2. Ask style questions
3. Get personalized advice
4. Save recommendations

## ⚙️ Requirements

### Core Dependencies
- Python 3.7+
- OpenCV
- NumPy
- Flask
- SQLite/MySQL
- PyTorch/TensorFlow

### Additional Libraries
- Pillow
- scikit-image
- scikit-learn
- Flask-SQLAlchemy
- Other dependencies in requirements.txt

## 🔧 Configuration

### Environment Setup
1. Copy example environment file:
```bash
cp env.example .env
```

2. Configure variables in .env:
   
### Database Setup
- SQLite (default):
  - Automatically created on first run
- MySQL (optional):
  - Run setup script: `python setup_db.py`
  - Configure in .env file

## 🌟 Features in Detail

### Skin Analysis
- Advanced tone detection
- Undertone classification
- Seasonal color analysis
- Custom color recommendations

### Clothing Detection
- Multiple item detection
- Pattern recognition
- Style categorization
- Brand identification (where possible)

### Smart Recommendations
- Color coordination
- Style matching
- Occasion-based suggestions
- Seasonal recommendations

### User Features
- Personal profiles
- Analysis history
- Favorite styles
- Shopping lists

## 📱 Mobile Support
- Responsive design
- Mobile-friendly interface
- Camera integration
- Touch-optimized controls

## 🔒 Security Features
- Secure image handling
- User data protection
- Privacy-focused analysis
- Safe storage practices

## 🤝 Contributing
1. Fork the repository
2. Create feature branch
3. Commit changes
4. Push to branch
5. Create Pull Request

## 📄 License
This project is licensed under the MIT License - see the LICENSE file for details.

## 👥 Authors
- Your Name
- Contributors welcome

## 🙏 Acknowledgments
- Fashion AI community
- Open source contributors
- Color theory experts
- Computer vision researchers
