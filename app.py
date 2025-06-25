from flask import Flask, render_template, request, jsonify, redirect, url_for, session, flash
import os
import cv2
import numpy as np
from PIL import Image
import io
import base64
from datetime import datetime
import traceback
import re
import sys
from skin_analyzer import SkinToneAnalyzer
from color_recommender import ColorRecommender
from shopping_recommender import ShoppingRecommender
from clothing_detector import ClothingDetector
from outfit_recommender import OutfitRecommender
from fashion_assistant import FashionAssistant
from jinja2 import FileSystemLoader
from werkzeug.utils import secure_filename
from dotenv import load_dotenv
import uuid
import json
from flask_sqlalchemy import SQLAlchemy
from db import db

# Load environment variables
load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv('SECRET_KEY', 'your_fallback_secret_key')

# Configure database to use SQLite
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///cloth_checker.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Initialize SQLAlchemy
db.init_app(app)

# Import models after db initialization
from models import User, Analysis, DetectedClothing, RecommendedColor, ShoppingSuggestion, ClothingAnalysis, WardrobeItem

# Initialize database
try:
    # Check database connection
    with app.app_context():
        db.create_all()
    db_available = True
    print("Successfully connected to SQLite database")
except Exception as e:
    print(f"Database initialization error: {e}")
    print("Running without database support")
    db_available = False

# Upload folder configuration
UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure upload directory exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Custom template loader to handle different encodings
custom_loader = FileSystemLoader('templates', encoding='utf-8')
app.jinja_loader = custom_loader

# Initialize the analyzers and recommenders
try:
    skin_analyzer = SkinToneAnalyzer()
    color_recommender = ColorRecommender()
    shopping_recommender = ShoppingRecommender()
    clothing_detector = ClothingDetector()
    outfit_recommender = OutfitRecommender()
    fashion_assistant = FashionAssistant()
    print("Successfully initialized analyzers and recommenders")
except Exception as e:
    print(f"Error initializing components: {str(e)}")
    print(traceback.format_exc())

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        print("Received analyze request")
        
        if 'image' in request.files:
            print("Processing uploaded image file")
            # Handle uploaded image
            file = request.files['image']
            if file.filename != '':
                # Read the image file
                file_bytes = file.read()
                nparr = np.frombuffer(file_bytes, np.uint8)
                image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                if image is None:
                    print("Failed to decode uploaded image")
                    return jsonify({'error': 'Invalid image file'}), 400
                
                # Save the uploaded file
                filename = f"{datetime.now().strftime('%Y%m%d%H%M%S')}.jpg"
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                cv2.imwrite(filepath, image)
                print(f"Saved uploaded image to {filepath}")
                
                # Store the image path in session
                session['image_path'] = filepath.replace('static/', '')
                
                # Analyze the image
                result = process_image(image)
                
                # Store results in session
                session['skin_tone'] = result['skin_tone']
                session['undertone'] = result['undertone']
                session['clothing_colors'] = result['clothing_colors']
                session['makeup_colors'] = result['makeup_colors']
                session['accessory_colors'] = result['accessory_colors']
                session['shopping_suggestions'] = result['shopping_suggestions']
                session['detected_clothing'] = result['detected_clothing']
                session['outfit_recommendations'] = result['outfit_recommendations']
                
                return redirect(url_for('results'))
        
        elif 'image_data' in request.form:
            print("Processing image data from form")
            # Handle base64 image data from webcam
            image_data = request.form['image_data']
            
            # Extract the base64 data
            if image_data.startswith('data:image/jpeg;base64,'):
                print("Detected JPEG image data")
                image_data = image_data.replace('data:image/jpeg;base64,', '')
            elif image_data.startswith('data:image/png;base64,'):
                print("Detected PNG image data")
                image_data = image_data.replace('data:image/png;base64,', '')
            else:
                print(f"Unknown image format: {image_data[:30]}...")
            
            # Clean the base64 string
            image_data = re.sub(r'\s+', '', image_data)
            
            try:
                # Decode the base64 data
                print("Decoding base64 data")
                image_bytes = base64.b64decode(image_data)
                print(f"Decoded {len(image_bytes)} bytes")
                
                # Save raw bytes for debugging
                debug_raw_path = os.path.join(app.config['UPLOAD_FOLDER'], f"debug_raw_{datetime.now().strftime('%Y%m%d%H%M%S')}.bin")
                with open(debug_raw_path, 'wb') as f:
                    f.write(image_bytes)
                print(f"Saved raw bytes to {debug_raw_path}")
                
                # Direct OpenCV approach - simpler and more reliable
                print("Using direct OpenCV approach")
                nparr = np.frombuffer(image_bytes, np.uint8)
                image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                if image is None:
                    print("Failed with direct OpenCV approach")
                    return jsonify({'error': 'Failed to decode image'}), 400
                
                # Save the image
                filename = f"{datetime.now().strftime('%Y%m%d%H%M%S')}.jpg"
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                cv2.imwrite(filepath, image)
                print(f"Saved image to {filepath}")
                
                # Store the image path in session
                session['image_path'] = filepath.replace('static/', '')
                
                # Analyze the image
                print("Analyzing image")
                result = process_image(image)
                print(f"Analysis result: {result}")
                
                # Store results in session
                session['skin_tone'] = result['skin_tone']
                session['undertone'] = result['undertone']
                session['clothing_colors'] = result['clothing_colors']
                session['makeup_colors'] = result['makeup_colors']
                session['accessory_colors'] = result['accessory_colors']
                session['shopping_suggestions'] = result['shopping_suggestions']
                session['detected_clothing'] = result['detected_clothing']
                session['outfit_recommendations'] = result['outfit_recommendations']
                
                return redirect(url_for('results'))
                
            except Exception as e:
                print(f"Base64 decode error: {str(e)}")
                print(traceback.format_exc())
                return jsonify({'error': 'Invalid image data: ' + str(e)}), 400
        
        print("No valid image data found in request")
        return jsonify({'error': 'No image data provided'}), 400
        
    except Exception as e:
        print(f"Error processing image: {str(e)}")
        print(traceback.format_exc())
        return jsonify({'error': 'Error processing image: ' + str(e)}), 500

def process_image(image):
    try:
        print("Starting image processing")
        
        # Analyze skin tone
        print("Analyzing skin tone")
        skin_tone, undertone = skin_analyzer.analyze(image)
        print(f"Skin tone: {skin_tone}, Undertone: {undertone}")
        
        # Detect clothing
        print("Detecting clothing")
        detected_clothing = clothing_detector.detect_clothing(image)
        print(f"Detected clothing: {detected_clothing}")
        
        # Get color recommendations
        print("Getting color recommendations")
        clothing_colors = color_recommender.get_clothing_colors(undertone)
        makeup_colors = color_recommender.get_makeup_colors(undertone)
        accessory_colors = color_recommender.get_accessory_colors(undertone)
        
        # Get outfit recommendations
        print("Getting outfit recommendations")
        outfit_recommendations = outfit_recommender.recommend_outfit(
            detected_clothing, 
            skin_tone=skin_tone, 
            undertone=undertone
        )
        
        # Get shopping recommendations
        print("Getting shopping recommendations")
        shopping_suggestions = shopping_recommender.get_suggestions(undertone)
        
        print("Image processing complete")
        return {
            'skin_tone': skin_tone,
            'undertone': undertone,
            'clothing_colors': clothing_colors,
            'makeup_colors': makeup_colors,
            'accessory_colors': accessory_colors,
            'shopping_suggestions': shopping_suggestions,
            'detected_clothing': detected_clothing,
            'outfit_recommendations': outfit_recommendations
        }
    except Exception as e:
        print(f"Error in process_image: {str(e)}")
        print(traceback.format_exc())
        raise

@app.route('/camera')
def camera():
    return render_template('camera.html')

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        
        file = request.files['file']
        
        # If user does not select file, browser also submits an empty part
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        
        if file and allowed_file(file.filename):
            # Generate a unique filename
            filename = secure_filename(str(uuid.uuid4()) + os.path.splitext(file.filename)[1])
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Store analysis in database if user is logged in and database is available
            analysis_id = None
            if db_available and 'user_id' in session:
                user_id = session.get('user_id')
                try:
                    analysis = Analysis(
                        user_id=user_id,
                        image_path=os.path.join('uploads', filename)
                    )
                    db.session.add(analysis)
                    db.session.commit()
                    
                    # This is where you'd process the image and update the analysis with real data
                    # For now, just pass dummy data
                    analysis_id = analysis.id
                    session['analysis_id'] = analysis_id
                except Exception as e:
                    print(f"Error saving to database: {e}")
                    # Continue without database storage
            
            # Process the image - in a real app, call your ML model here
            # For now, use mock data
            result = {
                'image_path': os.path.join('uploads', filename),
                'skin_tone': 'Warm Medium',
                'undertone': 'Warm',
                'clothing_colors': ['#1e88e5', '#43a047', '#ff6f00', '#8e24aa'],
                'makeup_colors': ['#ffb74d', '#a1887f', '#ce93d8'],
                'accessory_colors': ['#ffd54f', '#5c6bc0', '#66bb6a'],
                'detected_clothing': ['T-shirt', 'Jeans', 'Jacket'],
                'outfit_recommendations': {
                    'T-shirt': 'A fitted t-shirt complements your body type well.',
                    'Jeans': 'Straight-leg jeans work well with your proportions.',
                    'Jacket': 'A structured jacket adds polish to your look.'
                },
                'shopping_suggestions': {
                    'Tops': ['Burgundy blouse', 'Navy blue sweater', 'Forest green t-shirt'],
                    'Bottoms': ['Dark wash jeans', 'Camel trousers', 'Black skirt'],
                    'Accessories': ['Gold hoop earrings', 'Brown leather belt', 'Emerald scarf']
                }
            }
            
            # If we have a real analysis ID and database is available, populate the database
            if db_available and analysis_id:
                try:
                    # Update the analysis with mock data
                    analysis = Analysis.query.get(analysis_id)
                    if analysis:
                        analysis.skin_tone = result['skin_tone']
                        analysis.undertone = result['undertone']
                        
                        # Add detected clothing
                        for item in result['detected_clothing']:
                            recommendation = result['outfit_recommendations'].get(item, '')
                            clothing = DetectedClothing(
                                analysis_id=analysis_id,
                                item_name=item,
                                recommendation=recommendation
                            )
                            db.session.add(clothing)
                        
                        # Add recommended colors
                        for color in result['clothing_colors']:
                            rc = RecommendedColor(
                                analysis_id=analysis_id,
                                color_value=color,
                                color_type='clothing'
                            )
                            db.session.add(rc)
                        
                        for color in result['makeup_colors']:
                            rc = RecommendedColor(
                                analysis_id=analysis_id,
                                color_value=color,
                                color_type='makeup'
                            )
                            db.session.add(rc)
                        
                        for color in result['accessory_colors']:
                            rc = RecommendedColor(
                                analysis_id=analysis_id,
                                color_value=color,
                                color_type='accessory'
                            )
                            db.session.add(rc)
                        
                        # Add shopping suggestions
                        for category, suggestions in result['shopping_suggestions'].items():
                            for suggestion in suggestions:
                                ss = ShoppingSuggestion(
                                    analysis_id=analysis_id,
                                    category=category,
                                    suggestion=suggestion
                                )
                                db.session.add(ss)
                        
                        db.session.commit()
                except Exception as e:
                    print(f"Error updating database with analysis: {e}")
                    db.session.rollback()
                    # Continue without database storage
            
            # Store the result in session to retrieve on results page
            session['result'] = result
            return redirect(url_for('results'))
    
    return render_template('upload.html')

@app.route('/results')
def results():
    # Get result from session
    result = session.get('result', {})
    if not result:
        return redirect(url_for('upload'))
    
    return render_template('results.html', result=result)

@app.route('/chat', methods=['GET', 'POST'])
def chat():
    if request.method == 'POST':
        try:
            data = request.get_json()
            user_message = data.get('message', '')
            
            # Get context data for more personalized responses
            skin_tone = session.get('skin_tone')
            undertone = session.get('undertone')
            detected_clothing = session.get('detected_clothing', [])
            
            # Process the message
            assistant_response = fashion_assistant.process_message(
                user_message,
                clothing_items=detected_clothing,
                skin_tone=skin_tone,
                undertone=undertone
            )
            
            return jsonify({
                'response': assistant_response
            })
            
        except Exception as e:
            print(f"Error in chat processing: {str(e)}")
            print(traceback.format_exc())
            return jsonify({
                'response': "I'm sorry, I encountered an error processing your message. Could you try again?"
            }), 500
    
    # For GET requests, render the chat page
    return render_template('chat.html')

@app.route('/wardrobe')
def wardrobe():
    if not db_available:
        flash('Database features are currently unavailable')
        return redirect(url_for('index'))
        
    user_id = session.get('user_id')
    if not user_id:
        flash('Please log in to access your wardrobe')
        return redirect(url_for('login'))
    
    # Get user's wardrobe items
    wardrobe_items = WardrobeItem.query.filter_by(user_id=user_id).all()
    
    return render_template('wardrobe.html', wardrobe_items=wardrobe_items)

@app.route('/register', methods=['GET', 'POST'])
def register():
    if not db_available:
        flash('User registration is currently unavailable')
        return redirect(url_for('index'))
        
    # Add registration logic here
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if not db_available:
        flash('User login is currently unavailable')
        return redirect(url_for('index'))
        
    # Add login logic here
    return render_template('login.html')

@app.route('/profile')
def profile():
    if not db_available:
        flash('User profiles are currently unavailable')
        return redirect(url_for('index'))
        
    user_id = session.get('user_id')
    if not user_id:
        return redirect(url_for('login'))
    
    # Get user and their analysis history
    user = User.query.get(user_id)
    analyses = Analysis.query.filter_by(user_id=user_id).order_by(Analysis.created_at.desc()).all()
    
    return render_template('profile.html', user=user, analyses=analyses)

@app.route('/analysis/<int:analysis_id>')
def view_analysis(analysis_id):
    if not db_available:
        flash('Analysis history is currently unavailable')
        return redirect(url_for('index'))
        
    analysis = Analysis.query.get_or_404(analysis_id)
    
    # Ensure the user can only view their own analyses
    user_id = session.get('user_id')
    if not user_id or analysis.user_id != user_id:
        flash('You do not have permission to view this analysis')
        return redirect(url_for('login'))
    
    # Build result object similar to the one used in results()
    result = {
        'image_path': analysis.image_path,
        'skin_tone': analysis.skin_tone,
        'undertone': analysis.undertone,
        'clothing_colors': [c.color_value for c in analysis.recommended_colors if c.color_type == 'clothing'],
        'makeup_colors': [c.color_value for c in analysis.recommended_colors if c.color_type == 'makeup'],
        'accessory_colors': [c.color_value for c in analysis.recommended_colors if c.color_type == 'accessory'],
        'detected_clothing': [c.item_name for c in analysis.clothing_items],
        'outfit_recommendations': {c.item_name: c.recommendation for c in analysis.clothing_items},
        'shopping_suggestions': {}
    }
    
    # Organize shopping suggestions by category
    suggestions = ShoppingSuggestion.query.filter_by(analysis_id=analysis_id).all()
    for suggestion in suggestions:
        if suggestion.category not in result['shopping_suggestions']:
            result['shopping_suggestions'][suggestion.category] = []
        result['shopping_suggestions'][suggestion.category].append(suggestion.suggestion)
    
    return render_template('results.html', result=result)

@app.route('/api/clothing-analysis', methods=['POST'])
def receive_analysis():
    """
    API endpoint to receive clothing analysis data from the client.
    The data is stored in the database and made available for viewing in the portal.
    """
    try:
        # Get JSON data from request
        data = request.json
        
        # Validate data
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        # Extract fields from data
        timestamp = data.get('timestamp', datetime.now().isoformat())
        image_name = data.get('image_name', 'unknown')
        
        # Create new database entry
        analysis = ClothingAnalysis(
            timestamp=datetime.fromisoformat(timestamp),
            image_name=image_name,
            analysis_data=json.dumps(data.get('analysis_data', {})),
            skin_analysis=json.dumps(data.get('skin_analysis', {})),
            recommendations=json.dumps(data.get('recommendations', {}))
        )
        
        # Save to database
        db.session.add(analysis)
        db.session.commit()
        
        return jsonify({
            'status': 'success',
            'message': 'Analysis data received and stored',
            'analysis_id': analysis.id
        })
        
    except Exception as e:
        # Log error and return error response
        print(f"Error processing analysis data: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/analyses')
def view_analyses():
    """View all analyses in the portal"""
    analyses = ClothingAnalysis.query.order_by(ClothingAnalysis.timestamp.desc()).all()
    return render_template('analyses.html', analyses=analyses)

@app.route('/analysis/<int:analysis_id>')
def view_analysis_detail(analysis_id):
    """View a specific analysis"""
    analysis = ClothingAnalysis.query.get_or_404(analysis_id)
    
    # Parse JSON data for template
    analysis_data = json.loads(analysis.analysis_data)
    skin_analysis = json.loads(analysis.skin_analysis)
    recommendations = json.loads(analysis.recommendations)
    
    return render_template(
        'analysis_detail.html', 
        analysis=analysis,
        analysis_data=analysis_data,
        skin_analysis=skin_analysis, 
        recommendations=recommendations
    )

if __name__ == '__main__':
    print(f"Python version: {sys.version}")
    print(f"OpenCV version: {cv2.__version__}")
    print(f"NumPy version: {np.__version__}")
    print(f"PIL version: {Image.__version__}")
    app.run(debug=True) 