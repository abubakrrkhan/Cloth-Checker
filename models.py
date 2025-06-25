from datetime import datetime
from db import db
import uuid
import json

class User(db.Model):
    __tablename__ = 'users'
    
    id = db.Column(db.String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(128), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Relationships
    analyses = db.relationship('Analysis', backref='user', lazy=True)
    wardrobe_items = db.relationship('WardrobeItem', backref='user', lazy=True)
    
    def __repr__(self):
        return f'<User {self.username}>'


class Analysis(db.Model):
    __tablename__ = 'analyses'
    
    id = db.Column(db.String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = db.Column(db.String(36), db.ForeignKey('users.id'), nullable=False)
    image_path = db.Column(db.String(255), nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    skin_tone = db.Column(db.String(50), nullable=True)
    undertone = db.Column(db.String(50), nullable=True)
    
    # Relationships
    detected_items = db.relationship('DetectedClothing', backref='analysis', lazy=True)
    recommended_colors = db.relationship('RecommendedColor', backref='analysis', lazy=True)
    shopping_suggestions = db.relationship('ShoppingSuggestion', backref='analysis', lazy=True)
    
    def __repr__(self):
        return f'<Analysis {self.id}>'


class DetectedClothing(db.Model):
    __tablename__ = 'detected_clothing'
    
    id = db.Column(db.String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    analysis_id = db.Column(db.String(36), db.ForeignKey('analyses.id'), nullable=False)
    category = db.Column(db.String(50), nullable=False)
    color = db.Column(db.String(50))
    pattern = db.Column(db.String(50))
    confidence = db.Column(db.Float)
    bounding_box = db.Column(db.String(255))  # Stored as "x1,y1,x2,y2"
    item_name = db.Column(db.String(100))
    recommendation = db.Column(db.Text)
    
    def __repr__(self):
        return f'<DetectedClothing {self.category}>'


class RecommendedColor(db.Model):
    __tablename__ = 'recommended_colors'
    
    id = db.Column(db.String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    analysis_id = db.Column(db.String(36), db.ForeignKey('analyses.id'), nullable=False)
    color_hex = db.Column(db.String(7))  # #RRGGBB format
    color_name = db.Column(db.String(50))
    category = db.Column(db.String(50))  # e.g., 'top', 'pants', etc.
    reason = db.Column(db.Text)
    color_value = db.Column(db.String(50))
    color_type = db.Column(db.String(50))
    
    def __repr__(self):
        return f'<RecommendedColor {self.color_name}>'


class WardrobeItem(db.Model):
    __tablename__ = 'wardrobe_items'
    
    id = db.Column(db.String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = db.Column(db.String(36), db.ForeignKey('users.id'), nullable=False)
    item_type = db.Column(db.String(100), nullable=False)
    color = db.Column(db.String(30), nullable=True)
    description = db.Column(db.Text, nullable=True)
    image_path = db.Column(db.String(255), nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    def __repr__(self):
        return f'<WardrobeItem {self.item_type}>'


class ShoppingSuggestion(db.Model):
    __tablename__ = 'shopping_suggestions'
    
    id = db.Column(db.String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    analysis_id = db.Column(db.String(36), db.ForeignKey('analyses.id'), nullable=False)
    item_name = db.Column(db.String(100), nullable=False)
    description = db.Column(db.Text)
    category = db.Column(db.String(50))
    url = db.Column(db.String(255))
    price = db.Column(db.Float)
    image_url = db.Column(db.String(255))
    suggestion = db.Column(db.Text)
    
    def __repr__(self):
        return f'<ShoppingSuggestion {self.item_name}>'


class ClothingAnalysis(db.Model):
    """Model for storing clothing analysis data from the client application"""
    __tablename__ = 'clothing_analysis'
    
    id = db.Column(db.Integer, primary_key=True)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    image_name = db.Column(db.String(255), nullable=False)
    analysis_data = db.Column(db.Text, nullable=False)  # JSON string containing clothing items and attributes
    skin_analysis = db.Column(db.Text, nullable=False)  # JSON string containing skin tone and undertone
    recommendations = db.Column(db.Text, nullable=False)  # JSON string containing outfit and color recommendations
    
    def __repr__(self):
        return f'<ClothingAnalysis {self.id}>'
    
    def get_analysis_data(self):
        """Parse and return the analysis data as a dictionary"""
        return json.loads(self.analysis_data)
    
    def get_skin_analysis(self):
        """Parse and return the skin analysis as a dictionary"""
        return json.loads(self.skin_analysis)
    
    def get_recommendations(self):
        """Parse and return the recommendations as a dictionary"""
        return json.loads(self.recommendations) 