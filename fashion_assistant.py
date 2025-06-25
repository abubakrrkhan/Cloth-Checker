import random
import re
import traceback
from datetime import datetime

class FashionAssistant:
    def __init__(self):
        """Initialize the fashion assistant chatbot"""
        try:
            # Define assistant properties
            self.name = "Vogue"
            self.personality = "helpful, knowledgeable, and stylish"
            
            # Define conversation contexts
            self.contexts = {
                'greeting': {
                    'patterns': [
                        r'hello', r'hi', r'hey', r'greetings', r'howdy',
                        r'what\'s up', r'good morning', r'good afternoon', r'good evening'
                    ],
                    'responses': [
                        "Hello! I'm {assistant_name}, your AI fashion assistant. How can I help with your style today?",
                        "Hi there! I'm {assistant_name}, ready to assist with fashion advice. What can I help you with?",
                        "Hey! I'm {assistant_name}, your personal style guide. How can I assist you today?",
                        "Welcome! I'm {assistant_name}, and I'm here to help with fashion recommendations. What would you like to know?"
                    ]
                },
                'clothing_advice': {
                    'patterns': [
                        r'what (should|can) I wear', r'recommend (some|an) (outfit|clothing)',
                        r'looking for (style|fashion) (advice|tips)', r'suggest (some|an) (clothes|outfit)',
                        r'help (me|with) (my|) (style|outfit|clothes)'
                    ],
                    'responses': [
                        "I'd be happy to suggest an outfit! Could you tell me what occasion you're dressing for?",
                        "I can definitely help with clothing recommendations. What's your style preference? Casual, business, formal, or athleisure?",
                        "Let me help you find the perfect outfit. What colors do you typically like to wear?",
                        "I'd love to suggest some clothes for you. Could you describe what you're looking for or the event you're attending?"
                    ],
                    'follow_up': 'ask_occasion'
                },
                'color_advice': {
                    'patterns': [
                        r'what colors (should|look good|work)', r'color (advice|recommendation)',
                        r'which (color|colors) (should|would|do)', r'suggest (some|) colors'
                    ],
                    'responses': [
                        "Colors are so important for completing a look! Based on your analysis, {colors_recommendation}",
                        "For your complexion, {colors_recommendation}",
                        "I'd recommend these colors that would enhance your natural features: {colors_recommendation}",
                        "Based on color theory and your skin tone, {colors_recommendation}"
                    ]
                },
                'pattern_advice': {
                    'patterns': [
                        r'(what|which) pattern', r'pattern (advice|recommendation)',
                        r'suggest (some|) patterns', r'(good|best) patterns'
                    ],
                    'responses': [
                        "Patterns can add visual interest to your outfit! {patterns_recommendation}",
                        "When it comes to patterns, {patterns_recommendation}",
                        "For your style, {patterns_recommendation}",
                        "I'd suggest these patterns: {patterns_recommendation}"
                    ]
                },
                'outfit_feedback': {
                    'patterns': [
                        r'how (does|do|is) (this|my|the) (outfit|look|style)',
                        r'what (do you|) think (of|about) (this|my|the) (outfit|look|style)',
                        r'give (me|) feedback', r'rate (my|this|the) (outfit|look|style)'
                    ],
                    'responses': [
                        "Based on my analysis of your outfit, {outfit_feedback}",
                        "Looking at your current outfit, {outfit_feedback}",
                        "From what I can see, {outfit_feedback}",
                        "My fashion assessment is that {outfit_feedback}"
                    ]
                },
                'seasonal_advice': {
                    'patterns': [
                        r'what (to|should I) wear (in|for|during) (spring|summer|fall|winter|autumn)',
                        r'(spring|summer|fall|winter|autumn) (outfit|clothing|fashion) (advice|tips|recommendations)',
                        r'(clothes|outfits) for (spring|summer|fall|winter|autumn)'
                    ],
                    'responses': [
                        "For {season}, I recommend {seasonal_recommendation}",
                        "{season} is a great time to wear {seasonal_recommendation}",
                        "During {season}, you might want to try {seasonal_recommendation}",
                        "Here's what works well for {season}: {seasonal_recommendation}"
                    ]
                },
                'trend_inquiry': {
                    'patterns': [
                        r'what\'s (trending|in style|fashionable|popular) (now|this season|this year)',
                        r'current (trends|styles|fashion)',
                        r'(latest|new) (fashion|style|clothing) (trends|styles)'
                    ],
                    'responses': [
                        "The current fashion trends include {trends_info}",
                        "Right now, fashion-forward individuals are wearing {trends_info}",
                        "This season's hottest trends are {trends_info}",
                        "The latest styles that are turning heads include {trends_info}"
                    ]
                },
                'confused': {
                    'patterns': [],  # This is a fallback context
                    'responses': [
                        "I'm not quite sure I understand. Could you tell me more about what you're looking for in terms of fashion advice?",
                        "I'd love to help with your fashion query, but I need a bit more information. Could you elaborate?",
                        "I want to give you the best fashion advice, but I'm not quite sure what you're asking. Could you rephrase that?",
                        "Hmm, I'm not certain what fashion guidance you need. Could you provide more details about what you're looking for?"
                    ]
                }
            }
            
            # Define fashion trends (would be updated regularly in a live system)
            current_year = datetime.now().year
            current_month = datetime.now().month
            
            if 3 <= current_month <= 5:  # Spring
                self.current_season = "Spring"
            elif 6 <= current_month <= 8:  # Summer
                self.current_season = "Summer"
            elif 9 <= current_month <= 11:  # Fall
                self.current_season = "Fall"
            else:  # Winter
                self.current_season = "Winter"
            
            self.current_trends = {
                'Spring': [
                    "pastel colors", "light denim", "floral patterns", 
                    "oversized blazers", "statement sleeves", "crochet items"
                ],
                'Summer': [
                    "bright colors", "linen fabrics", "crop tops", 
                    "wide-leg pants", "bucket hats", "chunky sandals"
                ],
                'Fall': [
                    "layered looks", "earth tones", "leather jackets", 
                    "chunky knits", "plaid patterns", "combat boots"
                ],
                'Winter': [
                    "monochromatic outfits", "faux fur", "oversized coats", 
                    "turtlenecks", "velvet fabrics", "knee-high boots"
                ]
            }
            
            # Occasion-based recommendations
            self.occasion_outfits = {
                'casual': [
                    "A well-fitted pair of jeans paired with a casual button-up shirt and clean sneakers",
                    "A comfortable t-shirt with chino shorts and canvas shoes for a relaxed look",
                    "A stylish athleisure outfit with technical fabrics and minimalist design",
                    "A casual dress with a denim jacket and ankle boots"
                ],
                'work': [
                    "A tailored blazer over a blouse or dress shirt with dress pants or a pencil skirt",
                    "A cotton button-up shirt with well-fitted chinos and leather shoes",
                    "A professional dress in a solid color with subtle accessories",
                    "A knit sweater over a collared shirt with tailored trousers and loafers"
                ],
                'date': [
                    "A fitted dress in a flattering color with minimal accessories",
                    "Dark jeans with a stylish top and a statement jacket",
                    "A well-fitted button-up shirt with chinos and leather shoes",
                    "A skirt or pants with a silk or satin top for an elegant touch"
                ],
                'formal': [
                    "A classic suit in navy or black with a crisp dress shirt and tie",
                    "A floor-length gown in a flattering silhouette",
                    "A tailored three-piece suit for a sophisticated look",
                    "A cocktail dress in an elegant fabric like silk or velvet"
                ]
            }
            
            # Pattern recommendations based on body type
            self.pattern_recommendations = {
                'tall': "Horizontal stripes can add width, and large patterns work well with your frame.",
                'petite': "Smaller patterns work best, and vertical stripes can create the illusion of height.",
                'curvy': "Medium-sized patterns that follow your natural shape can accentuate your curves beautifully.",
                'athletic': "Patterns with texture and dimension can add visual interest and soften a muscular frame.",
                'apple': "Vertical patterns can elongate your silhouette, especially when focusing on your lower half.",
                'pear': "Patterns on your upper body can balance your proportions nicely.",
                'hourglass': "Patterns that accentuate your waist can highlight your balanced proportions.",
                'rectangle': "Patterns that create curves, like florals or abstract designs, can add dimension to your shape."
            }
            
            # State variables
            self.conversation_state = {
                'context': None,
                'follow_up': None,
                'last_query': None,
                'user_preferences': {},
                'detected_clothing': [],
                'skin_tone': None,
                'undertone': None
            }
            
            print("Fashion assistant initialized successfully")
        except Exception as e:
            print(f"Error initializing fashion assistant: {str(e)}")
            print(traceback.format_exc())
    
    def process_message(self, message, clothing_items=None, skin_tone=None, undertone=None):
        """
        Process a user message and generate a response
        
        Args:
            message: User's text message
            clothing_items: Optional list of detected clothing items
            skin_tone: Optional skin tone information
            undertone: Optional undertone information
            
        Returns:
            Assistant's response as a string
        """
        try:
            # Update state with any provided information
            if clothing_items is not None:
                self.conversation_state['detected_clothing'] = clothing_items
            
            if skin_tone is not None:
                self.conversation_state['skin_tone'] = skin_tone
                
            if undertone is not None:
                self.conversation_state['undertone'] = undertone
            
            # Preprocess the message
            processed_message = message.lower().strip()
            self.conversation_state['last_query'] = processed_message
            
            # Check for follow-up contexts first
            if self.conversation_state['follow_up'] == 'ask_occasion':
                # Extract occasion information
                occasion = self.identify_occasion(processed_message)
                if occasion:
                    self.conversation_state['user_preferences']['occasion'] = occasion
                    self.conversation_state['follow_up'] = None
                    return self.get_occasion_recommendation(occasion)
            
            # Identify the message context
            context = self.identify_context(processed_message)
            self.conversation_state['context'] = context
            
            # Generate a response based on the context
            response = self.generate_response(context, processed_message)
            
            # Update any follow-up states
            if 'follow_up' in self.contexts.get(context, {}):
                self.conversation_state['follow_up'] = self.contexts[context]['follow_up']
            
            return response
        
        except Exception as e:
            print(f"Error processing message: {str(e)}")
            print(traceback.format_exc())
            return "I'm sorry, I encountered an issue processing your request. Could you try asking again?"
    
    def identify_context(self, message):
        """Identify the context of the user's message"""
        for context, data in self.contexts.items():
            if 'patterns' in data:
                for pattern in data['patterns']:
                    if re.search(pattern, message, re.IGNORECASE):
                        return context
        
        return 'confused'  # Default context if no patterns match
    
    def identify_occasion(self, message):
        """Identify the occasion mentioned in the user's message"""
        occasions = {
            'casual': ['casual', 'everyday', 'daily', 'weekend', 'hang out', 'hanging out', 'relaxed'],
            'work': ['work', 'office', 'business', 'professional', 'job', 'interview'],
            'date': ['date', 'dinner', 'romantic', 'evening out', 'night out'],
            'formal': ['formal', 'wedding', 'gala', 'black tie', 'ceremony', 'fancy']
        }
        
        for occasion, keywords in occasions.items():
            for keyword in keywords:
                if keyword in message.lower():
                    return occasion
        
        return 'casual'  # Default to casual if no specific occasion is identified
    
    def generate_response(self, context, message):
        """Generate a response based on the identified context"""
        if context not in self.contexts:
            context = 'confused'
        
        # Select a random response template from the context
        response_template = random.choice(self.contexts[context]['responses'])
        
        # Fill in the template with relevant information
        response = response_template.format(
            assistant_name=self.name,
            colors_recommendation=self.get_colors_recommendation(),
            patterns_recommendation=self.get_patterns_recommendation(),
            outfit_feedback=self.get_outfit_feedback(),
            seasonal_recommendation=self.get_seasonal_recommendation(),
            trends_info=self.get_trends_info(),
            season=self.current_season
        )
        
        return response
    
    def get_colors_recommendation(self):
        """Get color recommendations based on user's skin tone and undertone"""
        undertone = self.conversation_state.get('undertone')
        
        if undertone == 'Warm':
            return "colors with warm undertones like coral, peach, gold, amber, and olive green tend to work well. These enhance your natural warmth."
        elif undertone == 'Cool':
            return "colors with cool undertones like royal blue, emerald green, lavender, ruby, and silver complement your complexion beautifully."
        else:  # Neutral or unknown
            return "you can wear a wide range of colors. Jewel tones like teal, burgundy, and navy are particularly versatile and flattering for most people."
    
    def get_patterns_recommendation(self):
        """Get pattern recommendations based on user's body type"""
        # In a real implementation, this would use detected body type
        # For now, we'll randomly select a recommendation
        body_types = list(self.pattern_recommendations.keys())
        selected_body_type = random.choice(body_types)
        
        return self.pattern_recommendations[selected_body_type]
    
    def get_outfit_feedback(self):
        """Generate feedback on the user's outfit"""
        detected_clothing = self.conversation_state.get('detected_clothing', [])
        
        if not detected_clothing:
            return "I'd need to see your outfit to provide specific feedback. You can upload a photo for analysis."
        
        # In a real implementation, this would use sophisticated analysis
        # For demo purposes, we'll provide generic positive feedback
        clothing_types = [item.get('type', 'item') for item in detected_clothing]
        clothing_colors = [item.get('color_name', 'colored') for item in detected_clothing]
        
        feedback_templates = [
            "your {item1} pairs nicely with your {item2}. The {color1} and {color2} complement each other well.",
            "the {color1} {item1} is a great choice. It works well with your overall look.",
            "your outfit has a cohesive color palette with the {color1} and {color2} elements. Nice job!",
            "the {item1} has a flattering fit and the {color1} tone works well for you."
        ]
        
        selected_template = random.choice(feedback_templates)
        
        # Fill in the template with detected items
        if len(clothing_types) >= 2 and len(clothing_colors) >= 2:
            feedback = selected_template.format(
                item1=clothing_types[0].lower(),
                item2=clothing_types[1].lower(),
                color1=clothing_colors[0].lower(),
                color2=clothing_colors[1].lower()
            )
        elif len(clothing_types) >= 1 and len(clothing_colors) >= 1:
            feedback = selected_template.format(
                item1=clothing_types[0].lower(),
                item2=clothing_types[0].lower(),  # Use the same item as fallback
                color1=clothing_colors[0].lower(),
                color2=clothing_colors[0].lower()  # Use the same color as fallback
            )
        else:
            feedback = "your outfit has a balanced and appealing look overall."
        
        return feedback
    
    def get_seasonal_recommendation(self):
        """Get clothing recommendations based on the current season"""
        season = self.current_season
        
        if season in self.current_trends:
            trends = ", ".join(random.sample(self.current_trends[season], 3))
            return f"{trends}, which are perfect for the {season.lower()} weather."
        else:
            return "versatile layers that can adapt to changing temperatures."
    
    def get_trends_info(self):
        """Get information about current fashion trends"""
        season = self.current_season
        
        if season in self.current_trends:
            # Select a random subset of trends
            selected_trends = random.sample(self.current_trends[season], min(4, len(self.current_trends[season])))
            trends_text = ", ".join(selected_trends)
            return f"{trends_text}. These are popular choices for {season.lower()} fashion this year."
        else:
            return "sustainable fashion, gender-neutral designs, and vintage-inspired pieces, which are becoming increasingly popular year-round."
    
    def get_occasion_recommendation(self, occasion):
        """Get outfit recommendations for a specific occasion"""
        if occasion in self.occasion_outfits:
            outfit = random.choice(self.occasion_outfits[occasion])
            return f"For a {occasion} occasion, I would recommend: {outfit}. Would you like more specific suggestions?"
        else:
            return "For that occasion, I'd recommend an outfit that makes you feel confident and comfortable. Could you tell me more about your style preferences?" 