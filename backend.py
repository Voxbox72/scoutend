import os
from flask import Flask, request, jsonify, session
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required
from flask_cors import CORS
from werkzeug.security import generate_password_hash, check_password_hash
from pymongo import MongoClient
from bson.objectid import ObjectId

app = Flask(__name__)
app.secret_key = os.getenv('SECRET_KEY', 'your-secret-key')
app.config['SESSION_COOKIE_SECURE'] = True
app.config['PERMANENT_SESSION_LIFETIME'] = 604800  # 7 days
CORS(app)
login_manager = LoginManager()
login_manager.init_app(app)

# Connect to MongoDB Atlas
client = MongoClient(os.getenv('MONGODB_URI'))
db = client.scouting_db

class User(UserMixin):
    def __init__(self, id, email):
        self.id = id
        self.email = email

@login_manager.user_loader
def load_user(user_id):
    user = db.users.find_one({"_id": ObjectId(user_id)})
    if user:
        return User(str(user['_id']), user['email'])
    return None

@app.route('/api/register', methods=['POST'])
def register():
    data = request.get_json()
    email = data.get('email')
    password = data.get('password')
    if not email or not password:
        return jsonify({'error': 'Email and password are required'}), 400
    hashed_password = generate_password_hash(password)
    try:
        result = db.users.insert_one({"email": email, "password": hashed_password})
        return jsonify({'message': 'Registered successfully', 'user': {'email': email}}), 201
    except Exception as e:
        return jsonify({'error': 'Email already exists'}), 400

@app.route('/api/login', methods=['POST'])
def login():
    data = request.get_json()
    email = data.get('email')
    password = data.get('password')
    user = db.users.find_one({"email": email})
    if user and check_password_hash(user['password'], password):
        user_obj = User(str(user['_id']), user['email'])
        login_user(user_obj, remember=True)
        return jsonify({'message': 'Logged in successfully', 'user': {'email': user['email']}})
    return jsonify({'error': 'Invalid credentials'}), 401

@app.route('/api/logout', methods=['POST'])
@login_required
def logout():
    logout_user()
    return jsonify({'message': 'Logged out successfully'})

@app.route('/api/check_access', methods=['GET'])
def check_access():
    user_id = session.get('user_id')
    if not user_id:
        return jsonify({'authenticated': False, 'subscribed': False}), 401
    subscription = db.subscriptions.find_one({"user_id": user_id})
    subscribed = subscription['active'] if subscription else False
    return jsonify({'authenticated': True, 'subscribed': subscribed})

@app.route('/api/subscription/activate', methods=['POST'])
@login_required
def activate_subscription():
    user_id = session.get('user_id')
    db.subscriptions.update_one(
        {"user_id": user_id},
        {"$set": {"user_id": user_id, "active": True}},
        upsert=True
    )
    return jsonify({'message': 'Subscription activated'})

if __name__ == '__main__':
    port = int(os.getenv('PORT', 5000))  # Use PORT env var, default to 5000 for local testing
    app.run(debug=True, host='0.0.0.0', port=port)