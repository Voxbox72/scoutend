import os
import pymongo.errors
from flask import Flask, request, jsonify, session
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required
from flask_cors import CORS
from werkzeug.security import generate_password_hash, check_password_hash
from pymongo import MongoClient
from bson.objectid import ObjectId
from tenacity import retry, stop_after_attempt, wait_fixed

app = Flask(__name__)
app.secret_key = os.getenv('SECRET_KEY', 'your-secret-key')
app.config['SESSION_COOKIE_SECURE'] = True
app.config['PERMANENT_SESSION_LIFETIME'] = 604800  # 7 days
CORS(app)
login_manager = LoginManager()
login_manager.init_app(app)

# Connect to MongoDB Atlas
try:
    client = MongoClient(
        os.getenv('MONGODB_URI'),
        serverSelectionTimeoutMS=30000,
        connectTimeoutMS=10000,
        socketTimeoutMS=30000
    )
    db = client.scouting_db
    # Test the connection
    db.command('ping')
    print("MongoDB connection successful")
except Exception as e:
    print(f"Failed to connect to MongoDB: {str(e)}")
    raise e

class User(UserMixin):
    def __init__(self, id, email):
        self.id = id
        self.email = email

@login_manager.user_loader
def load_user(user_id):
    try:
        user = db.users.find_one({"_id": ObjectId(user_id)})
        if user:
            print(f"Loaded user: {user}")
            return User(str(user['_id']), user['email'])
        print(f"User not found: {user_id}")
        return None
    except Exception as e:
        print(f"Error loading user {user_id}: {str(e)}")
        return None

@app.route('/api/register', methods=['POST'])
def register():
    try:
        data = request.get_json()
        email = data.get('email')
        password = data.get('password')
        if not email or not password:
            return jsonify({'error': 'Email and password are required'}), 400

        @retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
        def check_existing():
            return db.users.find_one({"email": email})

        existing_user = check_existing()
        if existing_user:
            return jsonify({'error': 'Email already exists'}), 400

        hashed_password = generate_password_hash(password)
        @retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
        def insert_user():
            return db.users.insert_one({"email": email, "password": hashed_password})

        result = insert_user()
        print(f"Registered user: {email}, Inserted ID: {result.inserted_id}")
        return jsonify({'message': 'Registered successfully', 'user': {'email': email}}), 201
    except pymongo.errors.DuplicateKeyError:
        return jsonify({'error': 'Email already exists'}), 400
    except pymongo.errors.ConnectionError as ce:
        print(f"MongoDB connection error: {str(ce)}")
        return jsonify({'error': 'Database connection failed'}), 500
    except Exception as e:
        print(f"Register error: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/api/login', methods=['POST'])
def login():
    try:
        data = request.get_json()
        email = data.get('email')
        password = data.get('password')
        if not email or not password:
            return jsonify({'error': 'Email and password are required'}), 400

        @retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
        def find_user():
            return db.users.find_one({"email": email})

        user = find_user()
        if not user:
            return jsonify({'error': 'Invalid credentials'}), 401

        if 'password' not in user:
            print(f"User data corrupted for {email}: {user}")
            return jsonify({'error': 'User data corrupted'}), 500

        if not check_password_hash(user['password'], password):
            return jsonify({'error': 'Invalid credentials'}), 401

        user_obj = User(str(user['_id']), user['email'])
        login_user(user_obj, remember=True)
        print(f"Logged in user: {email}")
        return jsonify({'message': 'Logged in successfully', 'user': {'email': user['email']}})
    except pymongo.errors.ConnectionError as ce:
        print(f"MongoDB connection error: {str(ce)}")
        return jsonify({'error': 'Database connection failed'}), 500
    except Exception as e:
        print(f"Login error: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/api/logout', methods=['POST'])
@login_required
def logout():
    logout_user()
    return jsonify({'message': 'Logged out successfully'})

@app.route('/api/check_access', methods=['GET'])
def check_access():
    try:
        user_id = session.get('user_id')
        if not user_id:
            return jsonify({'authenticated': False, 'subscribed': False}), 401
        subscription = db.subscriptions.find_one({"user_id": user_id})
        subscribed = subscription['active'] if subscription else False
        return jsonify({'authenticated': True, 'subscribed': subscribed})
    except Exception as e:
        print(f"Check access error: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/api/subscription/activate', methods=['POST'])
@login_required
def activate_subscription():
    try:
        user_id = session.get('user_id')
        db.subscriptions.update_one(
            {"user_id": user_id},
            {"$set": {"user_id": user_id, "active": True}},
            upsert=True
        )
        return jsonify({'message': 'Subscription activated'})
    except Exception as e:
        print(f"Activate subscription error: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    port = int(os.getenv('PORT', 5000))
    app.run(debug=True, host='0.0.0.0', port=port)