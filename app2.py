app2.py:

from flask import Flask, render_template, request, redirect, url_for, session, send_from_directory
from pymongo import MongoClient
from werkzeug.security import generate_password_hash, check_password_hash
from flask_session import Session
import os

app = Flask(__name__)
app.secret_key = os.urandom(24)

# MongoDB setup
uri = "mongodb+srv://LocktAdmin:L0cktForever@locktcluster.dfui7.mongodb.net/?retryWrites=true&w=majority&appName=LocktC>client = MongoClient(uri)
db = client.LocktDatabase
users_collection = db.users

# Session config
app.config['SESSION_TYPE'] = 'mongodb'
app.config['SESSION_MONGODB'] = client
app.config['SESSION_MONGODB_DB'] = 'LocktDatabase'
app.config['SESSION_MONGODB_COLLECT'] = 'sessions'
app.config['SESSION_PERMANENT'] = False
app.config['SESSION_USE_SIGNER'] = True
Session(app)

@app.route('/')
def home():
 return render_template('index.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form['username']
        password = generate_password_hash(request.form['password'])
        email = request.form['email']

        if users_collection.find_one({'username': username}):
            return "Username already taken.", 400

        users_collection.insert_one({
            'username': username,
            'password': password,
            'email': email,
            'serial_number': '',
            'sub_users': []
        })
        return redirect(url_for('login'))
    return render_template('signup.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
 if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        user = users_collection.find_one({'username': username})
        if user and check_password_hash(user['password'], password):
            session['username'] = username
            return redirect(url_for('profile', username=username))
        return 'Invalid credentials', 401
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.pop('username', None)
    return redirect(url_for('login'))

@app.route('/profile/<username>', methods=['GET', 'POST'])
def profile(username):
    if 'username' not in session or session['username'] != username:
        return redirect(url_for('login'))

    user = users_collection.find_one({'username': username})
    if not user:
        return 'User not found', 404

    if request.method == 'POST':
        serial = request.form.get('serial_number')
  if serial:
            users_collection.update_one({'username': username}, {'$set': {'serial_number': serial}})

        if 'sub_user_name' in request.form:
            new_sub = {'name': request.form['sub_user_name'], 'photos': [], 'voice_sample': ''}
            users_collection.update_one({'username': username}, {'$push': {'sub_users': new_sub}})

    user = users_collection.find_one({'username': username})
    return render_template('profile.html', user=user)

@app.route('/upload_photo/<username>/<sub_user_name>', methods=['POST'])
def upload_photo(username, sub_user_name):
    if 'username' not in session or session['username'] != username:
        return redirect(url_for('login'))

    photo = request.files.get('photo')
    if photo:
        filename = f"{username}_{sub_user_name}_{photo.filename}"
        filepath = os.path.join('static/uploads', filename)
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        photo.save(filepath)
        users_collection.update_one(
            {'username': username, 'sub_users.name': sub_user_name},
            {'$push': {'sub_users.$.photos': filename}}
        )
    return redirect(url_for('profile', username=username))
@app.route('/delete_photo/<username>/<sub_user_name>/<photo_filename>', methods=['POST'])
def delete_photo(username, sub_user_name, photo_filename):
    if 'username' not in session or session['username'] != username:
        return redirect(url_for('login'))

    filepath = os.path.join('static/uploads', photo_filename)
    if os.path.exists(filepath):
        os.remove(filepath)

    users_collection.update_one(
        {'username': username, 'sub_users.name': sub_user_name},
        {'$pull': {'sub_users.$.photos': photo_filename}}
    )
    return redirect(url_for('profile', username=username))

@app.route('/upload_voice/<username>/<sub_user_name>', methods=['POST'])
def upload_voice(username, sub_user_name):
    if 'username' not in session or session['username'] != username:
        return 'Unauthorized', 403

    voice = request.files.get('voice_sample')
    if voice:
        filename = f"{username}_{sub_user_name}.mp3"
        filepath = os.path.join('static/voice', filename)
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        voice.save(filepath)
        users_collection.update_one(
            {'username': username, 'sub_users.name': sub_user_name},
            {'$set': {'sub_users.$.voice_sample': filename}}
        )
    return 'Success', 200

@app.route('/delete_voice/<username>/<sub_user_name>', methods=['POST'])
def delete_voice(username, sub_user_name):
    if 'username' not in session or session['username'] != username:
        return redirect(url_for('login'))

    user = users_collection.find_one({'username': username})
    if user:
        sub_users = user.get('sub_users', [])
        for sub_user in sub_users:
            if sub_user['name'] == sub_user_name:
                filename = sub_user.get('voice_sample')
                if filename:
                    filepath = os.path.join('static/voice', filename)
                    if os.path.exists(filepath):
                        os.remove(filepath)
                users_collection.update_one(
                    {'username': username, 'sub_users.name': sub_user_name},
                    {'$set': {'sub_users.$.voice_sample': ''}}
                )
                break
    return redirect(url_for('profile', username=username))

@app.route('/install')
def install():
    return render_template('install.html')

if __name__ == '__main__':
    app.run(debug=True)
