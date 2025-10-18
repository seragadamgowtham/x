import os
import sqlite3
import time
from datetime import datetime, date, timedelta
import random
import base64
import cv2
import numpy as np
import requests
from flask import Flask, Response, g, jsonify, render_template, request

# --- Flask App Initialization ---
app = Flask(__name__, template_folder='.')
app.secret_key = 'your_super_secret_key' 

# --- System Configuration ---
DB_FILE = "bus_management_system.db"
TRAINER_FILE = "face_trainer.yml"
HAAR_CASCADE_URL = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml"
HAAR_CASCADE_FILENAME = "haarcascade_frontalface_default.xml"
ENROLL_FACE_SAMPLES = 30
RECOGNITION_CONFIDENCE_THRESHOLD = 85
BUS_CAPACITY = 50
GROUP_1_LOCATIONS = [
    "Vepagunta", "Naidu Thota", "Gopalapatnam (GPT)", "Akkayyapalem", "NAD Junction (NAD X Road)",
    "Baji Junction / Gajuwaka Road junction", "Satyam Junction / Gurudwara Junction", "Hanumanthawaka Junction",
    "Madhurawada", "Tenneti Park", "Jodugullapalem", "Kailasagiri / Hill Point", "Sagar Nagar", "Yendada / Gayatri"
]
GROUP_1_BUSES = [1, 2, 3]
GROUP_2_LOCATIONS = [
    "Gajuwaka Bus Station", "Kurmannapalem", "Autonagar", "Lankelapalem Junction", "Steel Plant Arch",
    "Sheela Nagar", "Duvvada", "Kanithi Road", "Kancharapalem", "Railway Station", "RTC Complex (Main City Bus Terminal)",
    "Asilmetta Junction", "Maddilapalem Junction", "Venkojipalem", "Lawson's Bay Colony", "MVP Colony", "Sivajipalem",
    "Adavivaram Junction", "PM Palem (Pothinamallayya Palem)", "Kommadi Junction", "Yendada Junction",
    "Yendada Layout / Hill View Layout", "Gitam Junction", "Gitam Main Campus Stop (Rushikonda)", "Rushikonda Beach Stop"
]
GROUP_2_BUSES = [4, 5, 6]
ALL_LOCATIONS = sorted(list(set(GROUP_1_LOCATIONS + GROUP_2_LOCATIONS)))

# --- Global Variables for Face Recognition ---
face_detector = None
recognizer = cv2.face.LBPHFaceRecognizer_create()
camera = None

# --- Database Helper Functions ---
def get_db():
    db = getattr(g, '_database', None)
    if db is None:
        db = g._database = sqlite3.connect(DB_FILE, check_same_thread=False)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA foreign_keys = ON;")
    return db

@app.teardown_appcontext
def close_connection(exception):
    db = getattr(g, '_database', None)
    if db is not None:
        db.close()

# --- Core Logic Functions ---
def setup_dependencies():
    global face_detector
    if not os.path.exists(HAAR_CASCADE_FILENAME):
        print(f"Downloading model ({HAAR_CASCADE_FILENAME})...")
        try:
            r = requests.get(HAAR_CASCADE_URL, allow_redirects=True, timeout=10)
            r.raise_for_status()
            with open(HAAR_CASCADE_FILENAME, 'wb') as f: f.write(r.content)
            print("Download complete.")
        except requests.RequestException as e:
            print(f"FATAL ERROR: Could not download model: {e}")
            exit()
    face_detector = cv2.CascadeClassifier(HAAR_CASCADE_FILENAME)

def train_model():
    with app.app_context():
        db = get_db()
        cursor = db.cursor()
        cursor.execute("SELECT person_sno, image_data FROM face_samples")
        all_samples = cursor.fetchall()

        if not all_samples:
            if os.path.exists(TRAINER_FILE): os.remove(TRAINER_FILE)
            return False

        faces, ids = [], []
        for row in all_samples:
            nparr = np.frombuffer(row['image_data'], np.uint8)
            face_img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
            if face_img is not None:
                faces.append(face_img)
                ids.append(row['person_sno'])

        if not faces: return False
        
        recognizer.train(faces, np.array(ids))
        recognizer.write(TRAINER_FILE)
        return True

def allocate_bus_seat(person_sno, location):
    bus_codes = GROUP_1_BUSES if location in GROUP_1_LOCATIONS else GROUP_2_BUSES if location in GROUP_2_LOCATIONS else []
    if not bus_codes: return None, "Location not on any bus route."

    db = get_db()
    cursor = db.cursor()
    random.shuffle(bus_codes)

    for bus in bus_codes:
        cursor.execute("SELECT seat_number FROM bus_seats WHERE bus_code = ?", (bus,))
        taken_seats = {row['seat_number'] for row in cursor.fetchall()}
        if len(taken_seats) < BUS_CAPACITY:
            available_seats = list(set(range(1, BUS_CAPACITY + 1)) - taken_seats)
            seat_number = random.choice(available_seats)
            cursor.execute("INSERT INTO bus_seats (person_sno, bus_code, seat_number) VALUES (?, ?, ?)", (person_sno, bus, seat_number))
            db.commit()
            return f"BUS-{bus}-{seat_number}", None
    
    return None, "All buses for your route are full."

def has_attended_today(person_sno):
    today_str = date.today().strftime("%Y-%m-%d")
    cursor = get_db().cursor()
    cursor.execute("SELECT 1 FROM attendance WHERE person_sno = ? AND date(timestamp) = ?", (person_sno, today_str))
    return cursor.fetchone() is not None

# --- Video Streaming and Recognition ---
def generate_frames(mode='attendance'):
    global camera
    if camera is None: camera = cv2.VideoCapture(0)
    
    known_personnel = {}
    if os.path.exists(TRAINER_FILE) and mode == 'attendance':
        recognizer.read(TRAINER_FILE)
        with app.app_context():
            cursor = get_db().cursor()
            cursor.execute("SELECT sno, name FROM persons")
            known_personnel = {row['sno']: row['name'] for row in cursor.fetchall()}

    while True:
        success, frame = camera.read()
        if not success: continue
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_detector.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(100, 100))

        for (x, y, w, h) in faces:
            label, color = "Scanning...", (0, 123, 255) # Blue for scanning
            if mode == 'attendance' and known_personnel:
                sno, confidence = recognizer.predict(gray[y:y+h, x:x+w])
                if confidence < RECOGNITION_CONFIDENCE_THRESHOLD:
                    name = known_personnel.get(sno, "Error")
                    label = f"Marked, {name}" if has_attended_today(sno) else f"Welcome, {name}"
                    color = (255, 193, 7) if has_attended_today(sno) else (40, 167, 69) # Yellowish / Green
            
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/video_feed/<mode>')
def video_feed(mode):
    return Response(generate_frames(mode), mimetype='multipart/x-mixed-replace; boundary=frame')

# --- Single Page Route ---
@app.route('/')
def index():
    return render_template('index_simple.html', locations=ALL_LOCATIONS)

# --- API Endpoints ---
@app.route('/api/submit_enrollment', methods=['POST'])
def submit_enrollment():
    data = request.json
    db = get_db()
    cursor = db.cursor()
    try:
        cursor.execute("INSERT INTO persons (unique_id, name, role, phone_number) VALUES (?, ?, ?, ?)",
                       (data['unique_id'], data['name'], data['role'], data['phone']))
        person_sno = cursor.lastrowid

        seat_info = "N/A"
        if data['role'] == 'passenger':
            cursor.execute("INSERT INTO passenger_details (person_sno, location, previous_year_score, department) VALUES (?, ?, ?, ?)",
                           (person_sno, data['location'], data['score'], 'N/A'))
            seat_info, error = allocate_bus_seat(person_sno, data['location'])
            if error: raise Exception(error)
        elif data['role'] == 'driver':
            cursor.execute("INSERT INTO driver_details (person_sno, license_number) VALUES (?, ?)",
                           (person_sno, data['license']))

        # Store face samples if provided
        if 'samples' in data:
            for img_data_b64 in data['samples']:
                img_data = base64.b64decode(img_data_b64.split(',')[1])
                cursor.execute("INSERT INTO face_samples (person_sno, image_data) VALUES (?, ?)", (person_sno, img_data))

        db.commit()
        train_model()  # Retrain the face recognition model
        return jsonify(success=True, message=f"Enrollment successful! Seat: {seat_info}.")
    except sqlite3.IntegrityError:
        db.rollback()
        return jsonify(success=False, message=f"The ID '{data['unique_id']}' already exists.")
    except Exception as e:
        db.rollback()
        return jsonify(success=False, message=str(e))

@app.route('/api/personnel', methods=['GET'])
def get_personnel():
    personnel = get_db().execute("SELECT unique_id, name, role, phone_number FROM persons ORDER BY role, name").fetchall()
    return jsonify([dict(p) for p in personnel])

@app.route('/api/seats', methods=['GET'])
def get_seats():
    seats = get_db().execute('''
        SELECT p.unique_id, p.name, b.bus_code, b.seat_number FROM persons p 
        JOIN bus_seats b ON p.sno = b.person_sno ORDER BY b.bus_code, b.seat_number
    ''').fetchall()
    total_capacity = BUS_CAPACITY * len(set(GROUP_1_BUSES + GROUP_2_BUSES))
    return jsonify(data=[dict(s) for s in seats], count=len(seats), capacity=total_capacity)

@app.route('/api/attendance_log', methods=['GET'])
def get_attendance_log():
    records = get_db().execute('''
        SELECT p.unique_id, p.name, p.role, a.timestamp FROM attendance a
        JOIN persons p ON a.person_sno = p.sno ORDER BY a.timestamp DESC
    ''').fetchall()
    return jsonify([dict(r) for r in records])

@app.route('/api/percentage', methods=['GET'])
def get_percentage():
    db = get_db()
    first_date_row = db.execute("SELECT MIN(date(timestamp)) as min_date FROM attendance").fetchone()
    if not first_date_row or not first_date_row['min_date']:
        return jsonify(records=[], working_days=0, start_date='N/A', end_date='N/A')

    start_date_obj = datetime.strptime(first_date_row['min_date'], "%Y-%m-%d").date()
    end_date_obj = date.today()
    total_days = sum(1 for d in (start_date_obj + timedelta(n) for n in range((end_date_obj - start_date_obj).days + 1)) if d.weekday() < 6)

    passengers = db.execute("SELECT sno, unique_id, name FROM persons WHERE role = 'passenger'").fetchall()
    records = []
    for p in passengers:
        present_days = db.execute("SELECT COUNT(DISTINCT date(timestamp)) as count FROM attendance WHERE person_sno = ?", (p['sno'],)).fetchone()['count']
        percentage = (present_days / total_days * 100) if total_days > 0 else 0
        records.append({'uid': p['unique_id'], 'name': p['name'], 'present': present_days, 'percentage': percentage})
    
    return jsonify(records=records, working_days=total_days, start_date=str(start_date_obj), end_date=str(end_date_obj))

@app.route('/api/remove_person', methods=['POST'])
def remove_person():
    unique_id = request.json['unique_id']
    db = get_db()
    person = db.execute("SELECT sno, name FROM persons WHERE unique_id = ?", (unique_id,)).fetchone()
    if person:
        db.execute("DELETE FROM persons WHERE sno = ?", (person['sno'],))
        db.commit()
        return jsonify(success=True, message=f"Removed {person['name']}.")
    else:
        return jsonify(success=False, message="Person not found.")

@app.route('/api/get_person_details', methods=['GET'])
def get_person_details():
    unique_id = request.args.get('unique_id')
    if not unique_id:
        return jsonify(success=False, message="Unique ID required")
    
    db = get_db()
    person = db.execute("SELECT * FROM persons WHERE unique_id = ?", (unique_id,)).fetchone()
    if not person:
        return jsonify(success=False, message="Person not found")
    
    person_data = dict(person)
    
    # Get role-specific details
    if person['role'] == 'passenger':
        passenger_details = db.execute("SELECT * FROM passenger_details WHERE person_sno = ?", (person['sno'],)).fetchone()
        if passenger_details:
            person_data.update(dict(passenger_details))
        
        # Get seat allocation
        seat = db.execute("SELECT bus_code, seat_number FROM bus_seats WHERE person_sno = ?", (person['sno'],)).fetchone()
        if seat:
            person_data['seat_info'] = f"BUS-{seat['bus_code']}-{seat['seat_number']}"
    elif person['role'] == 'driver':
        driver_details = db.execute("SELECT * FROM driver_details WHERE person_sno = ?", (person['sno'],)).fetchone()
        if driver_details:
            person_data.update(dict(driver_details))
    
    # Get attendance records
    attendance_records = db.execute("SELECT timestamp FROM attendance WHERE person_sno = ? ORDER BY timestamp DESC LIMIT 10", (person['sno'],)).fetchall()
    person_data['recent_attendance'] = [dict(record) for record in attendance_records]
    
    return jsonify(success=True, data=person_data)

@app.route('/api/update_person', methods=['POST'])
def update_person():
    data = request.json
    unique_id = data.get('unique_id')
    
    if not unique_id:
        return jsonify(success=False, message="Unique ID required")
    
    db = get_db()
    try:
        # Update basic person info
        db.execute("UPDATE persons SET name = ?, phone_number = ? WHERE unique_id = ?", 
                  (data['name'], data['phone_number'], unique_id))
        
        person = db.execute("SELECT sno FROM persons WHERE unique_id = ?", (unique_id,)).fetchone()
        person_sno = person['sno']
        
        # Update role-specific details
        if data['role'] == 'passenger':
            db.execute("UPDATE passenger_details SET location = ?, previous_year_score = ? WHERE person_sno = ?",
                      (data['location'], data['previous_year_score'], person_sno))
        elif data['role'] == 'driver':
            db.execute("UPDATE driver_details SET license_number = ? WHERE person_sno = ?",
                      (data['license_number'], person_sno))
        
        db.commit()
        return jsonify(success=True, message="Person updated successfully")
    except Exception as e:
        db.rollback()
        return jsonify(success=False, message=str(e))

@app.route('/api/mark_attendance', methods=['POST'])
def mark_attendance():
    data = request.json
    unique_id = data.get('unique_id')
    role = data.get('role', 'passenger')
    
    if not unique_id:
        return jsonify(success=False, message="Unique ID required")
    
    db = get_db()
    try:
        # Check if person exists
        person = db.execute("SELECT sno, name, role FROM persons WHERE unique_id = ?", (unique_id,)).fetchone()
        if not person:
            return jsonify(success=False, message="Person not found")
        
        # Check role match
        if person['role'] != role:
            return jsonify(success=False, message=f"Role mismatch. You are a {person['role']}.")
        
        # Check if already attended today
        if has_attended_today(person['sno']):
            return jsonify(success=False, message="Attendance already marked today.")
        
        # Mark attendance
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        db.execute("INSERT INTO attendance (person_sno, timestamp) VALUES (?, ?)", (person['sno'], current_time))
        db.commit()
        
        return jsonify(success=True, message=f"Welcome, {person['name']}! Attendance marked successfully.")
        
    except Exception as e:
        db.rollback()
        return jsonify(success=False, message=str(e))

@app.route('/api/mark_attendance_face', methods=['POST'])
def mark_attendance_face():
    """Face recognition based attendance marking"""
    data = request.json
    role = data.get('role', 'passenger')
    
    if 'image' not in data:
        return jsonify(success=False, message="No image provided")
    
    try:
        # Decode and process the image
        img_data = base64.b64decode(data['image'].split(',')[1])
        nparr = np.frombuffer(img_data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = face_detector.detectMultiScale(gray, 1.2, 5, minSize=(100, 100))
        if len(faces) == 0:
            return jsonify(success=False, message="No face detected. Please ensure your face is visible.")
        
        # Use the first detected face
        (x, y, w, h) = faces[0]
        
        # Load the trained model
        if not os.path.exists(TRAINER_FILE):
            return jsonify(success=False, message="No trained faces found. Please enroll first.")
        
        recognizer.read(TRAINER_FILE)
        sno, confidence = recognizer.predict(gray[y:y+h, x:x+w])
        
        if confidence >= RECOGNITION_CONFIDENCE_THRESHOLD:
            return jsonify(success=False, message="Face not recognized. Please ensure you are enrolled.")
        
        # Check if person exists and role matches
        db = get_db()
        person = db.execute("SELECT sno, name, role FROM persons WHERE sno = ?", (sno,)).fetchone()
        if not person:
            return jsonify(success=False, message="Person not found in database.")
        
        if person['role'] != role:
            return jsonify(success=False, message=f"Role mismatch. You are a {person['role']}.")
        
        # Check if already attended today
        if has_attended_today(person['sno']):
            return jsonify(success=False, message="Attendance already marked today.")
        
        # Mark attendance
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        db.execute("INSERT INTO attendance (person_sno, timestamp) VALUES (?, ?)", (person['sno'], current_time))
        db.commit()
        
        return jsonify(success=True, message=f"Welcome, {person['name']}! Attendance marked successfully.")
        
    except Exception as e:
        return jsonify(success=False, message=f"Error processing face: {str(e)}")

@app.route('/api/manual_attendance', methods=['POST'])
def manual_attendance():
    """Manual attendance marking for owners"""
    data = request.json
    unique_id = data.get('unique_id')
    
    if not unique_id:
        return jsonify(success=False, message="Unique ID required")
    
    db = get_db()
    try:
        # Check if person exists
        person = db.execute("SELECT sno, name, role FROM persons WHERE unique_id = ?", (unique_id,)).fetchone()
        if not person:
            return jsonify(success=False, message="Person not found")
        
        # Check if already attended today
        if has_attended_today(person['sno']):
            return jsonify(success=False, message="Attendance already marked today.")
        
        # Mark attendance
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        db.execute("INSERT INTO attendance (person_sno, timestamp) VALUES (?, ?)", (person['sno'], current_time))
        db.commit()
        
        return jsonify(success=True, message=f"Attendance marked for {person['name']}.")
        
    except Exception as e:
        db.rollback()
        return jsonify(success=False, message=str(e))

# --- Initial Setup ---
def init_db_on_startup():
    print("Checking and initializing database...")
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("PRAGMA foreign_keys = ON;")
    
    # Create tables with updated schema
    cursor.execute('''CREATE TABLE IF NOT EXISTS persons (sno INTEGER PRIMARY KEY, unique_id TEXT UNIQUE NOT NULL, name TEXT NOT NULL, role TEXT NOT NULL CHECK(role IN ('passenger', 'driver')), phone_number TEXT NOT NULL)''')
    cursor.execute('''CREATE TABLE IF NOT EXISTS driver_details (person_sno INTEGER PRIMARY KEY, license_number TEXT NOT NULL, FOREIGN KEY(person_sno) REFERENCES persons(sno) ON DELETE CASCADE)''')
    cursor.execute('''CREATE TABLE IF NOT EXISTS passenger_details (person_sno INTEGER PRIMARY KEY, location TEXT NOT NULL, previous_year_score TEXT, department TEXT NOT NULL, FOREIGN KEY(person_sno) REFERENCES persons(sno) ON DELETE CASCADE)''')
    cursor.execute('''CREATE TABLE IF NOT EXISTS face_samples (id INTEGER PRIMARY KEY, person_sno INTEGER NOT NULL, image_data BLOB NOT NULL, FOREIGN KEY(person_sno) REFERENCES persons(sno) ON DELETE CASCADE)''')
    cursor.execute('''CREATE TABLE IF NOT EXISTS bus_seats (person_sno INTEGER PRIMARY KEY, bus_code INTEGER NOT NULL, seat_number INTEGER NOT NULL, FOREIGN KEY(person_sno) REFERENCES persons(sno) ON DELETE CASCADE, UNIQUE(bus_code, seat_number))''')
    cursor.execute('''CREATE TABLE IF NOT EXISTS attendance (id INTEGER PRIMARY KEY, person_sno INTEGER NOT NULL, timestamp TEXT NOT NULL, FOREIGN KEY (person_sno) REFERENCES persons (sno) ON DELETE CASCADE)''')
    
    # Check if phone_number column exists, if not add it (for existing databases)
    cursor.execute("PRAGMA table_info(persons)")
    schema = cursor.fetchall()
    columns = [col[1] for col in schema]
    if 'phone_number' not in columns:
        print("Adding phone_number column to existing database...")
        cursor.execute("ALTER TABLE persons ADD COLUMN phone_number TEXT")
        cursor.execute("UPDATE persons SET phone_number = 'N/A' WHERE phone_number IS NULL")
        print("Database migration completed.")
    
    conn.commit()
    conn.close()

if __name__ == '__main__':
    init_db_on_startup()
    setup_dependencies()
    if os.path.exists(TRAINER_FILE): 
        recognizer.read(TRAINER_FILE)
    app.run(debug=True, threaded=True)
