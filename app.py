# app.py

from flask import Flask, flash, render_template, Response, redirect, url_for, request, jsonify
import face_recognition
import cv2
import numpy as np
import csv
import datetime
import os

app = Flask(__name__)
app.secret_key = os.urandom(24)

# Directory to save the CSV files (attendance records)
attendance_dir = 'attendance_records'
photos_dir = 'photos'
if not os.path.exists(attendance_dir):
    os.makedirs(attendance_dir)
if not os.path.exists(photos_dir):
    os.makedirs(photos_dir)

# Global variables for attendance tracking
attendance_tracking = {
    "arrival": set(),  # Set to track arrivals
    "departure": set(),  # Set to track departures
    "messages": set()  # Set to store displayed messages in this session
}

def load_users():
    """Load users and their face encodings."""
    known_face_names = []
    known_face_encodings = []

    for file_name in os.listdir(photos_dir):
        if file_name.endswith('.jpg'):
            name = os.path.splitext(file_name)[0]
            file_path = os.path.join(photos_dir, file_name)
            image = face_recognition.load_image_file(file_path)
            encodings = face_recognition.face_encodings(image)
            if encodings:
                known_face_names.append(name)
                known_face_encodings.append(encodings[0])

    return known_face_names, known_face_encodings
                
# Get current date for the CSV file
current_date = datetime.datetime.now().strftime("%Y-%m-%d")

def load_attendance_records_for_date(date):
    """Load existing attendance records for a specific date from the CSV file."""
    attendance_file_path = os.path.join(attendance_dir, f"{date}.csv")
    if os.path.exists(attendance_file_path):
        with open(attendance_file_path, 'r') as f:
            csv_reader = csv.reader(f)
            try:
                next(csv_reader)  # Skip the header row
            except StopIteration:
                return []  # If the file is empty, return an empty list
            return list(csv_reader)  # Return the rest of the rows without the header
    return []

def save_attendance_record(name, arrival_time, departure_time=""):
    """Save attendance record in the CSV file for the current date."""
    attendance_csv_file_path = os.path.join(attendance_dir, f"{current_date}.csv")
    
    # Load attendance records for the current day (if any)
    attendance_data = load_attendance_records_for_date(current_date)
    
    # Prevent marking attendance multiple times for the same person on the same day
    for record in attendance_data:
        if record[0] == name:
            return  # Prevent re-marking
    
    # Create or append to the file
    file_exists = os.path.exists(attendance_csv_file_path)
    with open(attendance_csv_file_path, 'a', newline='') as attendance_file:
        writer = csv.writer(attendance_file)
        if not file_exists:  # If the file doesn't exist, write the header
            writer.writerow(["Name", "Arrival Time", "Departure Time"])
        writer.writerow([name, arrival_time, departure_time])  # Save attendance record

def save_departure_time(name, departure_time):
    """Update the departure time for a given name in the CSV file."""
    attendance_csv_file_path = os.path.join(attendance_dir, f"{current_date}.csv")
    
    # Load the existing attendance data
    attendance_data = load_attendance_records_for_date(current_date)
    
    # Find the record for the user and update the departure time
    updated = False
    for record in attendance_data:
        if record[0] == name and record[2] == "":  # Departure time is empty
            record[2] = departure_time
            updated = True
            break
    
    # If updated, write the new data back to the CSV file
    if updated:
        with open(attendance_csv_file_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Name", "Arrival Time", "Departure Time"])  # Header
            writer.writerows(attendance_data)  # Save the updated rows

@app.route('/mark/<type_of_attendance>', methods=['GET', 'POST'])
def mark_attendance(type_of_attendance):
    reference_data = []  # List to store marked students' info
    displayed_messages = set()  # Set to store displayed messages for this session

    if request.method == 'POST':
        # Clear attendance tracking for the selected type of attendance
        attendance_tracking[type_of_attendance].clear()
        reference_data.clear()
        displayed_messages.clear()
        return redirect(url_for('index'))
    
    return render_template('attendance.html', type_of_attendance=type_of_attendance, reference_data=reference_data)

# Global list to store attendance messages
attendance_messages = []

@app.route('/video_feed/<type_of_attendance>')
def video_feed(type_of_attendance):
    def generate_frames():
        video_capture = cv2.VideoCapture(0)
        video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, video_capture.get(cv2.CAP_PROP_FRAME_WIDTH) // 2)
        video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT) // 2)
        known_face_names, known_face_encodings = load_users()
        
        attendance_messages.clear()
        reference_data = []  # List to store marked users' info
        displayed_messages = set()  # Set to track displayed messages for each session

        while True:
            ret, frame = video_capture.read()
            if not ret:
                break

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face_locations = face_recognition.face_locations(rgb_frame)
            face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

            for face_encoding in face_encodings:
                matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.4)  
                name = "Unknown"

                if True in matches:
                    best_match_index = np.argmin(face_recognition.face_distance(known_face_encodings, face_encoding))
                    name = known_face_names[best_match_index]

                if name != "Unknown" and name not in displayed_messages:
                    # Load the current day's attendance records
                    attendance_data = load_attendance_records_for_date(current_date)

                    user_recorded = False
                    user_arrival = False
                    user_departure = False
                    
                    for record in attendance_data:
                        if record[0] == name:  # If the user exists in today's attendance records
                            user_recorded = True
                            if record[2] == "":  # If the user has marked arrival but not departure
                                user_arrival = True
                            else:  # If both arrival and departure are recorded
                                user_departure = True
                                reference_data.append(f"{name} already marked departure.")
                                print(f"{name} already marked departure.")
                                attendance_messages.append(f"{name} already marked departure.")
                            break

                    if user_recorded == True :
                        reference_data.append(f"{name} already marked arrival.")
                        print(f"{name} already marked arrival.")
                        attendance_messages.append(f"{name} already marked arrival.")
                        
                    if not user_recorded:
                        # If the user hasn't marked any attendance today
                        if type_of_attendance == "arrival":
                            attendance_tracking["arrival"].add(name)
                            save_attendance_record(name, datetime.datetime.now().strftime("%H:%M:%S"))
                            reference_data.append(f"{name} marked arrival.")
                            print(f"{name} marked arrival.")
                            attendance_messages.append(f"{name} marked arrival.")
                        elif type_of_attendance == "departure":
                            reference_data.append(f"{name}, please mark your arrival first before departure.")
                            print(f"{name}, please mark your arrival first before departure.")
                            attendance_messages.append(f"{name} please mark your arrival first before departure.")
                    
                    elif user_arrival and not user_departure:  # User has arrival marked but not departure
                        # Allow the user to mark departure
                        if type_of_attendance == "departure":
                            attendance_tracking["departure"].add(name)
                            save_departure_time(name, datetime.datetime.now().strftime("%H:%M:%S"))
                            reference_data.append(f"{name} marked departure.")
                            print(f"{name} marked departure.")
                            attendance_messages.append(f"{name} marked departure.")
                    elif user_departure:  # User has both marked arrival and departure
                        reference_data.append(f"{name} already marked attendance for today.")
                        print(f"{name} already marked attendance for today.")
                        attendance_messages.append(f"{name} already marked attendance for today.")
                                            
                    # Add user to displayed messages to prevent re-displaying in the same session
                    displayed_messages.add(name)

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

        video_capture.release()

    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# Route to fetch messages from the global list
@app.route('/get_messages', methods=['GET'])
def get_messages():
    return jsonify(attendance_messages)


# Remaining routes and functions unchanged...


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/admin', methods=['GET', 'POST'])
def admin_login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')

        # Validate credentials
        if username == 'a' and password == 'a':
            return redirect(url_for('admin_panel'))
        else:
            return render_template('admin_login.html', error="Incorrect username or password")
    
    return render_template('admin_login.html')


@app.route('/admin_panel', methods=['GET', 'POST'])
def admin_panel():
    if request.method == 'POST':
        action = request.form.get('action')

        # Add new user
        if action == 'add_user':
            add_user()

        # Delete user
        elif action == 'delete_user':
            name = request.form.get('delete_name')
            photo_path = os.path.join(photos_dir, f"{name}.jpg")
            if os.path.exists(photo_path):
                os.remove(photo_path)
                flash("User deleted successfully!", 'delete_user_success')
            else:
                flash("User not found", 'delete_user_error')

    return render_template('admin_panel.html')

def add_user():    
    name = request.form.get('name')
    photo = request.files.get('photo')
    
    if photo and (photo.filename.endswith('.jpg') or photo.filename.endswith('.jpeg')):
        photo_path = os.path.join(photos_dir, f"{name}.jpg")
        photo.save(photo_path)
        try:
            # Attempt to create face encoding for the new user
            new_image = face_recognition.load_image_file(photo_path)
            new_encodings = face_recognition.face_encodings(new_image)

            if not new_encodings:  # If no face found
                if os.path.exists(photo_path):
                    os.remove(photo_path)  # Remove the photo
                flash("No face detected in the uploaded photo. Please try again with a valid photo.", 'add_user_error')
            else:
                flash("User added successfully!", 'add_user_success')  # Flash success message

        except Exception as e:
            # Handle any other exceptions (e.g., invalid image format)
            if os.path.exists(photo_path):
                os.remove(photo_path)  # Remove the photo
            flash(f"An error occurred: {str(e)}", 'add_user_error')
    else:
        flash("Invalid file format. Please upload a JPG or JPEG photo.", 'add_user_error')
    
    return redirect(url_for('admin_panel'))



@app.route('/all_users')
def all_users():
    # Get the list of all known user names
    global user
    user=[]
    for file_name in os.listdir(photos_dir):
        if file_name.endswith('.jpg'):
            nam = os.path.splitext(file_name)[0]
            if nam:
                user.append(nam)
    users = user
    return render_template('all_users.html', users=users)


@app.route('/records', methods=['GET', 'POST'])
def records():
    current_date = datetime.datetime.now().strftime("%Y-%m-%d")
    dates = list_all_dates()

    if request.method == 'POST':
        selected_date = request.form.get('date')

        attendance_data = load_attendance_records_for_date(selected_date)
        if not attendance_data:
            return render_template('records.html', dates=dates, selected_date=selected_date, message="No records found for selected date.", current_date=current_date)
        return render_template('records.html', records=attendance_data, dates=dates, selected_date=selected_date, current_date=current_date)

    return render_template('records.html', dates=dates, current_date=current_date)

def list_all_dates():
    dates = []
    for filename in os.listdir(attendance_dir):
        if filename.endswith(".csv"):
            date = filename.split('.')[0]
            dates.append(date)
    return sorted(dates, reverse=True)

if __name__ == '__main__':
    app.run(debug=True)