import cv2
import os
import csv
import tkinter as tk
from tkinter import messagebox, simpledialog

# Define paths
BASE_DIR = r"D:\python\facerec_project"
CSV_FOLDER = os.path.join(BASE_DIR, "StudentRecords")
IMAGE_FOLDER = os.path.join(BASE_DIR, "TrainingImages")
CSV_FILE_PATH = os.path.join(CSV_FOLDER, "StudentDetails.csv")
HAARCASCADE_PATH = os.path.join(BASE_DIR, "haarcascade_frontalface_default.xml")

# Ensure necessary directories exist
def assure_path_exists(path):
    os.makedirs(path, exist_ok=True)

# Check if Haarcascade file exists
def check_haarcascadefile():
    if not os.path.isfile(HAARCASCADE_PATH):
        messagebox.showerror("Error", "Haarcascade file is missing! Please add it.")
        exit()

# Capture images and save student details with only one image filename in CSV
def TakeImages():
    check_haarcascadefile()
    
    # Create directories if they don't exist
    assure_path_exists(CSV_FOLDER)
    assure_path_exists(IMAGE_FOLDER)

    # Get user input for ID and Name
    Id = simpledialog.askstring("Input", "Enter ID:")
    name = simpledialog.askstring("Input", "Enter Name:")

    if not name or not Id or not name.replace(" ", "").isalpha() or not Id.isdigit():
        messagebox.showerror("Error", "Enter a valid numeric ID and alphabetic name!")
        return

    # Check if the ID already exists in CSV
    if os.path.isfile(CSV_FILE_PATH):
        with open(CSV_FILE_PATH, 'r', newline='') as csvFile:
            reader = csv.reader(csvFile)
            for row in reader:
                if row and row[0] == Id:
                    messagebox.showerror("Error", "Student ID already exists!")
                    return

    # Start capturing images
    cam = cv2.VideoCapture(0)
    if not cam.isOpened():
        messagebox.showerror("Error", "Could not open webcam!")
        return

    detector = cv2.CascadeClassifier(HAARCASCADE_PATH)
    sampleNum = 0
    first_image_filename = ""

    while True:
        ret, img = cam.read()
        if not ret:
            messagebox.showerror("Error", "Couldn't capture image!")
            break

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = detector.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
            sampleNum += 1
            img_filename = f"{name}_{Id}_{sampleNum}.jpg"
            img_path = os.path.join(IMAGE_FOLDER, img_filename)
            cv2.imwrite(img_path, gray[y:y + h, x:x + w])

            # Store only the first image filename
            if sampleNum == 1:
                first_image_filename = img_filename

        cv2.imshow('Capturing Images', img)

        if cv2.waitKey(1) & 0xFF == ord('q') or sampleNum >= 50:  # Stop after 50 images
            break

    cam.release()
    cv2.destroyAllWindows()

    # Save ID, Name, and only the first image filename in the CSV file
    if first_image_filename:
        with open(CSV_FILE_PATH, 'a', newline='') as csvFile:
            writer = csv.writer(csvFile)
            writer.writerow([Id, name, first_image_filename])

        messagebox.showinfo("Success", f"Images taken for ID: {Id}, and details saved in {CSV_FILE_PATH}")

# GUI setup
def start_gui():
    window = tk.Tk()
    window.title("Face Recognition - Image Capture")
    window.geometry("400x200")
    window.configure(bg="lightblue")

    tk.Label(window, text="Face Recognition System", font=("Arial", 16, "bold"), bg="lightblue").pack(pady=10)
    tk.Button(window, text="Take Images", command=TakeImages, font=("Arial", 12), bg="green", fg="white").pack(pady=10)
    tk.Button(window, text="Exit", command=window.quit, font=("Arial", 12), bg="red", fg="white").pack(pady=5)

    window.mainloop()

# Run GUI
if __name__ == "__main__":
    start_gui()
