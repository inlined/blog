import face_recognition
import os
import sys

from firebase_functions import storage_fn
from firebase_admin import initialize_app
from io import BytesIO
from math import atan2, degrees
from numpy import np
from PIL import Image

initialize_app()

ASPECT_RATIO = 2.25 / 1.75


@storage_fn.on_object_finalized(bucket="scuba-site")
def on_upload_profile(event: storage_fn.CloudEvent):
    pass


def download_image(image_path: str) -> BytesIO:
    pass


def align_image(bytes: BytesIO) -> BytesIO:
    image = face_recognition.load_image_file(bytes)

    # Find face landmarks
    face_landmarks_list = face_recognition.face_landmarks(image)

    # Ensure there is exactly one face in the image
    if len(face_landmarks_list) != 1:
        raise ValueError("The image does not have exactly one face.")

    face_landmarks = face_landmarks_list[0]

    # For simplicity, take the eyes' landmarks to calculate the angle
    left_eye = face_landmarks["left_eye"]
    right_eye = face_landmarks["right_eye"]

    # Calculate the angle between the eyes
    left_eye_center = np.mean(left_eye, axis=0)
    right_eye_center = np.mean(right_eye, axis=0)
    delta_x = right_eye_center[0] - left_eye_center[0]
    delta_y = right_eye_center[1] - left_eye_center[1]
    angle = atan2(delta_y, delta_x)

    # Rotate the original image
    pil_image = Image.open(bytes)
    rotated_image = pil_image.rotate(degrees(-angle))

    # Convert the rotated image back to bytes
    image_byte_array = BytesIO()
    rotated_image.save(image_byte_array, format="JPEG")
    rotated_image_bytes = image_byte_array.getvalue()

    return rotated_image_bytes


# Assuming image is the path to an image file
def crop_image(bytes: BytesIO) -> BytesIO:
    # Load the image file
    image = face_recognition.load_image_file(bytes)

    # Find all face locations in the image
    face_locations = face_recognition.face_locations(image)

    if len(face_locations) != 1:
        raise ValueError("Expected exactly one face")

    face_top, face_right, face_bottom, face_left = face_locations[0]

    face_height = face_bottom - face_top
    height = face_height / 0.75
    top = max(face_top - height * 0.06, 0)
    bottom = top + height

    face_center = (face_right + face_left) / 2
    width = height / ASPECT_RATIO
    left = face_center - width / 2
    right = face_center + width / 2

    # Create a PIL Image from the array
    pil_image = Image.fromarray(image)

    # Crop the image to the desired size
    cropped_image = pil_image.crop((left, top, right, bottom))

    image_byte_array = BytesIO()
    cropped_image.save(image_byte_array, format="JPEG")
    return image_byte_array.getvalue()


def resize_image(bytes: BytesIO) -> BytesIO:
    MAX_SIZE_KB = 350
    STEP_PERCENT = 5

    image = Image.open(bytes)
    scale_factor = 1
    resized_image = image
    resized_bytes = bytes
    while len(resized_bytes.getvalue()) > MAX_SIZE_KB * 1024:
        scale_factor *= (100 - STEP_PERCENT) / 100
        new_width = int(bytes.size[0] * scale_factor)
        new_height = int(bytes.size[1] * scale_factor)

        resized_image = image.resize((new_width, new_height), Image.ANTIALIAS)

        resized_bytes = BytesIO()
        resized_image.save(resized_bytes, format="JPEG")

    return resized_bytes


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python main.py <path to image>")
        sys.exit(1)

    directory, file_name = os.path.split(sys.argv[1])
    basename, extension = os.path.splitext(file_name)

    with open(sys.argv[1], "rb") as input_file:
        original_bytes = input_file.read()
    bytes = BytesIO(original_bytes)

    bytes = align_image(bytes)
    aligned_path = os.path.join(directory, f"{basename}_aligned.jpg")
    with open(aligned_path, "wb") as aligned_file:
        aligned_file.write(bytes.getvalue())

    bytes = crop_image(bytes)
    cropped_path = os.path.join(directory, f"{basename}_cropped.jpg")
    with open(cropped_path, "wb") as cropped_file:
        cropped_file.write(bytes.getvalue())

    bytes = resize_image(bytes)
    resized_path = os.path.join(directory, f"{basename}_resized.jpg")
    with open(resized_path, "wb") as resized_file:
        resized_file.write(bytes.getvalue())
