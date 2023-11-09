import face_recognition
import os
import sys

from firebase_functions import storage_fn
from firebase_admin import initialize_app
from io import BytesIO
from math import atan2, degrees
from numpy import mean
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
    if len(face_landmarks_list) == 0:
        raise ValueError("Could not find a face to align")

    face_landmarks = face_landmarks_list[0]

    # For simplicity, take the eyes' landmarks to calculate the angle
    left_eye = face_landmarks["left_eye"]
    right_eye = face_landmarks["right_eye"]

    # Calculate the angle between the eyes
    left_eye_center = mean(left_eye, axis=0)
    right_eye_center = mean(right_eye, axis=0)
    delta_x = right_eye_center[0] - left_eye_center[0]
    delta_y = right_eye_center[1] - left_eye_center[1]
    angle = atan2(delta_y, delta_x)

    # Rotate the original image
    pil_image = Image.open(bytes)
    rotated_image = pil_image.rotate(degrees(angle))

    # Convert the rotated image back to bytes
    image_byte_array = BytesIO()
    rotated_image.save(image_byte_array, format="JPEG", quality=100)
    return image_byte_array


# Assuming image is the path to an image file
def crop_image(bytes: BytesIO) -> BytesIO:
    # Load the image file
    image = face_recognition.load_image_file(bytes)

    # Find all face locations in the image
    face_locations = face_recognition.face_locations(image)

    if len(face_locations) == 0:
        raise ValueError("Could not find a face to crop")

    face_top, face_right, face_bottom, face_left = face_locations[0]

    face_height = face_bottom - face_top
    face_center = (face_top + face_bottom) / 2
    # multiplying by a heuristic numbers because face_recognition
    # does not recognize whole heads.
    head_height = 1.5 * face_height
    head_top = max(face_center - head_height * 0.75, 0)
    height = head_height / 0.75
    top = max(head_top - height * 0.06, 0)
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
    cropped_image.save(image_byte_array, format="JPEG", quality=100)
    return image_byte_array


def resize_image(bytes: BytesIO) -> BytesIO:
    MAX_SIZE_KB = 350
    STEP_PERCENT = 5

    quality = 100
    resized_bytes = bytes
    image = Image.open(bytes)

    while len(resized_bytes.getvalue()) > MAX_SIZE_KB * 1024:
        quality -= STEP_PERCENT
        resized_bytes = BytesIO()
        image.save(resized_bytes, format="JPEG", quality=quality)

    return resized_bytes


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python main.py <directory>")
        sys.exit(1)

    with os.scandir(sys.argv[1]) as it:
        for entry in it:
            if not entry.is_file():
                continue

            basename, extension = os.path.splitext(entry.name)

            if extension != ".jpg" and extension != ".jpeg":
                print(f"Skipping {basename}{extension}")
                continue

            print(f"Processing {entry.name}")

            try:
                print("  Loading original file")
                with open(entry, "rb") as input_file:
                    original_bytes = input_file.read()
                bytes = BytesIO(original_bytes)

                print("  Aligning face")
                bytes = align_image(bytes)

                print("  Cropping face")
                bytes = crop_image(bytes)

                print("Downsizing")
                bytes = resize_image(bytes)

                print("Saving")
                resized_path = os.path.join(sys.argv[1], f"{basename}_thumb.jpg")
                with open(resized_path, "wb") as resized_file:
                    resized_file.write(bytes.getvalue())

            except Exception as err:
                print(f"Failed with error {err}")
