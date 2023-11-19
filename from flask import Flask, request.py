from flask import Flask, request
from preprocessing import *
from staff_removal import *
from helper_methods import *
import os
import datetime
import pickle
from flask_cors import CORS
from werkzeug.utils import secure_filename

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Specify output folder path
output_folder = "./output"
input_folder = "./input"

threshold = 0.8
filename = "model/model.sav"
model = pickle.load(open(filename, "rb"))
accidentals = ["x", "hash", "b", "symbol_bb", "d"]


def preprocessing(fn, f):
    # Get image and its dimensions #
    height, width, in_img = preprocess_img(fn)

    # Get line thickness and list of staff lines #
    staff_lines_thicknesses, staff_lines = get_staff_lines(
        width, height, in_img, threshold
    )

    # Remove staff lines from the original image #
    cleaned = remove_staff_lines(in_img, width, staff_lines, staff_lines_thicknesses)

    # Get list of cut buckets and cutting positions #
    cut_positions, cutted = cut_image_into_buckets(cleaned, staff_lines)

    # Get reference line for each bucket #
    ref_lines, lines_spacing = get_ref_lines(cut_positions, staff_lines)

    return cutted, ref_lines, lines_spacing


# Rest of the code remains the same...


@app.route("/process", methods=["POST"])
def process():
    # Receive input file from the request
    input_file = request.files["image"]

    try:
        os.mkdir(output_folder)
    except OSError as error:
        pass

    # Save the uploaded file to the input folder
    input_filename = secure_filename(input_file.filename)
    input_filepath = os.path.join(input_folder, input_filename)
    input_file.save(input_filepath)

    # Open the output text file #
    file_prefix = input_filename.split(".")[0]
    output_filepath = os.path.join(output_folder, f"{file_prefix}.txt")
    f = open(output_filepath, "w")

    # Process the uploaded image
    try:
        process_image(input_filepath, f)
    except Exception as e:
        print(e)
        print(f"{input_filepath} has failed!")
        pass

    f.close()
    print("Finished processing!")

    return {"message": "Processing completed.", "downloadURL": output_filepath}


if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0")