from flask import Flask, request
from preprocessing import *
from staff_removal import *
from helper_methods import *
import os
import datetime
import pickle

app = Flask(__name__)

# Specify output folder path
output_folder = "./output"

threshold = 0.8
filename = "model/model.sav"
model = pickle.load(open(filename, "rb"))
accidentals = ["x", "hash", "b", "symbol_bb", "d"]


def preprocessing(input_folder, fn, f):
    # Get image and its dimensions #
    height, width, in_img = preprocess_img("{}/{}".format(input_folder, fn))

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


def get_target_boundaries(label, cur_symbol, y2):
    if label == "b_8":
        cutted_boundaries = cut_boundaries(cur_symbol, 2, y2)
        label = "a_8"
    elif label == "b_8_flipped":
        cutted_boundaries = cut_boundaries(cur_symbol, 2, y2)
        label = "a_8_flipped"
    elif label == "b_16":
        cutted_boundaries = cut_boundaries(cur_symbol, 4, y2)
        label = "a_16"
    elif label == "b_16_flipped":
        cutted_boundaries = cut_boundaries(cur_symbol, 4, y2)
        label = "a_16_flipped"
    else:
        cutted_boundaries = cut_boundaries(cur_symbol, 1, y2)

    return label, cutted_boundaries


def get_label_cutted_boundaries(boundary, height_before, cutted):
    # Get the current symbol #
    x1, y1, x2, y2 = boundary
    cur_symbol = cutted[y1 - height_before : y2 + 1 - height_before, x1 : x2 + 1]

    # Clean and cut #
    cur_symbol = clean_and_cut(cur_symbol)
    cur_symbol = 255 - cur_symbol

    # Start prediction of the current symbol #
    feature = extract_hog_features(cur_symbol)
    label = str(model.predict([feature])[0])

    return get_target_boundaries(label, cur_symbol, y2)


def process_image(input_folder, fn, f):
    cutted, ref_lines, lines_spacing = preprocessing(input_folder, fn, f)

    last_acc = ""
    last_num = ""
    height_before = 0

    if len(cutted) > 1:
        f.write("{\n")

    for it in range(len(cutted)):
        f.write("[")
        is_started = False

        symbols_boundaries = segmentation(height_before, cutted[it])
        symbols_boundaries.sort(key=lambda x: (x[0], x[1]))

        for boundary in symbols_boundaries:
            label, cutted_boundaries = get_label_cutted_boundaries(
                boundary, height_before, cutted[it]
            )

            if label == "clef":
                is_started = True

            for cutted_boundary in cutted_boundaries:
                _, y1, _, y2 = cutted_boundary
                if is_started == True and label != "barline" and label != "clef":
                    text = text_operation(
                        label, ref_lines[it], lines_spacing[it], y1, y2
                    )

                    if (label == "t_2" or label == "t_4") and last_num == "":
                        last_num = text
                    elif label in accidentals:
                        last_acc = text
                    else:
                        if last_acc != "":
                            text = text[0] + last_acc + text[1:]
                            last_acc = ""

                        if last_num != "":
                            text = f'\meter<"{text}/{last_num}">'
                            last_num = ""

                        not_dot = label != "dot"
                        f.write(not_dot * " " + text)

        height_before += cutted[it].shape[0]
        f.write(" ]\n")

    if len(cutted) > 1:
        f.write("}")


@app.route("/process", methods=["POST"])
def process():
    # Receive input folder path from the request
    input_folder = request.json["input_folder"]

    try:
        os.mkdir(output_folder)
    except OSError as error:
        pass

    list_of_images = os.listdir(input_folder)
    for _, fn in enumerate(list_of_images):
        # Open the output text file #
        file_prefix = fn.split(".")[0]
        f = open(f"{output_folder}/{file_prefix}.txt", "w")

        # Process each image separately #
        try:
            process_image(input_folder, fn, f)
        except Exception as e:
            print(e)
            print(f"{input_folder}-{fn} has been failed !!")
            pass

        f.close()
    print("Finished !!")

    return {"message": "Processing completed."}


if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0")
