from flask import Flask, send_file, abort
import os

app = Flask(__name__)

# Replace this with the actual path to your image folder
IMAGE_FOLDER = "datasets/ffhq/1k"


@app.route("/<image_id>.png")
def serve_image(image_id):
    # Construct the image path
    image_path = os.path.join(IMAGE_FOLDER, f"{image_id}.png")
    if os.path.exists(image_path):
        return send_file(image_path, mimetype="image/png")
    else:
        abort(404)  # Return a 404 error if the image is not found


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
