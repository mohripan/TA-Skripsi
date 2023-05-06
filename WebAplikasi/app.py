from flask import Flask, render_template, request
import os
import base64
import replicate
import requests

app = Flask(__name__)

api_token = os.environ.get("REPLICATE_API_TOKEN")
if api_token is None:
    raise ValueError("REPLICATE_API_TOKEN environment variable not set")

imgbb_api_key = "21bfe0fd42b7b01e1f230393a6c1c0ea"

def upload_image_to_imgbb(image_data):
    url = "https://api.imgbb.com/1/upload"
    payload = {
        "key": imgbb_api_key,
        "image": base64.b64encode(image_data).decode("utf-8"),
    }
    response = requests.post(url, data=payload)
    if response.status_code == 200:
        return response.json()["data"]["url"]
    else:
        raise Exception("Error uploading image to imgbb")

@app.route("/", methods=["GET", "POST"])
def index():
    input_image_url = None
    output_urls = None
    
    if request.method == "POST":
        image = request.files["image"]
        input_image = image.stream.read()
        input_image_url = upload_image_to_imgbb(input_image)

        # Get the user's model choice from the form
        model_choice = request.form["model"]

        # Call the appropriate model based on the user's choice
        if model_choice == "face_restoration":
            output = replicate.run(
                "tencentarc/gfpgan:9283608cc6b7be6b65a8e44983db012355fde4132009bf99d976b2f0896856a3",
                input={"img": input_image_url}
            )
        elif model_choice == "image_super_resolution":
            output = replicate.run(
                "cjwbw/real-esrgan:d0ee3d708c9b911f122a4ad90046c5d26a0293b99476d697f6bb7f2e251ce2d4",
                input={"image": input_image_url}
            )
        elif model_choice == "image_debluring":
            output = replicate.run(
                "megvii-research/nafnet:018241a6c880319404eaa2714b764313e27e11f950a7ff0a7b5b37b27b74dcf7",
                input={"image": input_image_url, "task_type": "Image Debluring (REDS)"}
            )
        else:
            raise ValueError("Invalid model choice")

        output_urls = [output]
        return render_template("index.html", input_image_url=input_image_url, output_urls=output_urls)

    return render_template("index.html", input_image_url=input_image_url, output_urls=output_urls)


if __name__ == "__main__":
    app.run(debug=True)
