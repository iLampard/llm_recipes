import requests
import base64


def generate_with_image_and_text(text_prompt: str, image_path=None, url="http://localhost:12000/v1/chat/completions"):
    """
    Generate text based on an image and text prompt using the Llama model.

    Args:
    - text_prompt (str): The text prompt to guide the generation.
    - image_url (str): URL to the image to use as context.

    Returns:
    - str: Generated text.
    """
    # Load the image from the provided URL
    # Load and encode the image as base64
    # Prepare the request data
    data = {
        "text": text_prompt,
        "image": ""
    }

    # Encode the image in Base64 if image_path is provided
    if image_path is not None:
        with open(image_path, "rb") as image_file:
            image_base64 = base64.b64encode(image_file.read()).decode("utf-8")
        data["image"] = image_base64  # Include image only if it's provided

    # Make the request
    headers = {"Content-Type": "application/json"}
    response = requests.post(url, json=data, headers=headers)

    # Check response status and print the result
    if response.status_code == 200:
        return response.json()
    else:
        print("Failed with status code:", response.status_code)
        print("Error details:", response.text)
        return


if __name__ == "__main__":
    print(generate_with_image_and_text('what is in the image', 'cloth.png'))
    print(generate_with_image_and_text('who are you'))

