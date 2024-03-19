import streamlit as st
import numpy as np
import cv2
import os
import string
import random
from pathlib import Path
from PIL import Image
import base64 
import time
timestr = time.strftime("%Y%m%d-%H%M%S")

class imgdown(object):
	
	def __init__(self, data,filename='myfile',file_ext='png,jpg'):
		super(imgdown, self).__init__()
		self.data = data
		self.filename = filename
		self.file_ext = file_ext

	def download(self):
		b64 = base64.b64encode(self.data.encode()).decode()
		new_filename = "{}_{}_.{}".format(self.filename,timestr,self.file_ext)
		st.markdown("#### Download File ###")
		href = f'<a href="data:file/{self.file_ext};base64,{b64}" download="{new_filename}">Click Here!!</a>'
		st.markdown(href,unsafe_allow_html=True)

def brighten_image(image, amount):
    img_bright = cv2.convertScaleAbs(image, beta=amount)
    return img_bright


def blur_image(image, amount):
    blur_img = cv2.GaussianBlur(image, (11, 11), amount)
    return blur_img


def enhance_details(img):
    hdr = cv2.detailEnhance(img, sigma_s=12, sigma_r=0.15)
    return hdr

def calculate_energy(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gradient_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gradient_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    energy_map = np.sqrt(gradient_x**2 + gradient_y**2)
    return energy_map

def seam_carve(image, energy_map, num_seams):
    for _ in range(num_seams):
        cumulative_energy = np.zeros_like(energy_map)
        cumulative_energy[0, :] = energy_map[0, :]
        for i in range(1, energy_map.shape[0]):
            for j in range(energy_map.shape[1]):
                if j == 0:
                    cumulative_energy[i, j] = energy_map[i, j] + min(cumulative_energy[i - 1, j], cumulative_energy[i - 1, j + 1])
                elif j == energy_map.shape[1] - 1:
                    cumulative_energy[i, j] = energy_map[i, j] + min(cumulative_energy[i - 1, j - 1], cumulative_energy[i - 1, j])
                else:
                    cumulative_energy[i, j] = energy_map[i, j] + min(cumulative_energy[i - 1, j - 1], cumulative_energy[i - 1, j], cumulative_energy[i - 1, j + 1])

        seam_mask = np.zeros_like(energy_map, dtype=bool)
        j = np.argmin(cumulative_energy[-1, :])
        for i in range(energy_map.shape[0] - 1, -1, -1):
            seam_mask[i, j] = True
            if j == 0:
                j = np.argmin(cumulative_energy[i - 1, j:j + 2])
            elif j == energy_map.shape[1] - 1:
                j = np.argmin(cumulative_energy[i - 1, j - 1:j + 1]) + j - 1
            else:
                j = np.argmin(cumulative_energy[i - 1, j - 1:j + 2]) + j - 1

        image = image[~seam_mask].reshape((image.shape[0], image.shape[1] - 1, image.shape[2]))
        energy_map = calculate_energy(image)

    return image
def seam_insert(image, energy_map, num_seams):
    for _ in range(num_seams):
        cumulative_energy = np.zeros_like(energy_map)
        cumulative_energy[0, :] = energy_map[0, :]
        for i in range(1, energy_map.shape[0]):
            for j in range(energy_map.shape[1]):
                if j == 0:
                    cumulative_energy[i, j] = energy_map[i, j] + min(cumulative_energy[i - 1, j], cumulative_energy[i - 1, j + 1])
                elif j == energy_map.shape[1] - 1:
                    cumulative_energy[i, j] = energy_map[i, j] + min(cumulative_energy[i - 1, j - 1], cumulative_energy[i - 1, j])
                else:
                    cumulative_energy[i, j] = energy_map[i, j] + min(cumulative_energy[i - 1, j - 1], cumulative_energy[i - 1, j], cumulative_energy[i - 1, j + 1])

        seam_mask = np.zeros_like(energy_map, dtype=bool)
        j = np.argmin(cumulative_energy[-1, :])
        for i in range(energy_map.shape[0] - 1, -1, -1):
            seam_mask[i, j] = True
            if j == 0:
                j = np.argmin(cumulative_energy[i - 1, max(0, j):j + 2])
            elif j == energy_map.shape[1] - 1:
                j = np.argmin(cumulative_energy[i - 1, max(0, j - 1):j + 1]) + j - 1
            else:
                j = np.argmin(cumulative_energy[i - 1, max(0, j - 1):j + 2]) + j - 1

        new_image = np.zeros((image.shape[0], image.shape[1] + 1, image.shape[2]), dtype=np.uint8)
        new_energy_map = np.zeros((energy_map.shape[0], energy_map.shape[1] + 1))
        
        for i in range(image.shape[0]):
            k = 0
            for j in range(image.shape[1] + 1):
                if j < image.shape[1] and seam_mask[i, j]:
                    new_image[i, j] = image[i, k]
                    new_energy_map[i, j] = energy_map[i, k]
                    k += 1
                else:
                    if j == 0:
                        new_image[i, j] = image[i, k]
                        new_energy_map[i, j] = energy_map[i, k]
                    elif j == image.shape[1]:
                        new_image[i, j] = image[i, k - 1]
                        new_energy_map[i, j] = energy_map[i, k - 1]
                    else:
                        new_image[i, j] = (image[i, k - 1] + image[i, k]) // 2
                        new_energy_map[i, j] = (energy_map[i, k - 1] + energy_map[i, k]) // 2
                    k += 1
        image = new_image
        energy_map = new_energy_map
    return image

def visualize_energy_map(img):
    energy_map = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2GRAY)
    energy_map = cv2.Sobel(energy_map, cv2.CV_64F, 1, 0) ** 2 + cv2.Sobel(energy_map, cv2.CV_64F, 0, 1) ** 2
    return energy_map.astype(np.uint8)

def namegen(size=6, chars=string.ascii_uppercase + string.digits):
    return ''.join(random.choice(chars) for _ in range(size))

def upload_image(img_file):
    if not img_file:
        return None

def main():
    st.title("Seam Carving App")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

    if uploaded_file:
        upload_image(uploaded_file)
        st.image(uploaded_file)
        file_details = {"filename":uploaded_file.name, "filetype":uploaded_file.type, "filesize":uploaded_file.size}
        st.write(file_details)
    else:
        st.image("a.jpg")
        st.stop()


    if uploaded_file is not None:
        col1, col2 = st.columns(2)
        original_image = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), 1)
        with col1:
            st.image(original_image, caption="Original Image", use_column_width=True)
        energy_map = calculate_energy(original_image)
        num_seams = st.sidebar.slider("Select number of seams to remove or insert", 1, 100, 10)
        operation = st.sidebar.selectbox("Select operation", ["Remove Seams", "Insert Seams"])
        energy_map_img = visualize_energy_map(original_image)
        with col2:
            st.image(energy_map_img, caption="Energy Map", clamp=True, channels="GRAY")
            if st.sidebar.button("Apply Operation"):
                if operation == "Remove Seams":
                    result_image = seam_carve(original_image, energy_map, num_seams)
                else:
                    result_image = seam_insert(original_image, energy_map, num_seams)
                    st.image(result_image, caption="Processed image ", use_column_width=True)
                    directory = "output_img"
                    path = os.path.join(os.getcwd(), directory)
                    p = Path(path)
                    if not p.exists():
                        os.mkdir(p)
                    randname = namegen()
                    processed_image_path = os.path.join(path, f"{randname}_processed.png")
                    # Save processed image using cv2
                    if st.sidebar.button("Save Processed Image"):
                        directory = "tempDir"
                        path = os.path.join(os.getcwd(), directory)
                        p = Path(path)
                        if not p.exists():
                            os.mkdir(p)
                        
                        randname = namegen()
                        processed_image_path = os.path.join(path, f"{randname}_processed.png")
                        # Save processed image using cv2
                        cv2.imwrite(processed_image_path, cv2.cvtColor(result_image, cv2.COLOR_RGB2BGR))
                        st.success(f"Processed image saved at: {processed_image_path}")


if __name__ == "__main__":
    main()
