import tkinter as tk
from tkinter import filedialog
from io import BytesIO
import requests
from PIL import Image, ImageTk
import numpy as np
from keras.preprocessing.image import img_to_array
from keras.applications import imagenet_utils
from keras.models import load_model


def start():

    global upload
    upload = tk.Button(form, text="Upload an image", command=upload_image, padx=20, pady=10)
    upload.configure(background='#364156', foreground='gray', font=('arial', 20, 'bold'))
    upload.pack(side=tk.TOP, pady=50)


def upload_image():

    upload.destroy()

    global load_from_url_button
    load_from_url_button = tk.Button(form, text='Load from URL', command=load_from_url, padx=50, pady=10)
    load_from_url_button.configure(background='#364156', foreground='gray', font=('arial', 15, 'bold'))
    load_from_url_button.place(relx=0.1, rely=0.1)

    global load_from_localdisk_button
    load_from_localdisk_button = tk.Button(form, text='Load from local disk', command=load_from_localdisk, padx=50, pady=10)
    load_from_localdisk_button.configure(background='#364156', foreground='gray', font=('arial', 15, 'bold'))
    load_from_localdisk_button.place(relx=0.55, rely=0.1)


def load_from_url():

    load_from_url_button.destroy()
    load_from_localdisk_button.destroy()

    global entry
    entry = tk.Entry(form)
    entry.place(width=230, height=30, relx=0.35, rely=0.15)

    global show_image_button
    show_image_button = tk.Button(form, text='Show Image', command=get_image, padx=50, pady=10)
    show_image_button.configure(background='#364156', foreground='gray', font=('arial', 15, 'bold'))
    show_image_button.place(relx=0.35, rely=0.25)

def load_from_localdisk():

    global image_local
    file_path = filedialog.askopenfilename()
    image_local = Image.open(file_path)
    image_local = image_local.resize((500, 300))

    load_from_url_button.destroy()
    load_from_localdisk_button.destroy()

    show_image(image_local)


def get_image():

    global image_url

    try:
        url = entry.get()
        response = requests.get(url)
        image_url = Image.open(BytesIO(response.content))
        image_url = image_url.resize((500, 300))
        entry.destroy()
        show_image_button.destroy()
        show_image(image_url)

    except:
        print("You entered wrong url, please try again!")
        entry.destroy()
        show_image_button.destroy()
        load_from_url()


def show_image(image):

    global label
    photo = ImageTk.PhotoImage(image)
    label = tk.Label(image=photo)
    label.image = photo
    label.pack(expand=1, anchor='center')

    image = image.resize((32, 32))
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = imagenet_utils.preprocess_input(image)
    image = image.reshape(1, 32, 32, 3)

    global pic
    pic = image

    global classify_button
    classify_button = tk.Button(form, text='Classify Image', command=classify_image, padx=50, pady=10)
    classify_button.configure(background='#364156', foreground='gray', font=('arial', 15, 'bold'))
    classify_button.place(relx=0.35, rely=0.1)


def classify_image():

    classify_button.destroy()
    pred = model.predict_classes([pic])[0]
    sign = classes[pred + 1]

    global class_label
    class_label = tk.Label(form, text=('Predicted Class: ' + sign), padx=50, pady=10)
    class_label.configure(background='#364156', foreground='gray', font=('arial', 15, 'bold'))
    class_label.place(relx=0.3, rely=0.1)

    global try_new_button
    try_new_button = tk.Button(form, text='I would like to try a new one!', command=delete_all, padx=50, pady=10)
    try_new_button.configure(background='#364156', foreground='gray', font=('arial', 15, 'bold'))
    try_new_button.place(relx=0.25, rely=0.9)


def delete_all():

    try_new_button.destroy()
    class_label.destroy()
    label.destroy()
    start()


model = load_model('my_model.h5')


classes = {1: 'Airplane',
            2: 'Automobile',
            3: 'Bird',
            4: 'Cat',
            5: 'Deer',
            6: 'Dog',
            7: 'Frog',
            8: 'Horse',
            9: 'Ship',
            10: 'Truck'}

form = tk.Tk()
form.geometry('800x800')
form.title('CIFAR-10 Dataset Classifier')
form.resizable(False, False)

start()

form.mainloop()
