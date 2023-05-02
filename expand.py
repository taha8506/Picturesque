from __future__ import division, print_function
import argparse
import os
import numpy as np
import torch
import cv2

import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk

from smooth import smoothen_luminance
from model import ExpandNet
from util import (
    process_path,
    split_path,
    map_range,
    str2bool,
    cv2torch,
    torch2cv,
    resize,
    tone_map,
    create_tmo_param_from_args,
)


def get_args():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    # arg('ldr', nargs='+', type=process_path, help='Ldr image(s)')
    arg('ldr', nargs='*', type=process_path, default=[], help='Ldr image(s)')  
    arg(
        '--out',
        type=lambda x: process_path(x, True),
        default=None,
        help='Output location.',
    )
    arg(
        '--video',
        type=str2bool,
        default=False,
        help='Whether input is a video.',
    )
    arg(
        '--patch_size',
        type=int,
        default=256,
        help='Patch size (to limit memory use).',
    )
    arg('--resize', type=str2bool, default=False, help='Use resized input.')
    arg(
        '--use_exr',
        type=str2bool,
        default=False,
        help='Produce .EXR instead of .HDR files.',
    )
    arg('--width', type=int, default=960, help='Image width resizing.')
    arg('--height', type=int, default=540, help='Image height resizing.')
    arg('--tag', default=None, help='Tag for outputs.')
    arg(
        '--use_gpu',
        type=str2bool,
        default=torch.cuda.is_available(),
        help='Use GPU for prediction.',
    )
    arg(
        '--tone_map',
        choices=['exposure', 'reinhard', 'mantiuk', 'drago', 'durand'],
        default=None,
        help='Tone Map resulting HDR image.',
    )
    arg(
        '--stops',
        type=float,
        default=0.0,
        help='Stops (loosely defined here) for exposure tone mapping.',
    )
    arg(
        '--gamma',
        type=float,
        default=1.0,
        help='Gamma curve value (if tone mapping).',
    )
    arg(
        '--use_weights',
        type=process_path,
        default='weights.pth',
        help='Weights to use for prediction',
    )
    arg(
        '--ldr_extensions',
        nargs='+',
        type=str,
        default=['.jpg', '.jpeg', '.tiff', '.bmp', '.png'],
        help='Allowed LDR image extensions',
    )
    opt = parser.parse_args()
    return opt


def load_pretrained(opt):
    net = ExpandNet()
    net.load_state_dict(
        torch.load(opt.use_weights, map_location=lambda s, l: s)
    )
    net.eval()
    return net


def preprocess(x, opt):
    x = x.astype('float32')
    if opt.resize:
        x = resize(x, size=(opt.width, opt.height))
    x = map_range(x)
    return x


def create_name(inp, tag, ext, out, extra_tag):
    root, name, _ = split_path(inp)
    if extra_tag is not None:
        tag = '{0}_{1}'.format(tag, extra_tag)
    if out is not None:
        root = out
    return os.path.join(root, '{0}_{1}.{2}'.format(name, tag, ext))


def convert_to_jpeg(ldr_image, output_path):
    ldr_image_8bit = np.clip(ldr_image * 255, 0, 255).astype('uint8')
    cv2.imwrite(output_path, ldr_image_8bit)

def create_images(opt, tone_map_algorithm):
    #  preprocess = create_preprocess(opt)

    output_folder = 'output'
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    net = load_pretrained(opt)
    if (len(opt.ldr) == 1) and os.path.isdir(opt.ldr[0]):
        opt.ldr = [
            os.path.join(opt.ldr[0], f)
            for f in os.listdir(opt.ldr[0])
            if any(f.lower().endswith(x) for x in opt.ldr_extensions)
        ]
    for ldr_file in opt.ldr:
        loaded = cv2.imread(
            ldr_file, flags=cv2.IMREAD_ANYDEPTH + cv2.IMREAD_COLOR
        )
        if loaded is None:
            print('Could not load {0}'.format(ldr_file))
            continue
        ldr_input = preprocess(loaded, opt)
        if opt.resize:
            out_name = create_name(
                ldr_file, 'resized', 'jpg', opt.out, opt.tag
            )
            cv2.imwrite(out_name, (ldr_input * 255).astype(int))

        t_input = cv2torch(ldr_input)
        if opt.use_gpu:
            net.cuda()
            t_input = t_input.cuda()
        prediction = map_range(
            torch2cv(net.predict(t_input, opt.patch_size).cpu()), 0, 1
        )

        extension = 'exr' if opt.use_exr else 'hdr'
        out_name = create_name(
            ldr_file, 'prediction', extension, opt.out, opt.tag
        )
        print(f'Writing {out_name}')
        cv2.imwrite(out_name, prediction)
        if opt.tone_map is not None:
            tmo_img = tone_map(
                prediction, opt.tone_map, **create_tmo_param_from_args(opt)
            )
            out_name = create_name(
                ldr_file,
                'prediction_{0}'.format(opt.tone_map),
                'jpg',
                opt.out,
                opt.tag,
            )
            cv2.imwrite(out_name, (tmo_img * 255).astype(int))

        # Tone map HDR to JPEG using Mantiuk algorithm
        hdr_image = cv2.imread(out_name, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_COLOR)
        gamma = 2.2
        scale = 0.9
        saturation = 1.2
        ldr_image = tone_map(hdr_image, tone_map_algorithm, gamma=gamma, scale=scale, saturation=saturation, intensity=0.0, light_adapt=1.0, color_adapt=0.0, bias=0.85, contrast=4.0, sigma_space=2.0, sigma_color=2.0)
        input_file_name, _ = os.path.splitext(os.path.basename(ldr_file))
        jpeg_output_path = os.path.join(output_folder, f"{input_file_name}_{tone_map_var.get()}.jpeg")
        convert_to_jpeg(ldr_image, jpeg_output_path)


def main():
    opt = get_args()
    if opt.video:
        create_video(opt)
  
    # else:
    #     create_images(opt)


if __name__ == '__main__':
    main()

def browse_file():
    global file_path
    file_path = filedialog.askopenfilename()
    if file_path.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff')):
        input_path_label.config(text=file_path)
        display_image(file_path, input_image_label)
    else:
        print("Invalid file type. Please select an image file.")


def process_file():
    if file_path:
        opt = get_args()
        opt.ldr = [file_path]
        output_file = create_images(opt, tone_map_var.get())  # Pass the selected tone mapping algorithm

        # Display the output image
        output_folder = 'output'
        input_file_name, _ = os.path.splitext(os.path.basename(file_path))
        output_image_path = os.path.join(output_folder, f"{input_file_name}_{tone_map_var.get()}.jpeg")
        display_image(output_image_path, output_image_label)
    else:
        print("No file selected.")


def display_image(image_path, image_label):
    if image_path is not None:
        image = Image.open(image_path)
        image.thumbnail((500, 500))
        photo = ImageTk.PhotoImage(image)
        image_label.config(image=photo)
        image_label.image = photo


root = tk.Tk()
root.title("LDR to HDR and JPEG Converter")
root.geometry("1200x800")

file_path = ""

bg_image = Image.open("assets/bgh.png").resize((1200, 800), Image.LANCZOS)
bg_photo = ImageTk.PhotoImage(bg_image)
bg_label = tk.Label(root, image=bg_photo)
bg_label.place(x=0, y=0, relwidth=1, relheight=1)

# Create UI elements
browse_button = tk.Button(root, text="Browse", command=browse_file, borderwidth=5,height=2, width=20,)
# browse_button.grid(row=0, column=0, padx=5, pady=5)
browse_button.place(relx=0.5,rely=0.3,anchor='center')

input_path_label = tk.Label(root, text="No file selected", wraplength=250)
input_path_label.grid(row=0, column=1, padx=5, pady=5)


input_image_label = tk.Label(root)
input_image_label.grid(row=1, column=0, padx=5, pady=50)
input_image_label.place(relx=0.28,rely=0.6,anchor='center')

output_image_label = tk.Label(root)
output_image_label.grid(row=1, column=1, padx=5, pady=50)
output_image_label.place(relx=0.72,rely=0.6,anchor='center')

process_button = tk.Button(root, text="Convert", command=process_file, borderwidth=5, width=10)
# process_button.grid(row=2, column=0, columnspan=2, padx=5, pady=5)
process_button.place(relx=0.5, rely=0.95, anchor='center')

# Create a variable to store the selected tone mapping algorithm
tone_map_var = tk.StringVar(root)
tone_map_var.set("mantiuk")  # Set the default value

# Create an OptionMenu widget to let users choose the tone mapping algorithm
tone_map_label = tk.Label(root, text="Tone Map Algorithm:", borderwidth=5, width=20,height=1)
tone_map_label.grid(row=3, column=0, padx=5, pady=5)
tone_map_label.place(relx=0.8,rely=0.9,anchor='center')

tone_map_option_menu = tk.OptionMenu(root, tone_map_var, "reinhard", "mantiuk", "drago")
tone_map_option_menu.grid(row=3, column=1, padx=5, pady=5)
tone_map_option_menu.place(relx=0.9,rely=0.9,anchor='center')


root.mainloop()
