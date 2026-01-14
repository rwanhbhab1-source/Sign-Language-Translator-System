import customtkinter as ctk
import tkinter
from PIL import Image, ImageTk, ImageDraw, ImageFont, ImageSequence
import requests
from io import BytesIO
import threading
import speech_recognition as sr
import cv2
import mediapipe as mp
import numpy as np
import pickle
import os
import sys
import arabic_reshaper
from bidi.algorithm import get_display
from gtts import gTTS
import pygame
import hashlib
import time 



if not os.path.exists("audio_cache"):
    os.makedirs("audio_cache")
if os.name == 'nt':
    os.system('chcp 65001')
sys.stdout.reconfigure(encoding='utf-8')
ctk.set_appearance_mode("dark")
#ctk.set_default_color_theme("pink.json")
#   print("الملف موجود:", os.path.exists("pink.json"))

mapping = {
      "ام": "https://raw.githubusercontent.com/dali22alk/my-gif/main/mom.gif",
    "الحمد لله": "https://raw.githubusercontent.com/dali22alk/my-gif/main/alhmdullah-ezgif.com-video-to-gif-converter.gif",
    "غضب": "https://raw.githubusercontent.com/dali22alk/my-gif/main/angry.gif",
    "اين المكان": "https://raw.githubusercontent.com/dali22alk/my-gif/main/aynaalmkan.gif",
    "طفل رضيع": "https://raw.githubusercontent.com/dali22alk/my-gif/main/baby.gif",
    "سيء": "https://raw.githubusercontent.com/dali22alk/my-gif/main/bad.gif",
    "اخ": "https://raw.githubusercontent.com/dali22alk/my-gif/main/brother.gif",
    "اب": "https://raw.githubusercontent.com/dali22alk/my-gif/main/dad.gif",
    "العائله": "https://raw.githubusercontent.com/dali22alk/my-gif/main/family.gif",
    "جيد": "https://raw.githubusercontent.com/dali22alk/my-gif/main/good.gif",
    "جد": "https://raw.githubusercontent.com/dali22alk/my-gif/main/grandfather.gif",
    "جده": "https://raw.githubusercontent.com/dali22alk/my-gif/main/grandmom.gif",
    "سعيد": "https://raw.githubusercontent.com/dali22alk/my-gif/main/happy.gif",
    "بيت": "https://raw.githubusercontent.com/dali22alk/my-gif/main/house.gif",
    "كيف حالك": "https://raw.githubusercontent.com/dali22alk/my-gif/main/howareu-ezgif.com-video-to-gif-converter.gif",
    "انا بخير": "https://raw.githubusercontent.com/dali22alk/my-gif/main/imgood-ezgif.com-video-to-gif-converter.gif",
    "قلق": "https://raw.githubusercontent.com/dali22alk/my-gif/main/kalak.gif",
    "لو سمحت": "https://raw.githubusercontent.com/dali22alk/my-gif/main/lawsama7t.gif",
    "يحب": "https://raw.githubusercontent.com/dali22alk/my-gif/main/love.gif", 
    "مبارك": "https://raw.githubusercontent.com/dali22alk/my-gif/main/mabrok.gif",
    "مدرسه": "https://raw.githubusercontent.com/dali22alk/my-gif/main/madrasah.gif",
    "حزين": "https://raw.githubusercontent.com/dali22alk/my-gif/main/sad.gif",
    "السلام عليكم": "https://raw.githubusercontent.com/dali22alk/my-gif/main/salamalaekom-ezgif.com-video-to-gif-converter.gif",
    "شركه": "https://raw.githubusercontent.com/dali22alk/my-gif/main/shareka.gif",
    "اخت": "https://raw.githubusercontent.com/dali22alk/my-gif/main/sister.gif",
    "ابن": "https://raw.githubusercontent.com/dali22alk/my-gif/main/son.gif",
    "اسف": "https://raw.githubusercontent.com/dali22alk/my-gif/main/sorry.gif",
    "جامعه": "https://raw.githubusercontent.com/dali22alk/my-gif/main/uni.gif",
    "صباح الخير": "https://raw.githubusercontent.com/dali22alk/my-gif/main/goodmorning-ezgif.com-video-to-gif-converter.gif",
    "مساء الخير": "https://raw.githubusercontent.com/dali22alk/my-gif/main/goodevening-ezgif.com-video-to-gif-converter.gif",
    "كم الساعه": "https://raw.githubusercontent.com/dali22alk/my-gif/main/alsa3akm.gif"

}

stop_animation = False

def animate(frames, delay, index=0):
    global stop_animation
    if stop_animation:
        return
    frame = frames[index]
    label_gif.config(image=frame)
    label_gif.image = frame
    tabview.after(delay, animate, frames, delay, (index + 1) % len(frames))

def load_gif_from_url(word):
    global stop_animation
    stop_animation = True
    if word in mapping:
        url = mapping[word]
        try:
            response = requests.get(url)
            response.raise_for_status()
            img = Image.open(BytesIO(response.content))
            frames = [ImageTk.PhotoImage(frame.copy().convert('RGBA')) for frame in ImageSequence.Iterator(img)]
            delay = img.info.get('duration', 100)
            stop_animation = False
            animate(frames, delay)
        except Exception as e:
            label_gif.config(text=f"خطأ في التحميل: {e}", image=None)
    else:
        label_gif.config(text="لا يوجد GIF لهذه الكلمة", image=None)

def show_gif_threaded(word):
    label_gif.config(text="...جاري التحميل", image=None)
    threading.Thread(target=load_gif_from_url, args=(word,), daemon=True).start()

def use_entry_word():
    word = entry.get().strip()
    show_gif_threaded(word)

def recognize_speech():
    label_gif.config(text="...جاري التسجيل", image=None)
    r = sr.Recognizer()
    with sr.Microphone() as source:
        try:
            audio = r.listen(source, timeout=5)
            text = r.recognize_google(audio, language='ar-SA')
            entry.delete(0, ctk.END)
            entry.insert(0, text)
            show_gif_threaded(text.strip())
        except sr.UnknownValueError:
            label_gif.config(text="تعذّر فهم الصوت")
        except sr.RequestError:
            label_gif.config(text="تعذر الاتصال بخدمة التحويل")
        except Exception as e:
            label_gif.config(text=f"خطأ: {e}")

labels_dict_numbers = {
   0: 'صفر', 1: 'واحد', 2: 'اثنان', 3: 'ثلاثة', 4: 'أربعة',
    5: 'خمسة', 6: 'ستة', 7: 'سبعة', 8: 'ثمانية', 9: 'تسعة', 10: 'عشرة'
}
labels_dict_letters = {
    0: 'ا', 1: 'ب', 2: 'ت', 3: 'ث', 4: 'ج', 5: 'ح',
    6: 'خ', 7: 'د', 8: 'ذ', 9: 'ر', 10: 'ز', 11: 'س',
    12: 'ش', 13: 'ص', 14: 'ض', 15: 'ط', 16: 'ظ', 17: 'ع',
    18: 'غ', 19: 'ف', 20: 'ق', 21: 'ك', 22: 'ل', 23: 'م',
    24: 'ن', 25: 'ه', 26: 'و', 27: 'ي', 28: 'ال', 29: 'ة'
}
labels_dict_words = {
    0: 'اتمنى لك حياة سعيدة', 1: 'أنت', 2: 'عم د',
    3: 'سيء', 4: 'مرحبا', 5: 'هذا رهيب',
    6: 'أحبك', 7: 'لو سمحت', 8: 'جملة او كلمات',
    9: 'اليوم'
}

class SignToSpeechApp(ctk.CTkFrame):
    def __init__(self, master):
        super().__init__(master)
        self.pack(fill="both", expand=True)

        self.cap = None
        self.camera_label = None
        self.model = None
        self.running = False
        self.labels_dict = {}
        self.speech_enabled = True
        self.text_from_sign = []

        # متغيرات التحكم بالنطق
        self.last_prediction = None
        self.prediction_count = 0
        self.prediction_threshold = 7  # عدد الإطارات لتثبيت الحرف
        self.last_spoken = None
        self.last_spoken_time = 0
        self.min_speak_interval = 2 
         # فاصل زمني بالنطق بالثواني
        self.use_left_hand = False  # افتراضيًا اليد اليمنى

        pygame.mixer.init()

        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)
        self.drawing = mp.solutions.drawing_utils
        self.drawing_styles = mp.solutions.drawing_styles

        sidebar = ctk.CTkFrame(self, width=150)
        sidebar.pack(side="left", fill="y")

        ctk.CTkButton(sidebar,text="الأرقام", command=lambda: self.start_model('D:\\img_num\\model.p', labels_dict_numbers),height=50,width=180, font=ctk.CTkFont(size=18, weight="bold")).pack(pady=15)
        ctk.CTkButton(sidebar,text="الحروف",command=lambda: self.start_model('D:\\img_char\\model.p', labels_dict_letters),height=50,width=180,font=ctk.CTkFont(size=18, weight="bold")).pack(pady=15)
        ctk.CTkButton(sidebar,text="الكلمات",command=lambda: self.start_model('D:\\img_word\\model.p', labels_dict_words), height=50,width=180, font=ctk.CTkFont(size=18, weight="bold")).pack(pady=15)
        ctk.CTkButton( sidebar, text="   الكاميرا ايقاف",command=self.stop_camera,height=50,width=180,font=ctk.CTkFont(size=18, weight="bold")).pack(pady=15)
        ctk.CTkButton( sidebar,text=" الصوت ايقاف/تشغيل",command=self.toggle_speech,height=50,width=180,font=ctk.CTkFont(size=18, weight="bold")).pack(pady=15)
        ctk.CTkButton(sidebar,text="🔄النص مسح", command=self.clear_text, height=50, width=180,font=ctk.CTkFont(size=18, weight="bold")).pack(pady=15)
        self.hand_toggle_button = ctk.CTkButton(sidebar, text="اليمنى يد استخدام", command=self.toggle_hand, height=50,  width=180, font=ctk.CTkFont(size=18, weight="bold"))
        self.hand_toggle_button.pack(pady=15)



        self.content_frame = ctk.CTkFrame(self)
        self.content_frame.pack(side="left", fill="both", expand=True, padx=20, pady=20)

        self.camera_label = ctk.CTkLabel(self.content_frame)
        self.camera_label.pack(pady=10)
        self.spoken_text_display_sign = ctk.CTkLabel(
        self.content_frame,
        text="",
        font=ctk.CTkFont(size=20),
        wraplength=900,
        anchor="e",
        justify="right"
)
        self.spoken_text_display_sign.pack(pady=10)

    def clear_text(self):
     self.text_from_sign.clear()  # تفريغ القائمة
     self.spoken_text_display_sign.configure(text="")  # إفراغ العنصر الظاهر
     self.last_spoken = None  # إعادة تهيئة آخر ما تم نطقه
     self.prediction_count = 0
     self.last_prediction = None

    def toggle_speech(self):
        self.speech_enabled = not self.speech_enabled
    def toggle_hand(self):
     self.use_left_hand = not self.use_left_hand
     if self.use_left_hand:
        self.hand_toggle_button.configure(text="اليسرى يد استخدام")
     else:
        self.hand_toggle_button.configure(text="اليمنى يد استخدام")

    def speak(self, text):
        if not self.speech_enabled:
            return
        try:
            filename = f"audio_cache/{hashlib.md5(text.encode('utf-8')).hexdigest()}.mp3"
            if not os.path.exists(filename):
                tts = gTTS(text=text, lang='ar')
                tts.save(filename)
            pygame.mixer.music.load(filename)
            pygame.mixer.music.play()
            while pygame.mixer.music.get_busy():
                continue
        except Exception as e:
            print(f"خطأ في النطق: {e}")

    def start_model(self, model_path, labels_dict):
        self.labels_dict = labels_dict
        try:
            with open(model_path, 'rb') as f:
                model_dict = pickle.load(f)
                self.model = model_dict['model']
        except Exception as e:
            print(f"Model load error: {e}")
            return

        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            print("فشل في فتح الكاميرا")
            return
        self.running = True
        self.update_frame()

    def update_frame(self):
        if self.running and self.cap:
            ret, frame = self.cap.read()
            if not ret:
                return

            frame = cv2.flip(frame, 1)
            data_aux, x_, y_ = [], [], []
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(frame_rgb)

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    self.drawing.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                    for lm in hand_landmarks.landmark:
                        x_.append(lm.x)
                        y_.append(lm.y)
                    for lm in hand_landmarks.landmark:
                        data_aux.append(lm.x - min(x_))
                        data_aux.append(lm.y - min(y_))

                if self.model and len(data_aux) == 42:
                    # عكس الاحداثيات في حال تم اختيار اليد اليسرى
                    if self.use_left_hand:
                     for i in range(0, len(data_aux), 2):
                      data_aux[i] = 1 - data_aux[i]

                    prediction = self.model.predict([np.asarray(data_aux)])
                    label = self.labels_dict.get(int(prediction[0]), '?')

                    if label == self.last_prediction:
                        self.prediction_count += 1
                    else:
                        self.last_prediction = label
                        self.prediction_count = 1

                    if self.prediction_count >= self.prediction_threshold:
                        current_time = time.time()
                        if current_time - self.last_spoken_time > self.min_speak_interval:
                            if self.last_spoken != label:
                                self.last_spoken = label
                                self.speak(label)
                                self.last_spoken_time = current_time
                                self.text_from_sign.append(label)
                                full_text = " ".join(self.text_from_sign)
                                reshaped_text = arabic_reshaper.reshape(full_text)
                                bidi_text = get_display(reshaped_text)
                                self.spoken_text_display_sign.configure(text=f" :النص \n {bidi_text}")



                        reshaped = arabic_reshaper.reshape(label)
                        label_display = get_display(reshaped)

                        img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                        draw = ImageDraw.Draw(img_pil)
                        try:
                            font = ImageFont.truetype("arial.ttf", 36)
                        except:
                            font = ImageFont.load_default()
                        draw.text((50, 50), label_display, font=font, fill=(0, 255, 0))
                        frame = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            img = img.resize((750, 550))
            imgtk = ImageTk.PhotoImage(image=img)
            self.camera_label.imgtk = imgtk
            self.camera_label.configure(image=imgtk)

            self.after(10, self.update_frame)

    def stop_camera(self):
        self.running = False
        if self.cap:
            self.cap.release()
            self.cap = None
        if self.camera_label:
            self.camera_label.configure(image='')

root = ctk.CTk()
root.geometry("1100x750")
root.title("مترجم ثنائي الاتجاه للغة الإشارة")

tabview = ctk.CTkTabview(root)
tabview.pack(fill="both", expand=True)

tab1 = tabview.add("صوت → إشارة")
entry = ctk.CTkEntry(tab1, width=300)
entry.pack(pady=10)
label_gif = tkinter.Label(tab1)
label_gif.pack(pady=10)

btn_show = ctk.CTkButton(tab1, text="الصورة عرض", command=use_entry_word, height=50, width=180, font=ctk.CTkFont(size=18, weight="bold")); btn_show.pack(pady=10)


btn_speech = ctk.CTkButton(tab1, text="للتسجيل أنقر", command=recognize_speech, height=50, width=180, font=ctk.CTkFont(size=18, weight="bold")); btn_speech.pack(pady=10)


app_tab2 = tabview.add("إشارة → صوت")
sign_to_speech_app = SignToSpeechApp(app_tab2)

root.mainloop()