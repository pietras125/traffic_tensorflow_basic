import numpy as np
import tkinter as tk
from PIL import Image, ImageTk
import datetime
import os
import tensorflow as tf
import pathlib
import cv2
import time
import ctypes
from IPython.display import display
from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

class LiczAuta:
    def __init__(self):
        ctypes.windll.shcore.SetProcessDpiAwareness(1)
        self.video_capture = cv2.VideoCapture(0)
        self.video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        ############# MODEL ##############################################################
        #self.model_name = 'centernet_resnet50_v1_fpn_512x512_coco17_tpu-8' #mało samochodów - szybki
        self.model_name = 'efficientdet_d1_coco17_tpu-32' #dużo samochodów, wolny
        ###################################################################################
        self.category_index = label_map_util.create_category_index_from_labelmap('models/research/object_detection/data/mscoco_label_map.pbtxt', use_display_name=True)

        # Okno główne
        self.window = tk.Tk()
        self.window.title("TYTUŁ")
        #LEWA RAMKA
        self.lewa_ramka = tk.Frame(self.window)
        self.lewa_ramka.pack(side = tk.LEFT)
        self.video_label = tk.Label(self.lewa_ramka)
        self.video_label.pack()
        #PRAWA RAMKA
        self.prawa_ramka = tk.Frame(self.window)
        self.prawa_ramka.pack(side = tk.RIGHT)
        #self.quit_button = tk.Button(self.window, text="Quit", command=self.window.destroy)
        #self.quit_button.pack()
        self.quit_button = tk.Button(self.window, text="rozszerz szerokosc", command=lambda : self.rozszerz_obszar_wykrywania("szerokosc","rozszerz"))
        self.quit_button.pack(fill=tk.BOTH, padx=10, pady=10)
        self.quit_button = tk.Button(self.window, text="zwez szerokosc", command=lambda : self.rozszerz_obszar_wykrywania("szerokosc","zwez"))
        self.quit_button.pack(fill=tk.BOTH, padx=10)
        self.quit_button = tk.Button(self.window, text="rozszerz wysokosc", command=lambda : self.rozszerz_obszar_wykrywania("wysokosc","rozszerz"))
        self.quit_button.pack(fill=tk.BOTH, padx=10)
        self.quit_button = tk.Button(self.window, text="zwez wysokosc", command=lambda : self.rozszerz_obszar_wykrywania("wysokosc","zwez"))
        self.quit_button.pack(fill=tk.BOTH, padx=10)

        #obszar wykrywania (prostokąt)
        self.obszar_wykrywania_x_min = 300
        self.obszar_wykrywania_y_min = 277
        self.obszar_wykrywania_x_max = 355
        self.obszar_wykrywania_y_max = 360
        self.obszar_wykrywania = [self.obszar_wykrywania_x_min,self.obszar_wykrywania_y_min,self.obszar_wykrywania_x_max,self.obszar_wykrywania_y_max]
        self.detection_model = self.load_model(self.model_name)
        self.suma_zliczonych_obiektow = 0
        self.obiektow_w_obszarze_poprzednia_klatka = 0
        self.czas_start = time.time()
        #start pierwszej klatki i potem rekurencyjnie reszta
        self.start_rozpoznawania_nowy()

    def load_model(self, model_name):
        base_url = 'http://download.tensorflow.org/models/object_detection/tf2/20200711/'
        model_file = model_name + '.tar.gz'
        model_dir = tf.keras.utils.get_file(
            fname=model_name, 
            origin=base_url + model_file,
            untar=True)
        model_dir = pathlib.Path(model_dir)/"saved_model"
        model = tf.saved_model.load(str(model_dir))
        return model
 
    def run_inference_for_single_image(self, model, image):
        image = np.asarray(image)
        # The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
        input_tensor = tf.convert_to_tensor(image)
        # The model expects a batch of images, so add an axis with `tf.newaxis`.
        input_tensor = input_tensor[tf.newaxis,...]
        
        # Run inference
        model_fn = model.signatures['serving_default']
        output_dict = model_fn(input_tensor)
        
        # All outputs are batches tensors.
        # Convert to numpy arrays, and take index [0] to remove the batch dimension.
        # We're only interested in the first num_detections.
        num_detections = int(output_dict.pop('num_detections'))
        output_dict = {key:value[0, :num_detections].numpy() for key,value in output_dict.items()}
        output_dict['num_detections'] = num_detections
        
        # detection_classes should be ints.
        output_dict['detection_classes'] = output_dict['detection_classes'].astype(np.int64)
            
        # Handle models with masks:
        if 'detection_masks' in output_dict:
            # Reframe the the bbox mask to the image size.
            detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
                    output_dict['detection_masks'], output_dict['detection_boxes'],
                    image.shape[0], image.shape[1])      
            detection_masks_reframed = tf.cast(detection_masks_reframed > 0.5,
                                            tf.uint8)
            output_dict['detection_masks_reframed'] = detection_masks_reframed.numpy()
        return output_dict

    def show_inference(self, model, frame):
        #take the frame from webcam feed and convert that to array
        image_np = np.array(frame)
        # Actual detection.
        output_dict = self.run_inference_for_single_image(model, image_np)
        # Visualization of the results of a detection.
        vis_util.visualize_boxes_and_labels_on_image_array(
            image_np,
            output_dict['detection_boxes'],
            output_dict['detection_classes'],
            output_dict['detection_scores'],
            self.category_index,
            instance_masks=output_dict.get('detection_masks_reframed', None),
            use_normalized_coordinates=True,
            line_thickness=5)
        
        return(image_np)

    def start_rozpoznawania_ORYGINAL(self):
        detection_model = self.load_model(self.model_name)
        czas_start = time.time()
        while True:
            try:
                re,frame = self.video_capture.read()
                Imagenp=self.show_inference(detection_model, frame)
                #cv2.imshow('object detection', cv2.resize(Imagenp, (800,600)))
                cv2.imshow('object detection', Imagenp)
                czas_stop = time.time()
                print ("\033[A                             \033[A")
                print(f'KLATEK/SEKUNDĘ: {int(1/(czas_stop-czas_start))}')
                czas_start = czas_stop
                #cv2.waitKey(10)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            except Exception as e:
                print(str(e))
                break
        self.video_capture.release()
        cv2.destroyAllWindows()

    def start_rozpoznawania_nowy(self):
        try:
            obiektow_w_obszarze_biezaca_klatka = 0
            zapisz_klatke = 0
            #odczytaj klatkę
            _,odczytana_klatka = self.video_capture.read()
            #przekonwertuj klatkę na tablicę numpy
            klatka_tablica_numpy = np.array(odczytana_klatka)#[self.obszar_wykr_y_start:self.obszar_wykr_y_stop,self.obszar_wykr_x_start:self.obszar_wykr_x_stop,:]
            #przeprowadź wykrywanie na klatce
            output_dict = self.run_inference_for_single_image(self.detection_model, klatka_tablica_numpy)
            prostokaty = output_dict['detection_boxes']
            pewnosci = output_dict['detection_scores']
            klasy = output_dict['detection_classes']
            #sprawdź ile jest prostokątów
            max_prostokaty_do_narysowania = prostokaty.shape[0]
            #od jakiej pewności pokazywać prostokąty
            min_pewnosc=.5
            #pętla przez wszystkie prostokaty
            lista_prostokatow = []
            for i in range(min(max_prostokaty_do_narysowania, prostokaty.shape[0])):
                #rysuj i bierz pod uwagę tylko obiekty z pewnością większą niż nastawiona
                if pewnosci[i] > min_pewnosc:
                    #tylko samochody (numer 3)
                    if klasy[i] == 3:
                        y_min, x_min, y_max, x_max = prostokaty[i] 
                        y_min = int(y_min*klatka_tablica_numpy.shape[0])
                        x_min = int(x_min*klatka_tablica_numpy.shape[1])
                        y_max = int(y_max*klatka_tablica_numpy.shape[0])
                        x_max = int(x_max*klatka_tablica_numpy.shape[1])
                        #lista prostokątów w zapisie pixeli (a nie względnym)
                        lista_prostokatow.append([y_min,x_min,y_max,x_max])
            #jeśli są wykryte jakieś elementy
            if lista_prostokatow != []:
                #pętla przez wszystkie 
                for prostokat in lista_prostokatow:
                    #WYŚWIETLANIE PROSTOKĄTÓW
                    #prostokat = cv2.rectangle(image_np, (prostokat[1], prostokat[0]), (prostokat[3],prostokat[2]), (0,0,255))
                    #klatka = cv2.addWeighted(image_np, 1.0, prostokat, 0.5, 1)
                    #WYŚWIETLANIE ŚRODKÓW PROSTOKĄTÓW (X,Y)
                    srodek_prostokata = (int(prostokat[1]+(prostokat[3] - prostokat[1])/2), int(prostokat[0]+(prostokat[2] - prostokat[0])/2),)
                    punkt = cv2.circle(klatka_tablica_numpy, srodek_prostokata, 9, (0,0,255), -4)
                    odczytana_klatka = cv2.addWeighted(klatka_tablica_numpy, 1.0, punkt, 0.5, 1)
                    #sprawdź czy prostokąt jest w obszarze wykrywania
                    if srodek_prostokata[0] > self.obszar_wykrywania_x_min and srodek_prostokata[0] < self.obszar_wykrywania_x_max and srodek_prostokata[1] > self.obszar_wykrywania_y_min and srodek_prostokata[1] < self.obszar_wykrywania_y_max:
                        obiektow_w_obszarze_biezaca_klatka += 1
                #sprawdź czy przybył jakiś obiekt
                if obiektow_w_obszarze_biezaca_klatka > self.obiektow_w_obszarze_poprzednia_klatka:
                    self.suma_zliczonych_obiektow += obiektow_w_obszarze_biezaca_klatka - self.obiektow_w_obszarze_poprzednia_klatka
                    zapisz_klatke = 1

            #generowanie prostokąta wykrywania lub punktu środkowego prostokąta
            prostokat_aktywnego_wykrywania = np.zeros(odczytana_klatka.shape, np.uint8)
            cv2.rectangle(prostokat_aktywnego_wykrywania, (self.obszar_wykrywania_x_min, self.obszar_wykrywania_y_min), (self.obszar_wykrywania_x_max, self.obszar_wykrywania_y_max), (255,0,0), cv2.FILLED)
            #nałożenie na klatkę źródłową  obszaru wykrywania  utworoznego z zer numpy z przezroczystością 0.25
            klatka_tablica_numpy = cv2.addWeighted(klatka_tablica_numpy, 1.0, prostokat_aktywnego_wykrywania, 0.25, 1)
            
            #dodanie tekstu z sumą aut
            #klatka_tablica_numpy = cv2.putText(klatka_tablica_numpy,'Suma=' + str(self.suma_zliczonych_obiektow),(50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3, cv2.LINE_AA)
            #statystyki FPS
            czas_stop = time.time()
            #dodanie tekstu z liczbą FPS
            #klatka_tablica_numpy = cv2.putText(klatka_tablica_numpy,'FPS=' + str(int(1/(czas_stop-self.czas_start))),(50,100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3, cv2.LINE_AA)
            
            self.czas_start = czas_stop
            
            #pokaż klatkę w tkinter
            b,g,r = cv2.split(klatka_tablica_numpy)
            image = cv2.merge((r,g,b))
            image = Image.fromarray(image)
            image = ImageTk.PhotoImage(image)
            self.video_label.config(image=image)
            self.video_label.image = image

            #czy zapisać klatkę z wykrytym obiektem
            if zapisz_klatke == 1:
                nazwa_klatki = datetime.datetime.now().strftime("%Y_%m_%d [%H_%M_%S]")
                cv2.imwrite(f"zdjecia\\{str(nazwa_klatki)}.jpg",klatka_tablica_numpy)
            
            #przepisanie stanu poprzedniej klatki
            self.obiektow_w_obszarze_poprzednia_klatka = obiektow_w_obszarze_biezaca_klatka

        except Exception as e:
            print(str(e))

        #przetworzenie kolejnej klatki
        self.window.after(5, self.start_rozpoznawania_nowy)

    def rozszerz_obszar_wykrywania(self, ktory_wymiar, rozszerz_zwez):
        if ktory_wymiar == "szerokosc":
            if rozszerz_zwez == "rozszerz":
                self.obszar_wykrywania_x_min = self.obszar_wykrywania_x_min - 2
                self.obszar_wykrywania_x_max = self.obszar_wykrywania_x_max + 2
            else:
                self.obszar_wykrywania_x_min = self.obszar_wykrywania_x_min + 2
                self.obszar_wykrywania_x_max = self.obszar_wykrywania_x_max - 2
        else:
            if rozszerz_zwez == "rozszerz":
                self.obszar_wykrywania_y_min = self.obszar_wykrywania_y_min - 2
                self.obszar_wykrywania_y_max = self.obszar_wykrywania_y_max + 2
            else:
                self.obszar_wykrywania_y_min = self.obszar_wykrywania_y_min + 2
                self.obszar_wykrywania_y_max = self.obszar_wykrywania_y_max - 2


if __name__ == "__main__":
    liczauta = LiczAuta()
    tk.mainloop()
    #liczauta.start_rozpoznawania_nowy()
    #liczauta.start_rozpoznawania_ORYGINAL()
    