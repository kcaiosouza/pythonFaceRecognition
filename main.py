import face_recognition
import os, sys
import cv2
import numpy as np
import math

def face_confidence(face_distance, face_match_threshold=0.6):
    range = (1.0 - face_match_threshold)
    linear_val = (1.0 - face_distance) / (range * 2)

    if face_distance > face_match_threshold:
        return str(round(linear_val * 100, 2)) + "%"
    else:
        value = (linear_val + ((1.0 - linear_val) * math.pow((linear_val - 0.5) * 2, 0.2))) * 100
        return  str(round(value, 2)) + "%"


class FaceRecognition:
    face_locations = []
    face_encodings = []
    face_names = []
    knows_face_encodings = []
    knows_face_names = []
    process_current_frames = True
    times_seeing_face = 0
    last_person_identified = ""
    verified_person = False
    registration_number = last_person_identified.split(".")

    def __init__(self):
        self.encode_face()

    def encode_face(self):
        for image in os.listdir('faces'):
            face_image = face_recognition.load_image_file(f"faces/{image}")
            face_encoding = face_recognition.face_encodings(face_image)[0]

            self.knows_face_encodings.append(face_encoding)
            self.knows_face_names.append(image)

        print(self.knows_face_names)

    def run_recognition(self):
        video_capture = cv2.VideoCapture(1)
        video_capture.set(3, 1280)
        video_capture.set(4, 720)

        if not video_capture:
            print("Não foi encontrado nenhum dispositivo de captura de imagem")

        while True:
            ret, frame = video_capture.read()

            if self.process_current_frames:
                small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy= 0.25)
                rgb_small_frame = np.ascontiguousarray(small_frame[:, :, ::-1])

                #Localizando todos os rostos no frame

                self.face_locations = face_recognition.face_locations(rgb_small_frame)
                self.face_encodings = face_recognition.face_encodings(rgb_small_frame, self.face_locations, model="small")

                self.face_names = []
                #Esse IF verifica se existe apenas uma pessoa sendo identificada, pois o sistema funciona apenas com uma pessoa por vez
                if(len(self.face_encodings) == 1):

                    for face_encoding in self.face_encodings:
                        matches = face_recognition.compare_faces(self.knows_face_encodings, face_encoding)
                        name = "Desconhecido"
                        confidence = "Desconhecido"

                        face_distances = face_recognition.face_distance(self.knows_face_encodings, face_encoding)
                        best_match_index = np.argmin(face_distances)

                        if (matches[best_match_index]):

                            # Esse IF verifica se a pessoa identificada no frame anterior é a mesma do frame atual, para garantir que a pessoas é a mesma dos 15 frame
                            if (self.last_person_identified == self.knows_face_names[best_match_index]):
                                if(self.times_seeing_face == 14):
                                    self.verified_person = True
                                    print("Verificado, realmente é", self.knows_face_names[best_match_index].split(".")[0])
                                    print("Buscando no banco de dados... mentira, mas é só pra deixar um exemplo")


                            else:
                                print("ops nao e a mesma pessoa")
                                self.times_seeing_face = 0

                            #Esse IF é uma verificação para saber se ele identifica a mesma pessoa por 15 frames, evitando com que ele localize outra pessoa semelhane
                            if(self.times_seeing_face < 15):
                                self.times_seeing_face += 1
                                self.last_person_identified = self.knows_face_names[best_match_index]

                            name = self.knows_face_names[best_match_index]
                            confidence = face_confidence(face_distances[best_match_index])

                        else:
                            if (name == "Desconhecido"):
                                print("Pessoa não encontrada no banco de imagens")

                        self.face_names.append(f"{name} ({confidence})")

            self.process_current_frames = not self.process_current_frames

            #Display annotations
            for(top, right, bottom, left), name in zip(self.face_locations, self.face_names):
                top *= 4
                right *= 4
                bottom *= 4
                left *= 4

                cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
                cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), -1)
                cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 1)
            cv2.namedWindow("Reconhecimento Facial", cv2.WND_PROP_FULLSCREEN)
            cv2.setWindowProperty("Reconhecimento Facial", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            cv2.imshow("Reconhecimento Facial", frame)

            if cv2.waitKey(1) == ord("q"):
                break

        video_capture.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    fr = FaceRecognition()
    fr.run_recognition()
