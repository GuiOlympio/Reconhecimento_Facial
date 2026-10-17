import cv2
import face_recognition
import numpy as np

# Inicializa a câmera
video_capture = cv2.VideoCapture(0)

# 1. Carrega e verifica a imagem do Guilherme
guilherme_img = face_recognition.load_image_file("Guilherme.jpg")
guilherme_encodings = face_recognition.face_encodings(guilherme_img)

if len(guilherme_encodings) > 0:
    guilherme_face_encoding = guilherme_encodings[0]
else:
    print("ERRO: Nenhum rosto foi detectado na imagem do Guilherme!")
    exit()

# 2. Carrega e verifica a imagem da Eloah
eloah_img = face_recognition.load_image_file("helloa.jpg")
eloah_encodings = face_recognition.face_encodings(eloah_img)

if len(eloah_encodings) > 0:
    eloah_face_encoding = eloah_encodings[0]
else:
    print("ERRO: Nenhum rosto foi detectado na imagem da Eloah!")
    exit()

# Listas de encodings e nomes conhecidos
conhece_face_encodings = [guilherme_face_encoding, eloah_face_encoding]
conhece_face_nomes = ["Guilherme", "Eloah"]

face_locations = []
face_encodings = []
face_nomes = []
process_this_frame = True

while True:
    ret, frame = video_capture.read()
    if not ret:
        print("ERRO: Não foi possível acessar a webcam.")
        break

    if process_this_frame:
        # Reduz o frame para 1/4 do tamanho para acelerar o processamento
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        # Converte de BGR (OpenCV) para RGB (face_recognition)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        # Localiza e codifica os rostos no frame atual
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(
            rgb_small_frame, face_locations
        )

        face_nomes = []

        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(
                conhece_face_encodings, face_encoding
            )
            nome = "Desconhecido"

            face_distances = face_recognition.face_distance(
                conhece_face_encodings, face_encoding
            )
            best_match_index = np.argmin(face_distances)

            if matches[best_match_index]:
                nome = conhece_face_nomes[best_match_index]

            face_nomes.append(nome)

    process_this_frame = not process_this_frame

    # Desenha os retângulos e os nomes na tela
    for (top, right, bottom, left), nome in zip(face_locations, face_nomes):
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        # Caixas ao redor do rosto
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
        cv2.rectangle(
            frame,
            (left, bottom - 35),
            (right, bottom),
            (0, 0, 255),
            cv2.FILLED,
        )
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(
            frame, nome, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1
        )

    # Exibe a imagem final na janela
    cv2.imshow("Video", frame)

    # Tecla 'Esc' para sair
    if cv2.waitKey(1) & 0xFF == 27:
        break

video_capture.release()
cv2.destroyAllWindows()