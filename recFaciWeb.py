import face_recognition
from flask import Flask, jsonify, request, redirect

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

app = Flask(__name__)

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def upload_image():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)

        file = request.files['file']

        if file.filename == '':
            return redirect(request.url)

        if file and allowed_file(file.filename):
            return detect_faces_in_image(file)

    return '''
        <!doctype html>
        <title>Reconhecimento Facial</title>
        <h1>Envie uma foto para ver quem é!</h1>
        <form method="POST" enctype="multipart/form-data">
          <input type="file" name="file">
          <input type="submit" value="Enviar">
        </form>
        '''

# --- CARREGAMENTO DAS FOTOS DE REFERÊNCIA ---
print("Carregando fotos de referência...")

# Guilherme
guilherme_img = face_recognition.load_image_file(r"C:\Users\gingu\Dropbox\PC\Downloads\ProgramsProgramação\reconhecimento facial\1791145635474.jpg")
guilherme_encodings = face_recognition.face_encodings(guilherme_img)
guilherme_img_encoding = guilherme_encodings[0] if len(guilherme_encodings) > 0 else None

# Eloah
eloah_img = face_recognition.load_image_file(r"C:\Users\gingu\Dropbox\PC\Downloads\ProgramsProgramação\reconhecimento facial\drive-download-20261004T193336Z-1-001\IMG_20260929_135001_562.jpg")
eloah_encodings = face_recognition.face_encodings(eloah_img)
eloah_img_encoding = eloah_encodings[0] if len(eloah_encodings) > 0 else None

# Lista de rostos conhecidos e seus respetivos nomes
known_face_encodings = []
known_face_names = []

if guilherme_img_encoding is not None:
    known_face_encodings.append(guilherme_img_encoding)
    known_face_names.append("Guilherme")

if eloah_img_encoding is not None:
    known_face_encodings.append(eloah_img_encoding)
    known_face_names.append("Eloah")

print("Fotos carregadas com sucesso!")


def detect_faces_in_image(file_stream):
    # Carrega a imagem enviada
    img = face_recognition.load_image_file(file_stream)
    unknown_face_encodings = face_recognition.face_encodings(img)
    
    face_found = False
    person_identified = "Desconhecido"

    if len(unknown_face_encodings) > 0:
        face_found = True
        
        # Compara o primeiro rosto da foto enviada com a lista de rostos conhecidos
        matches = face_recognition.compare_faces(known_face_encodings, unknown_face_encodings[0])
        
        # Verifica se houve alguma correspondência
        for index, match in enumerate(matches):
            if match:
                person_identified = known_face_names[index]
                break

    # Retorna o resultado no formato JSON
    result = {
        "rosto_encontrado": face_found,
        "pessoa_identificada": person_identified
    }
    return jsonify(result)


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5001, debug=True, use_reloader=False)