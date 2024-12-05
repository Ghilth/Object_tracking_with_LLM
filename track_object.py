import cv2
from ultralytics import YOLO


def track_specific_object_stream(model, object_name, stream_url):
    """
    Traque uniquement un objet spécifique à l'aide d'un modèle YOLO en utilisant un flux vidéo via URL.

    :param model: Le modèle YOLO chargé.
    :param object_name: Nom de l'objet à traquer (comme "person", "car", etc.).
    :param stream_url: URL du flux vidéo (RTSP, HTTP, etc.).
    """

    # Ouvrir le flux vidéo depuis l'URL
    cap = cv2.VideoCapture(stream_url,cv2.CAP_FFMPEG)
    #cap = cv2.VideoCapture(0)


    if not cap.isOpened():
        print("Connexion lente, nouvel essai...")
        import time
        time.sleep(10)  # Attendre 5 secondes
        cap.open(stream_url) 
        #print("Erreur : Impossible d'accéder au flux vidéo.")
        return

    print(f"Tracking de l'objet '{object_name}' depuis le flux vidéo. Appuyez sur 'q' pour quitter.")

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Erreur lors de la lecture de la trame.")
            break

        # Appliquer YOLO pour le tracking
        results = model.track(frame, persist=True)

        # Traiter chaque résultat détecté
        for result in results:
            for box in result.boxes:
                # Récupérer les informations de la boîte
                x_min, y_min, x_max, y_max = map(int, box.xyxy[0])  # Coordonnées
                class_id = int(box.cls[0])  # ID de la classe
                confidence = float(box.conf[0])  # Score de confiance

                # Vérifier si l'objet correspond à celui spécifié
                detected_object_name = model.names[class_id]
                if detected_object_name == object_name:
                    # Dessiner la boîte et ajouter une étiquette
                    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                    label = f"{detected_object_name} ({confidence:.2f})"
                    cv2.putText(frame, label, (x_min, y_min - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Afficher la trame annotée
        cv2.imshow("YOLO Object Tracking - Stream", frame)

        # Quitter avec 'q'
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # Libérer les ressources
    cap.release()
    cv2.destroyAllWindows()


# Charger le modèle YOLO
#model = YOLO("yolo11n.pt")
model = YOLO("yolov8n.pt")

# URL du flux vidéo (par exemple, RTSP ou HTTP)
stream_url = "http://192.168.218.129:8080/video"  # Remplacez par l'URL de votre flux vidéo

# Traquer un objet spécifique, par exemple "person"
track_specific_object_stream(model, 'laptop', stream_url)
