import cv2
from ultralytics import YOLO


model = YOLO("yolo11n.pt")



def track_specific_object_webcam(model, object_name):
    """
    Traque uniquement un objet spécifique à l'aide d'un modèle YOLO en utilisant la webcam.

    :param model: Le modèle YOLO chargé.
    :param object_name: Nom de l'objet à traquer (comme "person", "car", etc.).
    """

    # Ouvrir la webcam
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Erreur : Impossible d'accéder à la webcam.")
        return

    print(f"Tracking de l'objet '{object_name}' via la webcam. Appuyez sur 'q' pour quitter.")

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
        cv2.imshow("YOLO Object Tracking - Webcam", frame)

        # Quitter avec 'q'
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # Libérer les ressources
    cap.release()
    cv2.destroyAllWindows()


#track_specific_object_webcam(model, 'person')

