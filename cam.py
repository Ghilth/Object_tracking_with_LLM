import cv2
from ultralytics import YOLO

# Charger le modèle YOLO
model = YOLO("yolo11n.pt")

# Ouvrir la webcam (indice 0 pour la webcam par défaut)
cap = cv2.VideoCapture(0)

# Vérifier si la webcam est accessible
if not cap.isOpened():
    print("Erreur : Impossible d'accéder à la webcam.")
    exit()

# Boucle principale pour traiter les trames de la webcam
while cap.isOpened():
    # Lire une trame depuis la webcam
    success, frame = cap.read()

    if success:
        # Appliquer YOLO pour le suivi des objets
        results = model.track(frame, persist=True)

        # Visualiser les résultats annotés sur la trame
        annotated_frame = results[0].plot()

        # Afficher la trame annotée
        cv2.imshow("YOLO11 Tracking - Webcam", annotated_frame)

        # Interrompre la boucle si 'q' est pressé
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        print("Erreur lors de la lecture de la trame.")
        break

# Libérer les ressources
cap.release()
cv2.destroyAllWindows()
