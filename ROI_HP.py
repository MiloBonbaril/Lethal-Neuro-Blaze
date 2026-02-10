import cv2
import mss
import numpy as np
import ctypes
from ctypes import wintypes
import config

# --- VOTRE FONCTION DE DÉTECTION (Je la réutilise, elle est parfaite) ---
def get_game_window(title):
    user32 = ctypes.windll.user32
    handle = user32.FindWindowW(None, title)
    if not handle:
        return None
    rect = wintypes.RECT()
    user32.GetClientRect(handle, ctypes.byref(rect))
    point = wintypes.POINT()
    point.x = rect.left
    point.y = rect.top
    user32.ClientToScreen(handle, ctypes.byref(point))
    return {
        "top": point.y,
        "left": point.x,
        "width": rect.right - rect.left,
        "height": rect.bottom - rect.top
    }

def calibrate_pain_receptor():
    window_title = config.WINDOW_TITLE # Assurez-vous que c'est le bon titre
    game_window = get_game_window(window_title)
    
    if not game_window:
        print("❌ Fenêtre du jeu introuvable. Lancez le jeu d'abord !")
        return

    print("🧪 Initialisation du calibrage de l'Amygdale...")
    
    with mss.mss() as sct:
        # Capture une frame unique pour la calibration
        screenshot = sct.grab(game_window)
        img = np.array(screenshot)
        # Conversion BGRA -> BGR pour OpenCV (nécessaire pour selectROI)
        img_bgr = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

    print("\n--- INSTRUCTIONS CHIRURGICALES ---")
    print("1. Une fenêtre va s'ouvrir avec la capture du jeu.")
    print("2. Utilisez la souris pour dessiner un rectangle STRICTEMENT autour de la barre de vie (Jauge pleine).")
    print("3. Appuyez sur [ESPACE] ou [ENTRÉE] pour valider.")
    print("4. Appuyez sur [c] pour annuler.")
    
    # Outil de sélection de ROI (Region of Interest) natif d'OpenCV
    # Note: Cela peut figer un instant, c'est normal, c'est bloquant.
    roi = cv2.selectROI("Calibration de la Douleur", img_bgr, showCrosshair=True, fromCenter=False)
    cv2.destroyAllWindows()

    # roi est un tuple (x, y, w, h)
    if roi == (0, 0, 0, 0):
        print("❌ Calibration annulée.")
    else:
        x, y, w, h = roi
        print("\n✅ CALIBRAGE RÉUSSI !")
        print(f"Coordonnées relatives détectées (à copier-coller dans le code final) :")
        print(f"HP_ROI = {{'top': {y}, 'left': {x}, 'width': {w}, 'height': {h}}}")
        
        # Petit test de vérification visuelle
        roi_cropped = img_bgr[int(y):int(y+h), int(x):int(x+w)]
        cv2.imshow("Zone Surveillee (Amygdale)", roi_cropped)
        print("Appuyez sur une touche pour fermer la vérification...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == "__main__":
    calibrate_pain_receptor()