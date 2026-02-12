import cv2
import mss
import numpy as np
import ctypes
from ctypes import wintypes

# --- CONFIGURATION ---
WINDOW_TITLE = "LLBlaze"
# Vos coordonnées calibrées (copiées depuis votre message)
# Note: Ces coordonnées sont relatives à la fenêtre du jeu, pas à l'écran global !
HP_ROI_RELATIVE = {'top': 76, 'left': 94, 'width': 190, 'height': 69}

def get_game_window(title):
    user32 = ctypes.windll.user32
    handle = user32.FindWindowW(None, title)
    if not handle: return None
    rect = wintypes.RECT()
    user32.GetClientRect(handle, ctypes.byref(rect))
    point = wintypes.POINT()
    point.x = rect.left; point.y = rect.top
    user32.ClientToScreen(handle, ctypes.byref(point))
    return {"top": point.y, "left": point.x, "width": rect.right - rect.left, "height": rect.bottom - rect.top}

def nothing(x):
    pass

def tuner_mode():
    game_window = get_game_window(WINDOW_TITLE)
    if not game_window:
        print("❌ Jeu introuvable.")
        return

    # Calcul des coordonnées absolues de la barre de vie sur l'écran
    monitor = {
        "top": game_window["top"] + HP_ROI_RELATIVE["top"],
        "left": game_window["left"] + HP_ROI_RELATIVE["left"],
        "width": HP_ROI_RELATIVE["width"],
        "height": HP_ROI_RELATIVE["height"]
    }

    print("🧬 Lancement du calibrage HSV... Ajustez les curseurs !")

    cv2.namedWindow("Tuning Opsines")
    
    # Création des Trackbars pour ajuster le filtre jaune
    # Valeurs par défaut approximatives pour du Jaune
    cv2.createTrackbar("Low H", "Tuning Opsines", 15, 179, nothing) # Teinte min
    cv2.createTrackbar("High H", "Tuning Opsines", 35, 179, nothing)# Teinte max
    cv2.createTrackbar("Low S", "Tuning Opsines", 100, 255, nothing)# Saturation min
    cv2.createTrackbar("High S", "Tuning Opsines", 255, 255, nothing)
    cv2.createTrackbar("Low V", "Tuning Opsines", 100, 255, nothing)# Luminosité min
    cv2.createTrackbar("High V", "Tuning Opsines", 255, 255, nothing)

    sct = mss.mss()

    while True:
        # 1. Capture de la zone de vie uniquement
        img = np.array(sct.grab(monitor))
        
        # On supprime le canal Alpha pour avoir du BGR propre
        img_bgr = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        
        # 2. Conversion en HSV (Espace de couleur perceptuel)
        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

        # 3. Récupération des valeurs des sliders
        l_h = cv2.getTrackbarPos("Low H", "Tuning Opsines")
        h_h = cv2.getTrackbarPos("High H", "Tuning Opsines")
        l_s = cv2.getTrackbarPos("Low S", "Tuning Opsines")
        h_s = cv2.getTrackbarPos("High S", "Tuning Opsines")
        l_v = cv2.getTrackbarPos("Low V", "Tuning Opsines")
        h_v = cv2.getTrackbarPos("High V", "Tuning Opsines")

        lower_yellow = np.array([l_h, l_s, l_v])
        upper_yellow = np.array([h_h, h_s, h_v])

        # 4. Création du Masque (Thresholding)
        mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

        # 5. Calcul du pourcentage de pixels blancs (Vie restante)
        # On compte les pixels non-zéro
        pixels_total = monitor["width"] * monitor["height"]
        pixels_vie = cv2.countNonZero(mask)
        
        # Note: Ceci est une estimation brute, on affinera plus tard
        ratio = pixels_vie / pixels_total

        # Affichage
        result = cv2.bitwise_and(img_bgr, img_bgr, mask=mask)
        
        # Texte d'info
        cv2.putText(result, f"HP Signal: {pixels_vie} px", (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)
        
        # Stack des images pour comparaison : Original | Masque | Résultat
        # On convertit le masque en 3 canaux pour l'empiler
        mask_3d = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        stacked = np.hstack((img_bgr, mask_3d, result))

        cv2.imshow("Tuning Opsines", stacked)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print(f"\n✅ VALEURS FINALES :")
            print(f"LOWER_YELLOW = np.array([{l_h}, {l_s}, {l_v}])")
            print(f"UPPER_YELLOW = np.array([{h_h}, {h_s}, {h_v}])")
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    tuner_mode()