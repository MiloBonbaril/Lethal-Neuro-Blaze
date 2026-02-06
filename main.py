import numpy as np
import cv2
import mss
import time
import ctypes
from ctypes import wintypes
from collections import deque

# --- CONFIGURATION GÉNÉTIQUE ---
WINDOW_TITLE = "LLBlaze"
# Vos valeurs calibrées
HP_ROI_RELATIVE = {'top': 76, 'left': 94, 'width': 190, 'height': 69}
LOWER_YELLOW = np.array([27, 158, 151])
UPPER_YELLOW = np.array([38, 246, 250])

# --- FONCTIONS UTILITAIRES ---
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

# --- CLASSE 1 : LE BIO-MONITEUR (Gestion de la Santé/Récompense) ---
class BioMonitor:
    def __init__(self, game_window_abs):
        """
        Gère la détection de la vie et le calcul de la récompense.
        """
        # Calcul des coordonnées absolues de la ROI de vie
        self.monitor = {
            "top": game_window_abs["top"] + HP_ROI_RELATIVE["top"],
            "left": game_window_abs["left"] + HP_ROI_RELATIVE["left"],
            "width": HP_ROI_RELATIVE["width"],
            "height": HP_ROI_RELATIVE["height"]
        }
        
        self.sct = mss.mss()
        
        # Buffer pour lisser le signal (Anti-Clignotement et Anti-VFX)
        # 15 frames à ~30fps = 0.5 secondes de mémoire tampon
        self.hp_buffer = deque(maxlen=15)
        
        # On stocke le nombre max de pixels possibles pour normaliser (0.0 à 1.0)
        self.max_pixels = 900 # on Utilise 900 car c'est le nombre de pixels de la barre de vie. Sinon on devrait utiliser: self.monitor["width"] * self.monitor["height"]

    def read_hp(self):
        """
        Retourne le niveau de vie normalisé (0.0 à 1.0)
        Utilise un Max-Pooling temporel pour filtrer le bruit.
        """
        # 1. Capture & Traitement
        img = np.array(self.sct.grab(self.monitor))
        img_bgr = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        
        # 2. Masquage
        mask = cv2.inRange(hsv, LOWER_YELLOW, UPPER_YELLOW)
        
        # 3. Comptage brut
        current_pixels = cv2.countNonZero(mask)
        
        # 4. Intégration dans le buffer temporel
        self.hp_buffer.append(current_pixels)
        
        # 5. Filtrage (On prend le max des X dernières frames pour ignorer le clignotement)
        smoothed_pixels = max(self.hp_buffer)
        
        # 6. Normalisation (Ratio entre 0 et 1)
        # Note : On pourrait diviser par smoothed_pixels max observé au début de la partie 
        # pour être plus précis, mais diviser par l'aire totale est une approximation sûre.
        # On multiplie par un facteur arbitraire si la barre ne remplit pas tout le rectangle
        # Pour l'instant, on renvoie le ratio brut par rapport à la taille de la zone.
        hp_ratio = smoothed_pixels / self.max_pixels
        
        return hp_ratio, mask # On retourne le mask pour le debug

# --- CLASSE 2 : LA RÉTINE (Déjà validée) ---
class TemporalRetina:
    def __init__(self, bounding_box, stack_size=4):
        self.sct = mss.mss()
        self.monitor = bounding_box
        self.input_shape = (84, 84)
        self.frames_buffer = deque(maxlen=stack_size)
        self.stack_size = stack_size

    def get_state(self):
        sct_img = self.sct.grab(self.monitor)
        img = np.array(sct_img)
        gray = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
        frame = cv2.resize(gray, self.input_shape)
        
        if len(self.frames_buffer) == 0:
            for _ in range(self.stack_size):
                self.frames_buffer.append(frame)
        else:
            self.frames_buffer.append(frame)
            
        return np.stack(self.frames_buffer, axis=2)

# --- CORPS PRINCIPAL (TEST D'INTÉGRATION) ---
def run_full_sensory_test():
    window_geo = get_game_window(WINDOW_TITLE)
    if not window_geo:
        print("❌ Jeu introuvable.")
        return

    print("🤖 Initialisation de l'organisme...")
    eye = TemporalRetina(window_geo)
    amygdala = BioMonitor(window_geo) # L'amygdale gère la peur (HP)
    
    print("✅ Systèmes en ligne. Appuyez sur 'q' pour arrêter.")
    
    while True:
        # 1. Perception Visuelle
        vision_state = eye.get_state()
        
        # 2. Perception Interne (Proprioception / Douleur)
        hp_percent, hp_mask = amygdala.read_hp()
        
        # --- DEBUG VISUALISATION ---
        # Vue Rétine (Dernière frame)
        retina_view = cv2.resize(vision_state[:,:,-1], (300, 300), interpolation=cv2.INTER_NEAREST)
        cv2.imshow("Cortex Visuel", retina_view)
        
        # Vue Amygdale (Masque de vie)
        cv2.imshow("Amygdale (Detection Vie)", hp_mask)
        
        # Affichage Console des signes vitaux
        # On crée une fausse barre de progression en ASCII
        bar_len = 20
        filled_len = int(hp_percent * 100 / (100/bar_len)) # Approximation grossière pour l'affichage
        # Note: Votre ratio sera surement faible (ex: 0.3) car la barre ne remplit pas tout le rectangle
        # C'est normal. L'important est que ça baisse quand vous prenez des coups.
        bar = '█' * filled_len + '-' * (bar_len - filled_len)
        
        print(f"\rSanté: [{bar}] Raw Ratio: {hp_percent:.4f}", end="")

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_full_sensory_test()