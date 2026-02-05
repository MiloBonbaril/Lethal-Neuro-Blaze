import numpy as np
import cv2
import mss
import time
from collections import deque

import ctypes
from ctypes import wintypes

# Configuration
MONITOR_ID = 1 

WINDOW_TITLE = "LLBlaze"

def get_game_window(title):
    user32 = ctypes.windll.user32
    handle = user32.FindWindowW(None, title)
    
    if not handle:
        print(f"⚠️ Fenêtre '{title}' introuvable ! Utilisation de la configuration par défaut.")
        return {"top": 100, "left": 100, "width": 800, "height": 600}

    # Récupérer la zone client (sans les bordures)
    rect = wintypes.RECT()
    user32.GetClientRect(handle, ctypes.byref(rect))
    
    # Convertir les coordonnées locales (0,0) en coordonnées écran
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

GAME_WINDOW = get_game_window(WINDOW_TITLE)

class TemporalRetina:
    def __init__(self, bounding_box, stack_size=4):
        self.sct = mss.mss()
        self.monitor = bounding_box
        self.input_shape = (84, 84)
        self.stack_size = stack_size
        
        # Le Buffer de mémoire à court terme (Short-term memory)
        # deque avec maxlen éjecte automatiquement le plus vieux quand on ajoute un nouveau
        self.frames_buffer = deque(maxlen=stack_size)
        
        print(f"👁️ Rétine Temporelle initialisée. Stack: {stack_size} frames.")

    def capture_frame(self):
        """Capture une seule frame, la traite et la retourne."""
        sct_img = self.sct.grab(self.monitor)
        img = np.array(sct_img)
        gray = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
        processed = cv2.resize(gray, self.input_shape)
        return processed

    def get_state(self):
        """
        Retourne l'état complet (Le Tenseur empilé).
        Si le buffer n'est pas plein (début de partie), on le remplit avec la même image.
        """
        frame = self.capture_frame()
        
        # Initialisation : Si le buffer est vide, on le remplit avec la première frame x4
        if len(self.frames_buffer) == 0:
            for _ in range(self.stack_size):
                self.frames_buffer.append(frame)
        else:
            self.frames_buffer.append(frame)
        
        # Empilement le long du canal de profondeur (channel axis)
        # Format de sortie : (84, 84, 4)
        # axis=2 pour empiler en profondeur (H, W, C)
        stacked_state = np.stack(self.frames_buffer, axis=2)
        
        return stacked_state

def run_temporal_test():
    retina = TemporalRetina(GAME_WINDOW)
    print("🧠 Cortex Visuel Temporel activé.")
    
    last_time = time.time()
    frame_counter = 0
    
    while True:
        # C'est ici que la magie opère. 'state' contient maintenant le TEMPS.
        state = retina.get_state()
        
        # Vérification de la forme du tenseur (Crucial pour le Deep Learning)
        # Doit afficher (84, 84, 4)
        # print(f"Forme du tenseur d'entrée : {state.shape}") 
        
        # Mesure FPS optimisée
        frame_counter += 1
        if frame_counter % 30 == 0: # On ne calcule pas à chaque frame pour économiser le CPU
            current_time = time.time()
            fps = 30 / (current_time - last_time)
            last_time = current_time
            # Afficher les FPS dans le titre de la fenêtre est moins coûteux que d'écrire sur l'image
            print(f"Synaptic Hz: {fps:.2f} | Tensor Shape: {state.shape}")

        # VISUALISATION POUR HUMAIN (DEBUG SEULEMENT)
        # Pour voir ce que le réseau voit, on affiche la dernière frame capturée (la plus récente)
        # Note : state[:, :, -1] est la frame la plus récente, state[:, :, 0] la plus vieille
        latest_frame_view = state[:, :, -1] 
        cv2.imshow('Neural Input (Latest Frame)', cv2.resize(latest_frame_view, (400, 400), interpolation=cv2.INTER_NEAREST))

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_temporal_test()