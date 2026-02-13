import numpy as np

# --- GAME SETTINGS ---
WINDOW_TITLE = "LLBlaze"
MENU_KEY = 'space'

# --- HYPERPARAMETERS ---
EPISODES = 1000
MAX_STEPS_PER_EPISODE = 4000
SAVE_INTERVAL = 1
FRAME_SKIP = 1  # L'IA réfléchit 1 fois, agit 4 fois. (Gain FPS majeur)

# --- GENETIC HYPERPARAMETERS (AGENT) ---
BATCH_SIZE = 32         # Combien de souvenirs on ressasse à la fois
GAMMA = 0.99            # Facteur d'actualisation (L'importance du futur vs immédiat)
EPSILON_START = 1.0     # 100% exploration au début (Bébé ne sait rien)
EPSILON_END = 0.02      # 2% exploration à la fin (Reste un peu curieux)
EPSILON_DECAY = 10000   # Vitesse de réduction de la curiosité
TARGET_UPDATE = 1000    # On met à jour le cerveau "Cible" tous les 1000 pas
LEARNING_RATE = 1e-4    # Vitesse d'apprentissage (Neuroplasticité)
MEMORY_SIZE = 10000     # Capacité de l'Hippocampe

# --- MODEL SETTINGS ---
MODEL_FILE = "neuro_blaze_v1.1"
INPUT_SHAPE = (4, 84, 84) # (Channels, Height, Width)

# --- REWARDS ---
REWARD_SURVIVAL = 0.1
REWARD_DAMAGE = -50.0
REWARD_DEATH = -100.0
REWARD_WIN = 100.0

# --- SENSES ---
# Vos valeurs calibrées
HP_ROI_RELATIVE = {'top': 76, 'left': 94, 'width': 190, 'height': 69}
LOWER_YELLOW = np.array([27, 158, 151])
UPPER_YELLOW = np.array([38, 246, 250])

# --- ACTIONS ---
# Mapping : Index Neuronal -> Touche Clavier
# Assurez-vous que ces touches correspondent à votre configuration dans le jeu !
ACTION_MAP = {
    # --- ACTIONS ATOMIQUES (Base) ---
    0: None,             # Wait (Vital !)
    1: 'left',
    2: 'right',
    3: 'space',          # Saut Neutre
    4: 'c',              # Swing Neutre
    5: 'x',              # Bunt (Défense)
    6: 'w',              # Grab
    
    # --- COMBOS DE MOUVEMENT (Indispensables) ---
    # Sans ça, elle ne peut pas avancer en l'air
    7: ['right', 'space'], # Saut vers l'avant
    8: ['left', 'space'],  # Saut vers l'arrière
    
    # --- COMBOS D'ATTAQUE (Smash & Lob) ---
    # Indispensable pour varier les angles de la balle
    9: ['down', 'c'],      # Smash (Frapper vers le bas)
    10: ['up', 'c'],       # Lob / Tir vers le haut (nécessite d'ajouter 'up' seul ?)

    # --- COMBOS SPECIAUX ---
    11: ['c', 'c'],        # utilisation du spécial
    
    # --- COMBOS DE DEFENSE ---
    12: ['down', 'x'],        # bunt vers le bas
    13: ['up', 'x'],        # bunt vers le haut
    
    # --- COMBOS OPTIONNELS (À activer si l'IA s'ennuie) ---
    # 11: ['right', 'c'],  # Frapper en avançant (Souvent redondant avec Swing neutre + inertie)
    # 12: ['down', 'space'] # Fast fall (Tomber vite) ? Utile à haut niveau.
}