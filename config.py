import numpy as np

# --- GAME SETTINGS ---
WINDOW_TITLE = "LLBlaze"
MENU_KEY = 'space'

# --- HYPERPARAMETERS ---
EPISODES = 1000
MAX_STEPS_PER_EPISODE = 4000
SAVE_INTERVAL = 3
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
MODEL_FILE = "neuro_blaze_v1.pth"
INPUT_SHAPE = (4, 84, 84) # (Channels, Height, Width)

# --- REWARDS ---
REWARD_SURVIVAL = 0.01      # Réduit car on va l'accumuler sur 4 frames
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
    0: None,            # No-Op
    1: 'left',
    2: 'right',
    3: 'space',         # Saut
    4: 'c',             # Frappe (Swing)
    5: 'x',             # Bunt
    6: 'z'              # Grab
}
