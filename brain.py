import torch
import torch.nn as nn
import torch.nn.functional as F
import pydirectinput
import time
from collections import Counter

import config

class MotorCortex:
    def __init__(self):
        pydirectinput.FAILSAFE = False
        # On stocke un SET (ensemble) de touches actuellement enfoncées
        self.current_keys = set() 
        print("💪 Cortex Moteur Avancé (Polyvalent) connecté.")

    def execute(self, action_idx, action_map):
        target_action = action_map.get(action_idx)
        
        # 1. Déterminer quelles touches DOIVENT être enfoncées ou tappées
        target_keys = set()
        multi_tap_keys = [] # Liste de tuples (key, count)
        
        if target_action is None:
            pass # Aucune touche
        elif isinstance(target_action, list):
            # Compter les occurrences de chaque touche
            counts = Counter(target_action)
            
            for k, count in counts.items():
                if count > 1:
                    multi_tap_keys.append((k, count))
                else:
                    target_keys.add(k)
        else:
            target_keys.add(target_action) # Une seule touche

        # 2. Différentiel (Quoi relâcher ? Quoi appuyer ?)
        
        # Touches à relâcher : Celles qui étaient là AVANT mais pas MAINTENANT
        keys_to_release = self.current_keys - target_keys
        
        # Touches à appuyer : Celles qui sont là MAINTENANT mais pas AVANT
        keys_to_press = target_keys - self.current_keys
        
        # 3. Exécution physique
        
        # A. Relâcher les anciennes touches maintenues
        for k in keys_to_release:
            pydirectinput.keyUp(k)
            
        # B. Appuyer sur les nouvelles touches à maintenir
        for k in keys_to_press:
            pydirectinput.keyDown(k)
            
        # C. Gérer les multi-taps (ex: ['c', 'c'])
        for k, count in multi_tap_keys:
            pydirectinput.press(k, presses=count, interval=0.05)
            
        # 4. Mise à jour de la mémoire d'état (uniquement pour les touches maintenues)
        self.current_keys = target_keys

    def release_all(self):
        for k in self.current_keys:
            pydirectinput.keyUp(k)
        self.current_keys = set()

# --- ARCHITECTURE DU CERVEAU (DQN) ---
class Brain(nn.Module):
    def __init__(self, input_shape, num_actions):
        super(Brain, self).__init__()
        
        # input_shape attendu: (4, 84, 84) -> (Channels, Height, Width)
        c, h, w = input_shape
        
        # 1. Traitement Visuel (V1 -> V4)
        # Conv2d(in_channels, out_channels, kernel_size, stride)
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1)
        
        # 2. Calcul de la taille de sortie après les convolutions pour le "Flatten"
        # Maths de convolution: Output = (Input - Kernel)/Stride + 1
        # 84 -> (84-8)/4 + 1 = 20
        # 20 -> (20-4)/2 + 1 = 9
        # 9  -> (9-3)/1 + 1 = 7
        # Donc sortie finale : 64 canaux * 7 * 7
        self.fc_input_dim = 64 * 7 * 7
        
        # 3. Cortex Associatif (Prise de décision)
        self.fc1 = nn.Linear(self.fc_input_dim, 512)
        self.fc2 = nn.Linear(512, num_actions) # Couche de sortie (7 actions)

    def forward(self, x):
        """
        Propagation avant (Forward Pass).
        x: Tenseur d'entrée normalisé (0-1)
        """
        # Feature Extraction
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        
        # Aplatissement (Flatten) pour passer aux couches denses
        x = x.reshape(x.size(0), -1) 
        
        # Raisonnement
        x = F.relu(self.fc1(x))
        
        # Action (Pas de Softmax ici car on veut les Q-Values brutes)
        return self.fc2(x)

# --- TEST UNITAIRE RAPIDE ---
if __name__ == "__main__":
    # Simulation d'un cerveau
    input_shape = (4, 84, 84)
    input_shape = (4, 84, 84)
    num_actions = len(config.ACTION_MAP) - 1
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"🧠 Initialisation du cerveau sur : {device}")
    model = Brain(input_shape, num_actions).to(device)
    
    # Création d'un stimulus visuel factice (Batch de 1, bruit aléatoire)
    dummy_input = torch.zeros(1, *input_shape).to(device)
    
    # Test de propagation
    output = model(dummy_input)
    
    print(f"Structure du modèle : \n{model}")
    print(f"Sortie pour un stimulus vide (Q-Values) : {output.detach().cpu().numpy()}")
    print("✅ Le cerveau est structurellement viable.")
    
    # Test Moteur (Attention ça va appuyer sur des touches !)
    motor = MotorCortex()
    motor.execute(3) # Devrait faire 'Espace'