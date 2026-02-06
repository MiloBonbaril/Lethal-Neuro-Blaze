import numpy as np
import cv2
import time
import torch
import os

# Importation de vos organes
from senses import TemporalRetina, BioMonitor, get_game_window
from brain import MotorCortex, ACTION_MAP
from agent import Agent

# --- HYPERPARAMÈTRES DE L'EXPÉRIENCE ---
WINDOW_TITLE = "LLBlaze"
EPISODES = 500              # Nombre de parties à jouer
MAX_STEPS_PER_EPISODE = 2000 # Sécurité pour éviter les boucles infinies
SAVE_INTERVAL = 10          # Sauvegarder le cerveau tous les X épisodes
MODEL_FILE = "neuro_blaze_v1.pth"

# Récompenses (La chimie du plaisir et de la douleur)
REWARD_SURVIVAL = 0.1       # Joie d'être en vie à chaque frame
REWARD_DAMAGE = -50.0       # Douleur intense quand la vie baisse
REWARD_DEATH = -100.0       # Traumatisme final
REWARD_WIN = 100.0          # Extase de la victoire

def transmute_state(numpy_state):
    """
    Transforme la perception brute (Numpy HWC) en influx nerveux (Torch CHW).
    Entrée : (84, 84, 4) -> Sortie : (4, 84, 84)
    """
    # Transpose les axes : (2, 0, 1) met le canal (index 2) en premier
    return numpy_state.transpose(2, 0, 1)

def main():
    print("🧬 INITIALISATION DU PROJET LETHAL NEURO-BLAZE...")

    # 1. Connexion aux Organes Sensoriels
    game_geo = get_game_window(WINDOW_TITLE)
    if not game_geo:
        print("❌ ERREUR CRITIQUE : Jeu introuvable. Lancez Lethal League Blaze.")
        return

    eye = TemporalRetina(game_geo)
    amygdala = BioMonitor(game_geo)
    muscles = MotorCortex()
    
    # 2. Naissance de l'Agent
    # Input shape pour l'agent : (Channels, Height, Width)
    input_shape = (4, 84, 84) 
    num_actions = len(ACTION_MAP) # Devrait être 7 (0 à 6)
    
    neuro_agent = Agent(input_shape, num_actions)

    # Chargement d'un cerveau existant si disponible (Transmigration)
    if os.path.exists(MODEL_FILE):
        print(f"📂 Cerveau existant détecté. Chargement des poids synaptiques...")
        neuro_agent.load(MODEL_FILE)
    else:
        print(f"👶 Création d'un nouveau cerveau vierge.")

    print(f"\n🧠 DÉBUT DE L'ENTRAÎNEMENT ({EPISODES} générations prévues)")
    print("Passez sur la fenêtre du jeu. L'IA prend le contrôle dans 5 secondes...")
    time.sleep(5)

    # --- BOUCLE DES ÉPISODES (Générations) ---
    for episode in range(1, EPISODES + 1):
        # Reset de l'état pour une nouvelle partie
        # On vide un peu le buffer visuel pour ne pas voir la partie d'avant
        # Note: Idéalement, on devrait avoir une fonction reset() dans la rétine
        print(f"--- Épisode {episode} ---")
        
        # On capture l'état initial
        current_state = transmute_state(eye.get_state())
        
        last_hp = 1.0 # On commence full life (ou on l'espère)
        total_reward = 0
        step = 0
        
        while step < MAX_STEPS_PER_EPISODE:
            step += 1
            
            # A. DÉCISION (Le Cerveau choisit)
            action_idx = neuro_agent.select_action(current_state)
            
            # B. ACTION (Le Corps exécute)
            muscles.execute(action_idx)
            
            # C. DÉLAI DE RÉACTION & OBSERVATION
            # On laisse un tout petit temps au jeu pour réagir (physique)
            # Si le jeu tourne à 60FPS, 1 frame = ~0.016s.
            # On ne veut pas spammer trop vite.
            # time.sleep(0.01) # Optionnel, dépend de la vitesse de votre machine
            
            next_state_raw = eye.get_state()
            next_state = transmute_state(next_state_raw)
            
            # D. PERCEPTION DE LA RÉCOMPENSE (Amygdale)
            current_hp, _ = amygdala.read_hp()
            reward = 0
            done = False
            
            # Logique de Survie (Heuristique)
            # 1. Calcul de la différence de vie
            hp_delta = current_hp - last_hp
            
            if hp_delta < -0.01: # Perte de vie significative (Filtrage du bruit)
                # Si on passe brutalement à 0 alors qu'on avait de la vie -> MORT
                if current_hp == 0 and last_hp > 0.1:
                    reward = REWARD_DEATH
                    done = True
                    print("💀 MORT DETECTÉE.")
                else:
                    # Dégâts standard
                    reward = REWARD_DAMAGE * abs(hp_delta) # Plus on a mal, plus c'est punitif
                    # print(f"🩸 Dégâts reçus ! Reward: {reward:.2f}")
            
            elif current_hp == 0 and last_hp < 0.1:
                # On était déjà mort ou presque, et on reste à 0
                # C'est la fin de l'épisode (ou l'attente du respawn)
                done = True
            
            elif current_hp == 0 and last_hp > 0.1:
                 # Cas étrange : HUD disparait alors qu'on allait bien -> VICTOIRE ?
                 # Dans le doute, on considère cela comme une fin d'épisode positive
                 reward = REWARD_WIN
                 done = True
                 print("🏆 VICTOIRE PROBABLE (Disparition HUD).")
            
            else:
                # On est en vie et stable
                reward = REWARD_SURVIVAL

            # Mise à jour de la mémoire immédiate
            last_hp = current_hp
            total_reward += reward

            # E. MÉMORISATION (Replay Buffer)
            # On stocke l'expérience dans l'hippocampe
            neuro_agent.memory.push(current_state, action_idx, reward, next_state, done)

            # F. APPRENTISSAGE (Plasticité Synaptique)
            # L'agent rêve et optimise ses poids
            loss = neuro_agent.learn()
            
            # Transition d'état
            current_state = next_state
            
            # Affichage périodique (monitoring)
            if step % 100 == 0:
                print(f"Step {step} | Epsilon: {neuro_agent.epsilon:.3f} | HP: {current_hp:.2f}")

            if done:
                break
        
        # Fin de l'épisode
        neuro_agent.update_target_network()
        print(f"Fin Épisode {episode}. Reward Total: {total_reward:.2f}. Steps: {step}")
        
        if episode % SAVE_INTERVAL == 0:
            neuro_agent.save(MODEL_FILE)
            print("💾 Cerveau sauvegardé.")
            
        # Pause pour laisser le jeu recharger (Menu, Replay...)
        # Vous devrez peut-être appuyer manuellement sur 'A' pour relancer une partie
        # ou coder une fonction "press_continue" aveugle.
        time.sleep(3) 

if __name__ == "__main__":
    main()