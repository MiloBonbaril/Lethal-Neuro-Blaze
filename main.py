import numpy as np
import cv2
import time
import torch
import os
import pydirectinput

# Importation de vos organes
from senses import TemporalRetina, BioMonitor, get_game_window
from brain import MotorCortex
from agent import Agent
import config



def transmute_state(numpy_state):
    """ (84, 84, 4) -> (4, 84, 84) pour PyTorch """
    return numpy_state.transpose(2, 0, 1)

def wait_for_resurrection(amygdala, muscles):
    """
    Protocole de Salle d'Attente.
    Tant que la vie est à 0, on spamme la touche MENU et on attend.
    """
    print("💤 En attente de résurrection (Navigation Menu)...")
    pydirectinput.keyUp(config.MENU_KEY) # Sécurité
    
    no_hp_counter = 0
    
    while True:
        hp, _ = amygdala.read_hp()
        
        # Si la vie revient (plus de 5% pour être sûr que ce n'est pas du bruit)
        if hp > 0.05:
            print("✨ SIGNES VITAUX DÉTECTÉS ! LE COMBAT REPREND.")
            # On laisse une petite seconde pour que le "FIGHT" disparaisse
            time.sleep(1.0) 
            break
            
        # Sinon, on est probablement dans les menus / replay
        no_hp_counter += 1
        
        # On appuie sur Espace toutes les ~0.5 secondes pour passer les dialogues/menus
        if no_hp_counter % 10 == 0:
            pydirectinput.press(config.MENU_KEY)
            print(".", end="", flush=True)
            
        time.sleep(0.05) # On ne surcharge pas le CPU pendant l'attente

def main():
    print("🧬 INITIALISATION DU PROJET LETHAL NEURO-BLAZE (OPTIMISÉ)...")

    game_geo = get_game_window(config.WINDOW_TITLE)
    if not game_geo:
        print("❌ ERREUR : Jeu introuvable.")
        return

    eye = TemporalRetina(game_geo)
    amygdala = BioMonitor(game_geo)
    muscles = MotorCortex()
    
    input_shape = config.INPUT_SHAPE
    num_actions = len(config.ACTION_MAP)
    neuro_agent = Agent(input_shape, num_actions)

    if os.path.exists(config.MODEL_FILE):
        print(f"📂 Chargement du cerveau : {config.MODEL_FILE}")
        neuro_agent.load(config.MODEL_FILE)
    
    print(f"\n🧠 DÉBUT DE L'ENTRAÎNEMENT")
    print(">>> PLACEZ LE JEU EN PREMIER PLAN <<<")
    time.sleep(5)

    for episode in range(1, config.EPISODES + 1):
        # 1. ATTENTE ACTIVE (On ne commence que si on est VIVANT)
        wait_for_resurrection(amygdala, muscles)
        
        print(f"--- Épisode {episode} ---")
        
        # On vide le buffer visuel pour ne pas voir le menu d'avant
        # (Astuce: on capture 4 frames à vide pour purger la rétine)
        for _ in range(4): eye.get_state()
        
        current_state = transmute_state(eye.get_state())
        last_hp, _ = amygdala.read_hp()
        total_reward = 0
        step = 0
        done = False
        
        while step < config.MAX_STEPS_PER_EPISODE and not done:
            step += 1
            
            # A. DÉCISION (1 fois toutes les 4 frames)
            action_idx = neuro_agent.select_action(current_state)
            
            # B. ACTION REPEATER (Frame Skipping)
            # On maintient l'action et on observe le résultat cumulé
            accumulated_reward = 0
            
            for _ in range(config.FRAME_SKIP):
                # Exécution motrice
                muscles.execute(action_idx)
                
                # Petite pause physique (latence jeu)
                # time.sleep(0.005) # À ajuster si le jeu lag, sinon enlever pour max speed
                
                # Observation immédiate
                next_state_raw = eye.get_state()
                current_hp, _ = amygdala.read_hp()
                
                # Calcul de la récompense intermédiaire
                r = 0
                hp_delta = current_hp - last_hp
                
                # Logique de vie/mort (identique à avant)
                if hp_delta < -0.01:
                    if current_hp == 0 and last_hp > 0.1:
                        r = config.REWARD_DEATH
                        done = True
                    else:
                        r = config.REWARD_DAMAGE * abs(hp_delta)
                elif current_hp == 0 and last_hp < 0.1:
                    done = True # Déjà mort
                elif current_hp == 0 and last_hp > 0.1:
                    r = config.REWARD_WIN # HUD disparu subitement
                    done = True
                else:
                    r = config.REWARD_SURVIVAL
                
                accumulated_reward += r
                last_hp = current_hp
                
                if done:
                    break # Si on meurt pendant le frame skip, on arrête tout de suite
            
            # Fin du bloc Frame Skip
            next_state = transmute_state(next_state_raw)
            total_reward += accumulated_reward

            # C. MÉMORISATION & APPRENTISSAGE
            neuro_agent.memory.push(current_state, action_idx, accumulated_reward, next_state, done)
            
            # On apprend à chaque step (qui correspond maintenant à 4 frames réelles)
            loss = neuro_agent.learn()
            
            current_state = next_state
            
            # Monitoring léger
            if step % 50 == 0:
                print(f"Step {step} (x{config.FRAME_SKIP}) | Eps: {neuro_agent.epsilon:.2f} | HP: {last_hp:.2f} | R: {accumulated_reward:.1f}")

        # Fin de l'épisode
        neuro_agent.update_target_network()
        print(f"💀 Fin Épisode {episode}. Score: {total_reward:.2f}")
        
        if episode % config.SAVE_INTERVAL == 0:
            neuro_agent.save(config.MODEL_FILE)
            print("💾 Sauvegarde synaptique.")

if __name__ == "__main__":
    main()