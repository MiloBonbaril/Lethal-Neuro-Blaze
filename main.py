import time
import os
import config
from agent import Agent
from environment import Environment
from torch.utils.tensorboard import SummaryWriter
import torch # For grid_sample or other utils if needed

import argparse
import sys

def run_training(env, neuro_agent, start_episode, best_reward):
    writer = SummaryWriter(log_dir="runs/LethalNeuroBlaze_Experiment_1")
    print(f"\n🧠 DÉBUT DE L'ENTRAÎNEMENT (Épisode {start_episode}/{config.EPISODES})")
    print(">>> PLACEZ LE JEU EN PREMIER PLAN <<<")
    time.sleep(5)

    for episode in range(start_episode, config.EPISODES + 1):
        # 1. RESET (Attente résurrection + Initialisation)
        current_state = env.reset()
        
        # Log de l'image vue par l'agent (au début de l'épisode)
        # current_state shape: (4, 84, 84) -> TensorBoard veut (C, H, W) ou (N, C, H, W)
        # On va juste logger le premier channel (ou tout)
        # Comme c'est du N&B empilé, on peut visualiser les 3 premiers channels comme RGB ou juste 1.
        # Pour simplifier, on envoie tout le bloc (le writer gere le multichannel parfois ou on prend le dernier frame)
        # On va prendre le dernier frame (le plus récent) pour la visibilité : current_state[-1]
        
        # Astuce : make_grid pour voir les 4 frames ? Ou juste le dernier.
        # Voyons simple : Le dernier frame (l'instant présent).
        # current_state est un Tensor ou numpy ? reset retourne un Tensor cpu ou numpy... ah environment.py step retourne numpy (C, H, W) ?
        # environment.py (reset) -> _transmute_state -> transpose (C, H, W) numpy array.
        
        last_frame = current_state[-1, :, :] # (84, 84)
        writer.add_image("Vision/Input", last_frame, episode, dataformats='HW')

        total_reward = 0
        step = 0
        done = False
        
        print(f"--- Épisode {episode} ---")
        
        while step < config.MAX_STEPS_PER_EPISODE and not done:
            step += 1
            
            # A. DÉCISION
            action_idx = neuro_agent.select_action(current_state)
            
            # B. ACTION & OBSERVATION (via Environment)
            next_state, reward, done, info = env.step(action_idx)
            
            total_reward += reward

            # C. MÉMORISATION & APPRENTISSAGE
            neuro_agent.memory.push(current_state, action_idx, reward, next_state, done)
            loss = neuro_agent.learn()
            
            if loss is not None:
                writer.add_scalar("Loss/Train", loss, neuro_agent.steps_done)
                writer.add_scalar("Epsilon/Step", neuro_agent.epsilon, neuro_agent.steps_done)
            
            current_state = next_state
            
            # Monitoring léger
            if step % 50 == 0:
                print(f"Step {step} (x{config.FRAME_SKIP}) | Eps: {neuro_agent.epsilon:.2f} | HP: {info['hp']:.2f} | R: {reward:.1f}")

        # Fin de l'épisode
        neuro_agent.update_target_network()
        
        writer.add_scalar("Reward/Episode", total_reward, episode)
        writer.add_scalar("Health/Last_HP", info['hp'], episode) # HP à la fin (ou moyenne ?)
        writer.add_scalar("Epsilon/Episode", neuro_agent.epsilon, episode)
        
        print(f"💀 Fin Épisode {episode}. Score: {total_reward:.2f}")

        if total_reward > best_reward:
            best_reward = total_reward
        
        if episode % config.SAVE_INTERVAL == 0:
            neuro_agent.save(config.MODEL_FILE, episode, best_reward)
            print("💾 Sauvegarde synaptique.")
    
    writer.close()

def run_inference(env, neuro_agent):
    print(f"\n🧠 MODE EXÉCUTION (INFERENCE ONLY)")
    print(">>> PLACEZ LE JEU EN PREMIER PLAN <<<")
    
    # Force epsilon à 0 pour désactiver l'exploration (pure exploitation)
    neuro_agent.epsilon = 0.0
    print(f"Exploration (Epsilon) : {neuro_agent.epsilon}")
    
    # Mettre le modèle en mode évaluation (désactive dropout, batchnorm, etc. si utilisés)
    neuro_agent.policy_net.eval()
    
    time.sleep(5)

    episode = 1
    while True:
        # 1. RESET
        current_state = env.reset()
        
        total_reward = 0
        step = 0
        done = False
        
        print(f"--- Épisode {episode} (Exécution) ---")
        
        while step < config.MAX_STEPS_PER_EPISODE and not done:
            step += 1
            
            # A. DÉCISION (Sans gradient)
            with torch.no_grad():
                # select_action utilise déjà epsilon, et on l'a mis à 0.
                # Mais on peut appeler explicitement policy_net pour être sûr ou laisser select_action 
                # qui gère déjà le torch.no_grad() dans la branche exploitation.
                # On va utiliser select_action car il gère le formatage de l'état.
                action_idx = neuro_agent.select_action(current_state)
            
            # B. ACTION & OBSERVATION
            next_state, reward, done, info = env.step(action_idx)
            
            total_reward += reward
            current_state = next_state
            
            # Monitoring léger
            if step % 50 == 0:
                 print(f"Step {step} | HP: {info['hp']:.2f} | R: {reward:.1f} | Action: {config.ACTION_MAP[action_idx]}")

        print(f"💀 Fin Épisode {episode}. Score: {total_reward:.2f}")
        episode += 1
        time.sleep(2) # Petite pause entre les épisodes

def main():
    parser = argparse.ArgumentParser(description='Lethal Neuro-Blaze Agent')
    parser.add_argument('--mode', choices=['train', 'run'], default='train', help='Mode de fonctionnement: train (entraînement) ou run (exécution/inférence)')
    args = parser.parse_args()

    print("🧬 INITIALISATION DU PROJET LETHAL NEURO-BLAZE (OPTIMISÉ)...")

    try:
        env = Environment()
    except Exception as e:
        print(f"Erreur d'initialisation de l'environnement: {e}")
        return

    input_shape = config.INPUT_SHAPE
    num_actions = len(config.ACTION_MAP)
    neuro_agent = Agent(input_shape, num_actions)

    start_episode = 1
    best_reward = -float('inf')

    if os.path.exists(config.MODEL_FILE):
        print(f"📂 Chargement du cerveau : {config.MODEL_FILE}")
        start_episode, best_reward = neuro_agent.load(config.MODEL_FILE)
    else:
        if args.mode == 'run':
            print("⚠️ AUCUN MODÈLE TROUVÉ ! Impossible de lancer en mode exécution.")
            return

    if args.mode == 'train':
        run_training(env, neuro_agent, start_episode, best_reward)
    elif args.mode == 'run':
        run_inference(env, neuro_agent)

if __name__ == "__main__":
    main()