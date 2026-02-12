import time
import os
import config
from agent import Agent
from environment import Environment

def main():
    print("🧬 INITIALISATION DU PROJET LETHAL NEURO-BLAZE (OPTIMISÉ)...")

    try:
        env = Environment()
    except Exception as e:
        print(e)
        return

    input_shape = config.INPUT_SHAPE
    num_actions = len(config.ACTION_MAP)
    neuro_agent = Agent(input_shape, num_actions)

    start_episode = 1
    best_reward = -float('inf')

    if os.path.exists(config.MODEL_FILE):
        print(f"📂 Chargement du cerveau : {config.MODEL_FILE}")
        start_episode, best_reward = neuro_agent.load(config.MODEL_FILE)
    
    print(f"\n🧠 DÉBUT DE L'ENTRAÎNEMENT (Épisode {start_episode}/{config.EPISODES})")
    print(">>> PLACEZ LE JEU EN PREMIER PLAN <<<")
    time.sleep(5)

    for episode in range(start_episode, config.EPISODES + 1):
        # 1. RESET (Attente résurrection + Initialisation)
        current_state = env.reset()
        
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
            
            current_state = next_state
            
            # Monitoring léger
            if step % 50 == 0:
                print(f"Step {step} (x{config.FRAME_SKIP}) | Eps: {neuro_agent.epsilon:.2f} | HP: {info['hp']:.2f} | R: {reward:.1f}")

        # Fin de l'épisode
        neuro_agent.update_target_network()
        print(f"💀 Fin Épisode {episode}. Score: {total_reward:.2f}")

        if total_reward > best_reward:
            best_reward = total_reward
        
        if episode % config.SAVE_INTERVAL == 0:
            neuro_agent.save(config.MODEL_FILE, episode, best_reward)
            print("💾 Sauvegarde synaptique.")

if __name__ == "__main__":
    main()