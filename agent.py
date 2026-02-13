import torch
import os
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from collections import deque
from brain import Brain  # On importe votre cerveau validé

import config

# --- HYPERPARAMÈTRES GÉNÉTIQUES ---
# Valeurs déplacées dans config.py

class ReplayBuffer:
    """
    L'Hippocampe artificiel. Stocke les transitions pour l'apprentissage hors-ligne.
    """
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        """Enregistre un souvenir."""
        # On s'assure que state est bien au format (C, H, W) compact
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        """Récupère un lot de souvenirs aléatoires (Rêve)."""
        state, action, reward, next_state, done = zip(*random.sample(self.buffer, batch_size))
        return np.stack(state), action, reward, np.stack(next_state), done

    def __len__(self):
        return len(self.buffer)

class Agent:
    def __init__(self, input_shape, num_actions):
        self.num_actions = num_actions
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"🧠 Agent initialisé sur : {self.device}")

        # --- LE DOUBLE CERVEAU ---
        # 1. Policy Net : Celui qui agit
        self.policy_net = Brain(input_shape, num_actions).to(self.device)
        # 2. Target Net : La référence stable
        self.target_net = Brain(input_shape, num_actions).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval() # Le Target Net ne s'entraîne pas directement

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=config.LEARNING_RATE)
        self.memory = ReplayBuffer(config.MEMORY_SIZE)

        self.steps_done = 0
        self.epsilon = config.EPSILON_START

    def select_action(self, state):
        """
        Stratégie Epsilon-Greedy :
        Soit on explore (Action aléatoire), soit on exploite (Meilleure Q-Value).
        """
        # Mise à jour du taux d'exploration
        self.epsilon = config.EPSILON_END + (config.EPSILON_START - config.EPSILON_END) * \
                       np.exp(-1. * self.steps_done / config.EPSILON_DECAY)
        self.steps_done += 1

        # EXPLOITATION (Cerveau)
        if random.random() > self.epsilon:
            with torch.no_grad():
                # On transforme l'état (numpy) en tenseur PyTorch
                state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
                # Le réseau sort les Q-values pour les 7 actions. On prend l'index du max.
                return self.policy_net(state_t).max(1)[1].item()
        
        # EXPLORATION (Hasard)
        else:
            return random.randrange(self.num_actions)

    def learn(self):
        """
        Le cœur de l'apprentissage (Backpropagation).
        C'est ici que la magie mathématique opère.
        """
        if len(self.memory) < config.BATCH_SIZE:
            return # Pas assez de souvenirs pour apprendre

        # 1. On rêve (Récupération d'un batch)
        states, actions, rewards, next_states, dones = self.memory.sample(config.BATCH_SIZE)

        # Conversion en tenseurs PyTorch
        state_batch = torch.tensor(states, dtype=torch.float32).to(self.device)
        action_batch = torch.tensor(actions, dtype=torch.int64).unsqueeze(1).to(self.device)
        reward_batch = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        next_state_batch = torch.tensor(next_states, dtype=torch.float32).to(self.device)
        done_batch = torch.tensor(dones, dtype=torch.float32).to(self.device)

        # 2. Calcul du Q_current (Ce que le cerveau a prédit pour ces états/actions)
        # gather(1, action_batch) permet de ne garder que la Q-value de l'action qui a été réellement prise
        q_values = self.policy_net(state_batch).gather(1, action_batch)

        # 3. Calcul du Q_target (La réalité + le futur estimé par le Target Net)
        # On utilise le Target Net pour estimer la valeur du PROCHAIN état
        with torch.no_grad():
            next_q_values = self.target_net(next_state_batch).max(1)[0]
        
        # Formule de Bellman : R + gamma * max(Q_next) * (1 - done)
        # Si done est vrai (1), le futur vaut 0 (car le jeu est fini)
        expected_q_values = reward_batch + (config.GAMMA * next_q_values * (1 - done_batch))

        # 4. Calcul de la perte (Huber Loss ou MSE)
        # On compare la prédiction (q_values) avec la cible (expected_q_values)
        loss = F.smooth_l1_loss(q_values, expected_q_values.unsqueeze(1))

        # 5. Optimisation (Mise à jour des poids synaptiques)
        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping pour éviter les explosions (stabilité)
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

        return loss.item()

    def update_target_network(self):
        """Synchronise le Target Net avec le Policy Net."""
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def save(self, filename, episode, best_reward):
        """
        Sauvegarde le cerveau et tout l'état de l'agent.
        """
        checkpoint = {
            'model_state': self.policy_net.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'episode': episode,
            'best_reward': best_reward
        }
        torch.save(checkpoint, filename + f"_e{episode}" + ".pth")

    def load(self, filename):
        """
        Charge le cerveau. Gère la rétrocompatibilité (si ancien format).
        Retourne (start_episode, best_reward).
        """
        if not os.path.exists(filename):
            return 1, -float('inf')

        checkpoint = torch.load(filename, weights_only=False)
        
        # Vérification : Est-ce un dictionnaire complet ou juste le state_dict (ancien format) ?
        if isinstance(checkpoint, dict) and 'model_state' in checkpoint:
            print("📦 Chargement d'un Checkpoint COMPLET.")
            self.policy_net.load_state_dict(checkpoint['model_state'])
            self.target_net.load_state_dict(checkpoint['model_state'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state'])
            self.epsilon = checkpoint['epsilon']
            return checkpoint['episode'] + 1, checkpoint['best_reward']
        else:
            print("⚠️ Attention : Chargement d'un modèle LEGACY (Poids seuls).")
            # C'est l'ancien format (juste les poids)
            self.policy_net.load_state_dict(checkpoint)
            self.target_net.load_state_dict(checkpoint)
            return 1, -float('inf')