import time
import numpy as np
import pydirectinput
import config
from senses import TemporalRetina, BioMonitor, get_game_window
from brain import MotorCortex

class Environment:
    def __init__(self):
        print("🌍 Initialisation de l'environnement...")
        self.game_geo = get_game_window(config.WINDOW_TITLE)
        if not self.game_geo:
            raise Exception("❌ ERREUR : Jeu introuvable. Assurez-vous que le jeu est lancé et visible.")
            
        self.eye = TemporalRetina(self.game_geo)
        self.amygdala = BioMonitor(self.game_geo)
        self.muscles = MotorCortex()
        
        # État interne
        self.last_hp = 0.0
        
    def _transmute_state(self, numpy_state):
        """ (84, 84, 4) -> (4, 84, 84) pour PyTorch """
        return numpy_state.transpose(2, 0, 1)

    def reset(self):
        """
        Réinitialise l'environnement :
        1. Attend que le joueur soit vivant (wait_for_resurrection).
        2. Vide le buffer visuel.
        3. Retourne l'état initial.
        """
        self._wait_for_resurrection()
        
        # On vide le buffer visuel pour ne pas voir le menu d'avant
        for _ in range(4): 
            self.eye.get_state()
            
        current_state_raw = self.eye.get_state()
        self.last_hp, _, _ = self.amygdala.read_hp()
        
        return self._transmute_state(current_state_raw)

    def _wait_for_resurrection(self):
        """
        Protocole de Salle d'Attente.
        Tant que la vie est à 0, on spamme la touche MENU et on attend.
        """
        print("💤 En attente de résurrection (Navigation Menu)...")
        pydirectinput.keyUp(config.MENU_KEY) # Sécurité
        
        no_hp_counter = 0
        
        while True:
            hp, _, _ = self.amygdala.read_hp()
            
            # Si la vie revient (plus de 5% pour être sûr que ce n'est pas du bruit)
            if hp > 0.05:
                print("✨ SIGNES VITAUX DÉTECTÉS ! LE COMBAT REPREND.")
                time.sleep(1.0) # On laisse une petite seconde pour que le "FIGHT" disparaisse
                break
                
            # Sinon, on est probablement dans les menus / replay
            no_hp_counter += 1
            
            # On appuie sur Espace toutes les ~0.5 secondes pour passer les dialogues/menus
            if no_hp_counter % 10 == 0:
                pydirectinput.press(config.MENU_KEY)
                print(".", end="", flush=True)
                
            time.sleep(0.05) # On ne surcharge pas le CPU pendant l'attente

    def step(self, action_idx):
        """
        Exécute une action et retourne (next_state, reward, done).
        Intègre le Frame Skipping.
        """
        accumulated_reward = 0
        done = False
        
        for _ in range(config.FRAME_SKIP):
            # 1. Exécution motrice
            self.muscles.execute(action_idx)
            
            # 2. Observation immédiate
            next_state_raw = self.eye.get_state()
            current_hp, _, hp_threatened = self.amygdala.read_hp()
            
            # 3. Calcul de la récompense intermédiaire
            r = 0.0
            hp_delta = current_hp - self.last_hp
            
            # Logique de vie/mort
            if hp_delta < -0.01:
                if current_hp == 0 and hp_threatened:
                    r = config.REWARD_DEATH
                    done = True
                elif current_hp == 0 and not hp_threatened:
                    r = 0 # TODO: On ne peut pas récompenser si on ne sait pas précisément l'état de la mort, donc on fait comme s'il y avait égalité.
                    done = True
                else:
                    r = config.REWARD_DAMAGE * abs(hp_delta)
            elif current_hp == 0:
                done = True # Déjà mort
            else:
                r = config.REWARD_SURVIVAL
            
            accumulated_reward += r
            self.last_hp = current_hp
            
            if done:
                break # Si on meurt pendant le frame skip, on arrête tout de suite
        
        next_state = self._transmute_state(next_state_raw)
        
        # Info optionnel (pour compatibilité future gym)
        info = {'hp': self.last_hp}
        
        return next_state, accumulated_reward, done, info
