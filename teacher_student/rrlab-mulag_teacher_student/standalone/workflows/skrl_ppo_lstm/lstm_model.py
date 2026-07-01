import torch
import math
import torch.nn as nn

# import the skrl components to build the RL system
from skrl.models.torch import DeterministicMixin, GaussianMixin, Model
from skrl.utils import set_seed

# seed for reproducibility
set_seed(42)

# define models (stochastic and deterministic models) using mixins
class Shared(GaussianMixin, DeterministicMixin, Model):
    def __init__(self, observation_space, action_space, device, 
                 clip_actions=False, clip_log_std=True, 
                 min_log_std=-20, max_log_std=2, reduction="sum"):
        
        Model.__init__(self, observation_space, action_space, device)
        GaussianMixin.__init__(self, clip_actions, clip_log_std, min_log_std, max_log_std, reduction)
        DeterministicMixin.__init__(self, clip_actions)

        self._shared_output = None

        # ==========================================
        # 1. Vision Encoder
        # ==========================================
        self.vision_encoder = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5, stride=2, padding=2),
            nn.ELU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1),
            nn.ELU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=0),
            nn.ELU(),
            nn.Flatten(),
            nn.Linear(2304, 512), 
            nn.Tanh()
        )

        # ==========================================
        # 2. Proprioceptive Encoder
        # ==========================================
        self.proprio_encoder = nn.Sequential(
            nn.Linear(115, 512),
            nn.ELU(),
            nn.Linear(512, 256),
            nn.ELU(),
            nn.Linear(256, 128),
            nn.ELU()
        )

        # ==========================================
        # 3. Policy Head (Actor)
        # ==========================================
        self.policy_head = nn.Sequential(
            nn.Linear(512 + 128, 512),
            nn.ELU(),
            nn.Linear(512, 256),
            nn.ELU(),
            nn.Linear(256, 128),
            nn.ELU()
        )
        self.mean_layer = nn.Linear(128, self.num_actions)
        self.log_std_parameter = nn.Parameter(torch.zeros(self.num_actions))

        # ==========================================
        # 4. Value Head (Critic)
        # ==========================================
        self.value_head = nn.Sequential(
            nn.Linear(512 + 128, 512),
            nn.ELU(),
            nn.Linear(512, 256),
            nn.ELU(),
            nn.Linear(256, 128),
            nn.ELU()
        )
        self.value_layer = nn.Linear(128, 1)

    def _get_shared_features(self, states):
        # Index 0 - 2583: height_map (3 * 21 * 41 = 2583)
        height_map_flat = states[:, 0:2583]
        height_map = height_map_flat.view(-1, 1, 21, 41)
        #print("Map: ", height_map.shape)
        vision_features = self.vision_encoder(height_map)

        # --- 2. Proprioception ---
        proprio_input = states[:, 2583:2698]
        #print("Propio: ", proprio_input.shape)
        proprio_features = self.proprio_encoder(proprio_input)

        return torch.cat([vision_features, proprio_features], dim=-1)
    

    def act(self, inputs, role):
        if role == "policy":
            return GaussianMixin.act(self, inputs, role)
        elif role == "value":
            return DeterministicMixin.act(self, inputs, role)

    def compute(self, inputs, role):
        states = inputs["states"]

        if role == "policy":
            self._shared_output = self._get_shared_features(states)
            policy_out = self.policy_head(self._shared_output)
            return self.mean_layer(policy_out), self.log_std_parameter, {}

        elif role == "value":
            shared_output = self._shared_output if self._shared_output is not None else self._get_shared_features(states)
            self._shared_output = None 
            value_out = self.value_head(shared_output)
            return self.value_layer(value_out), {}
        


# define models (stochastic and deterministic models) using mixins
class SharedRNN(GaussianMixin, DeterministicMixin, Model):
    def __init__(self, observation_space, action_space, device, 
                 clip_actions=False, clip_log_std=True, 
                 min_log_std=-20, max_log_std=2, reduction="sum",
                 num_envs=1, num_layers=1, hidden_size=256, sequence_length=64):
        
        Model.__init__(self, observation_space, action_space, device)
        GaussianMixin.__init__(self, clip_actions, clip_log_std, min_log_std, max_log_std, reduction)
        DeterministicMixin.__init__(self, clip_actions)

        # RNN params
        self.num_envs = num_envs
        self.num_layers = num_layers
        self.hidden_size = hidden_size 
        self.sequence_length = sequence_length

        # Cache for Shared Model
        self._shared_output = None
        self._shared_rnn_states = None

        # ==========================================
        # 1. Vision Encoder (Output: 512)
        # ==========================================
        self.vision_encoder = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5, stride=2, padding=2),
            nn.ELU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1),
            nn.ELU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=0),
            nn.ELU(),
            nn.Flatten(),
            nn.Linear(2304, 512), 
            nn.Tanh()
        )

        # ==========================================
        # 2. Proprioceptive Encoder (Output: 256)
        # ==========================================
        self.proprio_encoder = nn.Sequential(
            nn.Linear(115, 512),
            nn.ELU(),
            nn.Linear(512, 256),
            nn.ELU(),
            nn.Linear(256, 256),
            nn.ELU()
        )

        # ==========================================
        # 3. LSTM
        # Input: 512 (Vision) + 256 (Proprio) = 768
        # ==========================================
        self.lstm = nn.LSTM(input_size=768,
                            hidden_size=self.hidden_size,
                            num_layers=self.num_layers,
                            batch_first=True)

        # ==========================================
        # 4. Policy Head (Actor)
        # ==========================================
        self.policy_head = nn.Sequential(
            nn.Linear(self.hidden_size, 256),
            nn.ELU(),
            nn.Linear(256, 128),
            nn.ELU()
        )
        self.mean_layer = nn.Linear(128, self.num_actions)
        self.log_std_parameter = nn.Parameter(torch.zeros(self.num_actions))

        # ==========================================
        # 5. Value Head (Critic)
        # ==========================================
        self.value_head = nn.Sequential(
            nn.Linear(self.hidden_size, 256),
            nn.ELU(),
            nn.Linear(256, 128),
            nn.ELU()
        )
        self.value_layer = nn.Linear(128, 1)

        self._init_weights()


    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=math.sqrt(2))
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Conv2d):
                nn.init.orthogonal_(m.weight, gain=math.sqrt(2))
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            
        nn.init.orthogonal_(self.mean_layer.weight, gain=0.01)
        
        nn.init.orthogonal_(self.value_layer.weight, gain=1.0)


    def get_specification(self):
        return {"rnn": {"sequence_length": self.sequence_length,
                        "sizes": [(self.num_layers, self.num_envs, self.hidden_size),    
                                  (self.num_layers, self.num_envs, self.hidden_size)]}}  

    def _get_shared_features(self, states):
        # Height Map Features
        hm_size = 3*21*41
        height_map_flat = states[:, 0:hm_size]
        height_map = height_map_flat.view(-1, 3, 21, 41)
        vision_features = self.vision_encoder(height_map)

        # Proprioception Features
        proprio_input = states[:, hm_size:(hm_size + 115)]
        proprio_features = self.proprio_encoder(proprio_input)

        return torch.cat([vision_features, proprio_features], dim=-1)

    def _process_rnn(self, features, inputs):
        terminated = inputs.get("terminated", None)
        hidden_states, cell_states = inputs["rnn"][0], inputs["rnn"][1]

        if self.training:
            rnn_input = features.view(-1, self.sequence_length, features.shape[-1])
            hidden_states = hidden_states.view(self.num_layers, -1, self.sequence_length, hidden_states.shape[-1])
            cell_states = cell_states.view(self.num_layers, -1, self.sequence_length, cell_states.shape[-1])
            
            hidden_states = hidden_states[:,:,0,:].contiguous()
            cell_states = cell_states[:,:,0,:].contiguous()

            if terminated is not None and torch.any(terminated):
                rnn_outputs = []
                terminated = terminated.view(-1, self.sequence_length)
                indexes = [0] + (terminated[:,:-1].any(dim=0).nonzero(as_tuple=True)[0] + 1).tolist() + [self.sequence_length]

                for i in range(len(indexes) - 1):
                    i0, i1 = indexes[i], indexes[i + 1]
                    rnn_output, (hidden_states, cell_states) = self.lstm(rnn_input[:,i0:i1,:], (hidden_states, cell_states))
                    hidden_states[:, (terminated[:,i1-1]), :] = 0
                    cell_states[:, (terminated[:,i1-1]), :] = 0
                    rnn_outputs.append(rnn_output)

                rnn_states = (hidden_states, cell_states)
                rnn_output = torch.cat(rnn_outputs, dim=1)
            else:
                rnn_output, rnn_states = self.lstm(rnn_input, (hidden_states, cell_states))
        else:
            # Rollout / Inference
            rnn_input = features.view(-1, 1, features.shape[-1])
            rnn_output, rnn_states = self.lstm(rnn_input, (hidden_states, cell_states))

        rnn_output = torch.flatten(rnn_output, start_dim=0, end_dim=1)
        return rnn_output, rnn_states
    
    def act(self, inputs, role):
        if role == "policy":
            return GaussianMixin.act(self, inputs, role)
        elif role == "value":
            return DeterministicMixin.act(self, inputs, role)

    def compute(self, inputs, role):
        states = inputs["states"]

        if role == "policy":
            features = self._get_shared_features(states)

            rnn_output, rnn_states = self._process_rnn(features, inputs)

            self._shared_output = rnn_output
            self._shared_rnn_states = rnn_states

            policy_out = self.policy_head(rnn_output)
            return self.mean_layer(policy_out), self.log_std_parameter, {"rnn": [rnn_states[0], rnn_states[1]]}

        elif role == "value":
            if self._shared_output is not None:
                rnn_output = self._shared_output
                rnn_states = self._shared_rnn_states

                self._shared_output = None
                self._shared_rnn_states = None
            else:
                features = self._get_shared_features(states)
                rnn_output, rnn_states = self._process_rnn(features, inputs)

            value_out = self.value_head(rnn_output)
            return self.value_layer(value_out), {"rnn": [rnn_states[0], rnn_states[1]]}        