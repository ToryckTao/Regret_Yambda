import torch.nn.functional as F
import torch.nn as nn
import torch

# Add parent directory to path
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.components import DNN
from utils import get_regularization


class GeneralCritic(nn.Module):
    @staticmethod
    def parse_model_args(parser):
        '''
        args:
        - critic_hidden_dims
        - critic_dropout_rate
        '''
        parser.add_argument('--critic_hidden_dims', type=int, nargs='+', default=[128, 64], 
                            help='hidden layer dimensions for critic')
        parser.add_argument('--critic_dropout_rate', type=float, default=0.2, 
                            help='dropout rate in critic layers')
        return parser
    
    def __init__(self, args, environment, actor):
        super().__init__()
        self.state_dim = actor.state_dim
        self.action_dim = actor.action_dim
        self.critic_net = DNN(self.state_dim + self.action_dim, 
                             args.critic_hidden_dims, 1, 
                             dropout_rate = args.critic_dropout_rate)
        
    def forward(self, feed_dict):
        '''
        @input:
        - feed_dict: {"state_emb": (B, state_dim), "action_emb": (B, action_dim)}
        @output:
        - out_dict: {"q": (B,)}
        '''
        state_emb = feed_dict['state_emb']
        action_emb = feed_dict['action_emb']
        state_action_emb = torch.cat((state_emb, action_emb), dim = -1)
        q = self.critic_net(state_action_emb)
        q = q.squeeze(-1)
        return {'q': q}
