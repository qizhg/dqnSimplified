if not dqn then
    require 'initenv'
end

local nql = torch.class('dqn.NeuralQLearner')


function nql:__init(args)
--args are agent params


    self.network = args.q_net or "convnet_atari3" -- name of the Q_net
    self.preproc = args.preproc_net or "preproc_screen" -- name of preprocessing network

    self.state_dim  = args.state_dim or 84*84 --size of screen after preproc
    self.hist_len   = args.hist_len or 4      --last 4 frames as a full_state
    self.input_dims = args.input_dims or {self.hist_len, 84, 84} --input_dims to the Q_net

    self.actions    = args.actions
    self.n_actions  = #self.actions

    --Load Q_net, self.network: string-> function -> nn.Module
    self.network = require(self.network) 
    self.network = self:network()

    --Load preproc net, self.preproc: string-> function -> nn.Module
    self.preproc = require(self.preproc)
    self.preproc = self:preproc()

    --epsilon(-greedy) annealing from ep_start to ep_end in ep_endt steps
    self.ep_start = args.ep_start or 1
    self.ep_end   = args.ep_end or 0.1
    self.ep_endt  = args.ep_endt or 1000000
    self.ep       = self.ep_start

    --lr annealing from lr_start to lr_end in lr_endt steps
    self.lr_start = args.lr_start or 0.01
    self.lr_end   = args.lr_end or 0.00025
    self.lr_endt  = args.lr_endt or 1000000
    self.lr       = self.lr_start

    --L2 weight cost
    self.wc = args.wc or 0

    --Q-learning update frequency, update every update_freq steps, use minibatch_size transitions to do n_replay times weight gradient descent in each update 
    self.discount       = args.discount or 0.99 --discount factor, gamma
    self.update_freq    = args.update_freq or 4
    self.minibatch_size = args.minibatch_size or 32
    self.n_replay       = args.n_replay or 1
    
    --target_q
    self.target_q       = args.target_q or 10000  --update target_network every target_q steps
    self.target_network = self.network:clone()

    --reward clipping

    --delta (Q error) clipping

    --Replay memory

    --current number of steps









end