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
    self.fullState_dims = args.fullState_dims or {self.hist_len, 84, 84} --fullState_dims to the Q_net

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
    self.learn_start    = args.discount or 50000
    self.update_freq    = args.update_freq or 4
    self.minibatch_size = args.minibatch_size or 32
    self.n_replay       = args.n_replay or 1
    
    --target_q
    self.target_q       = args.target_q or 10000  --update target_network every target_q steps
    self.target_network = self.network:clone()

    --reward clipping
    self.max_reward    = args.max_reward or 1
    self.min_reward    = args.min_reward or -1

    --delta (Q error) clipping

    --Replay memory
    ReplayMemory_args ={
        maxSize  = 1000000 --unit is stateDim, i.e. dim of a frame after preproc
        histLen  = self.hist_len
        stateDim = self.state_dim --84 x 84, frame size after preproc
        fullStateDims =self.fullState_dims
        numActions = self.n_actions
    }
    self.replayMemory  = dqn.ReplayMemory(ReplayMemory_args)

    --current number of steps
    self.step = 0
end

function nql:perceive(screen, reward, terminal)

    --preproc screnn                                --screen (1,3,210,160)
    state = self.preproc:forward(screen):clone()    --state  (1,84,84)
    state = torch.reshape(state,self.state_dim)     --state  (84*84,)

    --clip reward
    if self.max_reward then
        reward = math.min(reward,self.max_reward)
    end

    if self.min_reward then
        reward = math.max(reward,self.min_reward)
    end

    --store (s = lastState, t = lastTerminal, a = lastAction, r=reward, s'=state)
    if self.lastState then
        self.replayMemory:add(self.lastState, self.lastTerminal, self.lastAction, reward)
    end

    --get fullstate, i.e. hist_len number of states stacked
    self.replayMemory:add_recent_state(state,terminal)
    local fullState = self.replayMemory:get_current_fullState()

    --select and action, no action index (index = 0) when terminal
    local actionIndex = 0
    if not terminal then
        actionIndex = self.eGreedy(fullState)
    end
    -----self.replayMemory:add_recent_action(actionIndex) --???? why do we need recent_action

    --Q updates, after learn_start steps update every update_freq steps, use minibatch_size transitions to do n_replay times weight gradient descent in each update 
    if self.step >= self.learn_start and self.step % self.update_freq==0 then
        for i = 1, self.n_replay
            self:qLearnMinibatch()
        end
    end

    --target Q updates
    if self.target_q and self.steps % self.target_q == 0 then
        self.target_network = self.network:clone()
    end 

    self.lastState = state:clone()
    self.lastTerminal = terminal
    self.lastAction = actionIndex

    self.step = self.step + 1

    return actionIndex  
end

function nql:eGreedy(fullState)
    self.ep =(self.ep_end +
                math.max(0, (self.ep_start - self.ep_end) * (self.ep_endt -
                math.max(0, self.numSteps - self.learn_start))/self.ep_endt))
    if torch.uniform() < self.ep then
        return torch.random(1, self.n_actions)
    else
        return self:greedy(fullState)
    end
    
end

function nql:Greedy(fullState)
    --fullState {hist_len, 84, 84}

    local q = self.network:forward(fullState):clone():float():squeeze() 
           --squeeze removes all singleton dimensions of the tensor, s.t. q[a] is Q(s,a)
    local qmax  = q[1]
    local besta = {1}

    for a =2, self.n_actions do
        if q[a] > qmax then
            qmax = q[a]
            besta ={a}
        elseif q[a] == qmax then
            besta[#besta+1] =a
        end
    end

    local r = torch.random(1,#besta)
    return besta[r] --with random tie-breaking
end


function nql:qLearnMinibatch()

    local s,a,r,s_prime,term_prime = self.replayMemory:sample(self.minibatch_size)
    --s, s_prime (minibatch_size, hist_len x 84 x 84)  fullstate
    --t, a, r (minibatch_size,)

    local gradQ = self:get_gradQ(s,a,r,s_prime,term_prime)  --gradQ (minibatch_size, n_actions)

    local w, dw = self.network:getParameters()
    dw:zero()
    self.network:backward(s,gradQ)
    self.dw:add(-self.wc, self.w) --L2 weight cost

    --Perform Gradient Update
       ------CODE HERE

end

function nql:get_gradQ(s,a,r,s_prime,term_prime)
--input:
    --s, s_prime, (minibatch_size, hist_len x 84 x 84) minibatch of fullState
    --t, a, r, (minibatch_size,)
--output:
    --gradQ (minibatch_size,n_actions)
    --gradQ(s,action) = r + (1-termina) * gamma * max_a Q_target(s_prime, a) - Q(s, a), for action in a
    --gradQ(s,action) = 0, for action not in a

    local term_flip = term_prime:clone():float():mul(-1):add(1) --terminal -> (1-termina)
    local Q_target = self.target_network:forward(s_prime):clone():float() --Q_target (minibatch_size,n_actions)
    local Q_target_max = Q_target:max(2)  --Q_target_max (minibatch_size,)


    --delta = r + (1-termina) * gamma * max_a Q_target(s_prime, a) - Q(s, a)
    local delta = r:clone():float() --delta = r
    delta:add(Q_target_max:mul(self.discount):cmul(term_flip)) --delta += (1-termina) * gamma * max_a Q_target(s_prime, a)
    local Q = self.target_network:forward(s):float()
    for i=1,Q:size(1) do  --Q:size(1) = minibatch_size
        delta[i] = delta[i] - Q[i][a[i]]
    end


    --clip_data
       ------CODE HERE
    

    local gradQ = torch.zeros(self.minibatch_size, self.n_actions):float()
    for i=1,Q:size(1) do  --Q:size(1) = minibatch_size
        gradQ[i][a[i]] = delta[i]
    end
    return gradQ

end
