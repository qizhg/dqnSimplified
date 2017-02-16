if not dqn then
    require 'initenv'
end

local rplmem = torch.class('dqn.ReplayMemory')


function rplmem:__init(args)
    self.maxSize = args.maxSize or 1024^2 --unit is stateDim, i.e. dim of a frame after preproc
    self.numEntries = 0 --curent size
    self.insertIndex = 0 --the index to add one more transition
    self.histLen = args. histLen or ^4
    self.stateDim = args.stateDim or 84*84 --84 x 84, frame size after preproc
    self.fullStateDims = args.fullStateDims  or {self.histLen, 84, 84}
    self.numActions = args.numActions

    --the whole memory
    self.s = torch.ByteTensor(self.maxSize, self.stateDim):fill(0) --ByteTensor, integer in [0,255]
    self.t = torch.ByteTensor(self.maxSize):fill(0)
    self.a = torch.LongTensor(self.maxSize):fill(0)
    self.r = torch.zeros(self.maxSize)
    

    --table for storing the last histLen states, used for constructing current fullState more easily
    self.recent_s = {} --table of ByteTensors
    self.recent_t = {}

    --buffer for sampling fullState transitions
    self.bufferSize = args.bufferSize or 1024
    self.buf_a      = torch.LongTensor(self.bufferSize):fill(0)
    self.buf_r      = torch.zeros(self.bufferSize)
    self.buf_term   = torch.ByteTensor(self.bufferSize):fill(0)
    self.buf_s      = torch.ByteTensor(self.bufferSize, self.histLen * self.stateDim):fill(0)
    self.buf_s_prime= torch.ByteTensor(self.bufferSize, self.histLen * self.stateDim):fill(0)


end

function rplmem:reset()
    self.numEntries = 0
end

function rplmem:size()
    return self.numEntries = 0
end


function rplmem:add(s,term,a,r)

    if self.numEntries < self.maxSize then
        self.numEntries = self.numEntries + 1
    end

    self.insertIndex = self.insertIndex + 1 --insert at the next position
    
    if self.insertIndex > self.maxSize then
        self.insertIndex = 1
    end

    self.s[self.insertIndex] = s:clone():view(x:nElement()):float():mul(255):byte()
    if term then 
        self.t[self.insertIndex] = 1
    else
        self.t[self.insertIndex] = 0
    end
    self.a[self.insertIndex] = a
    self.r[self.insertIndex] = r
end

function rplmem:add_recent_state(s,term)

    local s = s:clone():view(x:nElement()):float():mul(255):byte()
    
    if #self.recent_s == 0 then --first call, #self.recent_t ==0
        for i = 1, self.histLen do
            table.insert(self.recent_s, s:clone:zero()) --init with zero
            table.insert(self.recent_t, 1)  --init with term = 1
        end
    end

    table.insert(self.recent_s, s)
    if term then
        table.insert(self.recent_t, 1)
    else
        table.insert(self.recent_t, 0)
    end

    -- Keep histLen states.
    if #self.recent_s > self.histLen then
        table.remove(self.recent_s, 1)
        table.remove(self.recent_t, 1)
    end
end

function rplmem:get_current_fullState()
    local index, use_recent = 1, true --current
    return self.stackStates(index, use_recent):float():div(255)
end


function rplmem:stackStates(index, use_recent)
    local s,t
    if use_recent then
        s, t = self.recent_s, self.recent_t
    else
        s, t = self.s, self,t
    end

    local fullState = s[1].new() -- same type tensor of s[1] with no dimenstion
    fullState:resize(unpack(self.fullStateDims)):zero()   --(self.histLen, 84, 84)

    --stack index, index + 1,..., index + histLen -1 frames in s, within the same episode
    local episode_start = self.histLen
    
    for i=self.histLen-1,1,-1 do
        
        if t[index + i -1] = 1 then
            break
        end

        episode_start = i
    end

    -- Copy frames from the current episode.
    for i=episode_start,self.histLen do
        fullstate[i] = s[index+i-1]:clone():view(self.fullStateDims[2],self.fullStateDims[3])
    end

    return fullState

end


function rplmem:sample(batch_size)
--sample batch_size transitions from the buffer
    local batch_size = batch_size or 1
    assert(batch_size < self.bufferSize)

    if not self.buf_index or self.buf_index + batch_size -1 > self.bufferSize then
        self:fill_buffer() --re-sample bufferSize fullState transitions into buffer
    end

    local index = self.buf_index
    local range ={{index, index + batch_size -1}}
    self.buf_index = self.buf_index + batch_size

    local buf_s, buf_s_prime, buf_a, buf_r, buf_term = self.buf_s, self.buf_s_prime,
        self.buf_a, self.buf_r, self.buf_term

    return buf_s[range], buf_a[range], buf_r[range], buf_s_prime[range],buf_term[range] 
end

function rplmem:fill_buffer()
    
    assert(self.numEntries >= self.bufferSize)

    --set self.buf_index
    self.buf_index = 1

    --sample bufferSize fullState transitions into buffer
    for buf_index = 1, self.bufferSize do
        local s, a, r, s_prime, term_prime = self.sample_one()
        self.buf_s[buf_ind]:copy(s)
        self.buf_a[buf_ind] = a
        self.buf_r[buf_ind] = r
        self.buf_s_prime[buf_ind]:copy(s_prime)
        self.buf_term[buf_ind] = term_prime
    end

    --conver buf_s, buf_s_prime to floatTensor in range [0,1]
    self.buf_s  = self.buf_s:float():div(255)
    self.buf_s_prime = self.buf_s_prime:float():div(255)
end

function rplmem:sample_one()
    assert(self.numEntries > 1) --??? >4
    local index
    local valid = false
    while not valid do
        -- start at 2 because of previous action ???
        index = torch.random(2, self.numEntries-self.recentMemSize)
        
        if self.t[index+self.recentMemSize-1] == 0 then
            valid = true --must be non-term state
        end
    end

    return self:get_fullState_transition(index)
end

function rplmem:get_fullState_transition(index)
    local s = self:stackStates(index,false)
    local s_prime = self:stackStates(index+1,false)
    local ar_index = index+self.recentMemSize-1

    return s, self.a[ar_index], self.r[ar_index], s_prime, self.t[ar_index+1]
end
