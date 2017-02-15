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
    

    --table for storing the last histLen states, used for constructing fullState more easily
    self.recent_s = {} --table of ByteTensors
    self.recent_t = {}
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

function rplmem:get_mostRecent_fullState()
    local index, use_recent = 1, true --most recent
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
end