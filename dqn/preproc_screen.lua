require "initenv" --'nn', 'image'


--require this file when initilize NeuralQLearner
--"arg" will be "self" of NeuralQLearner


--Define nn.Scale class
local scale = torch.class('nn.Scale', 'nn.Module')

function scale:__init(width, height)
	self.width = width
	self.height = height
end

function scale:updateOutput(input) -- forward
	
	--input: (1,3,210,160) in alewrap environment
	local x = input
	if x:dim() >3 then
		x = x[1]
	end

	x = image.rgb2y(x)
	x = image.scale(x, self.width, self.height)
	return x --(1,84,84)
end
--End of Define nn.Scale class

function create_network(args)

	--width and height after pre_proc
	local width, height  = 84, 84
	return nn.Scale(width, height)
end


return create_network
