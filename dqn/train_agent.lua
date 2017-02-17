--[[
Copyright (c) 2014 Google Inc.

See LICENSE file for full terms of limited license.
]]

require "initenv" --dqn.NeuralQLearner, 'nn', 'nngraph','torch','image'

local cmd = torch.CmdLine()
cmd:text()
cmd:text('Train Agent in Environment:')
cmd:text()
cmd:text('Options:')

cmd:option('-env', '', 'name of environment to use')
cmd:option('-framework', '', 'name of training framework')
cmd:option('-game_path', '', 'path to environment file (ROM)')
cmd:option('-actrep', 4, 'epeat each action 4 times')
cmd:option('-random_starts', 30, 'for every new episode, play null actions a random number of time [0,30]')

cmd:option('-agent_name', '', 'name of the class defining the agent')
cmd:option('-q_net', '', 'file defining the q net')
cmd:option('-preproc_net', '', 'file defining the q net')


cmd:option('-steps',10^5, 'total steps to run')




cmd:text()

local args = cmd:parse(arg)

---torch setup, tensorType, numthreads, seed
args.tensorType =  args.tensorType or 'torch.FloatTensor'
torch.setdefaulttensortype(args.tensorType)
args.threads = args.threads or 4
torch.setnumthreads(args.threads)
args.seed = args.seed or 1
torch.manualSeed(args.seed)

---game environment setup
local framework = require(args.framework)
local game_env = framework.GameEnvironment(args)
local game_actions = game_env:getActions()

---init NeuralQLearner
args.actions = game_env:getActions()
local agent = dqn[args.agent_name](args) --args.agent_name = "NeuralQLearner"


--run steps
local step = 0
local screen, reward, terminal = game_env:getState()
while step < args.steps do
	step = step + 1
	print(step)
	local action_index = agent:perceive(screen, reward, terminal)

	if not terminal then
		screen, reward, terminal = game_env:step(game_actions[action_index],true) --true: training mode, losing a life = episode end = terminal 
	else
		if args.random_starts and args.random_starts>0 then
			screen, reward, terminal = game_env:nextRandomGame()
		else
			screen, reward, terminal = game_env:newGame()
		end
	end
end