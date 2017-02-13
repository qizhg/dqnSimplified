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
cmd:option('-env_params', '', 'string of environment parameters')

cmd:option('-agent_name', '', 'name of the class defining the agent')
cmd:option('-q_net', '', 'file defining the q net')
cmd:option('-preproc_net', '', 'file defining the q net')
--cmd:option('-pool_frms', '','string of frame pooling parameters (e.g.: size=2,type="max")')
--cmd:option('-actrep', 1, 'how many times to repeat action')
--cmd:option('-random_starts', 0, 'play action 0 between 1 and random_starts ' ..'number of times at the start of each training episode')

--cmd:option('-name', '', 'filename used for saving network and training history')
--cmd:option('-network', '', 'reload pretrained network')
--cmd:option('-agent', '', 'name of agent file to use')
--cmd:option('-agent_params', '', 'string of agent parameters')
--cmd:option('-seed', 1, 'fixed input seed for repeatable experiments')
--cmd:option('-saveNetworkParams', false,'saves the agent network in a separate file')
--cmd:option('-prog_freq', 5*10^3, 'frequency of progress output')
--cmd:option('-save_freq', 5*10^4, 'the model is saved every save_freq steps')
--cmd:option('-eval_freq', 10^4, 'frequency of greedy evaluation')
--cmd:option('-save_versions', 0, '')

--cmd:option('-steps', 10^5, 'number of training steps to perform')
--cmd:option('-eval_steps', 10^5, 'number of evaluation steps')

--cmd:option('-verbose', 2,'the higher the level, the more information is printed to screen')
--cmd:option('-threads', 1, 'number of BLAS threads')
--cmd:option('-gpu', -1, 'gpu flag')

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
