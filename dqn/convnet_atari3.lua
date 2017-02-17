require "initenv" --'nn', 'nngraph', 'torch', 'image'

--require this file when initilize NeuralQLearner
--"arg" will be "self" of NeuralQLearner

function create_network(args)

    --convnet params
    args.n_units        = {32, 64, 64} --number of filers
    args.filter_size    = {8, 4, 3}
    args.filter_stride  = {4, 2, 1}
    args.n_hid          = {512}
    args.nl             = nn.ReLU

    local net = nn.Sequential()
    net:add(nn.Reshape(unpack(args.fullState_dims))) --input_dims {self.hist_len, 84, 84}

    --first conv layer
    net:add(nn.SpatialConvolution(args.hist_len, args.n_units[1], 
    	                         args.filter_size[1], args.filter_size[1],
    	                         args.filter_stride[1],args.filter_stride[1],1
    	                         ))
    net:add(args.nl())

    -- Add the rest of convolutional layers
    for i = 1, #args.n_units-1 do
    	net:add(nn.SpatialConvolution(args.n_units[i], args.n_units[i+1], 
    	                         args.filter_size[i+1], args.filter_size[i+1],
    	                         args.filter_stride[i+1],args.filter_stride[i+1]
    	                         ))
    	net:add(args.nl())
    end

    local nel = net:forward( torch.zeros(1, unpack(args.fullState_dims)) ):nElement()

    --add fc layers
    net:add(nn.Reshape(nel))
    net:add(nn.Linear(nel,args.n_hid[1]))
    net:add(args.nl())
    local last_layer_size = args.n_hid[1]
    for i = 1, #args.n_hid-1 do
    	net:add(nn.Linear(args.n_hid[i],args.n_hid[i+1]))
    	net:add(args.nl())
    	last_layer_size = args.n_hid[i+1]
    end

    --final layer: n_actions output
    net:add(nn.Linear(last_layer_size, args.n_actions))

    return net
end

return create_network

