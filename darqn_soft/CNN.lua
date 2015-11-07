-- adapted from: convnet.lua

local CNN = {}

function CNN.cnn(args)

    local network = nn.Sequential()
    network:add(nn.Reshape(unpack(args.input_dims)))

    --- first convolutional layer
    local convLayer = nn.SpatialConvolution
    if args.gpu >= 0 then
        require 'cutorch'
        require 'cudnn'
        convLayer = cudnn.SpatialConvolution
    end

    network:add(convLayer(args.ncols, args.n_units[1],
                        args.filter_size[1], args.filter_size[1],
                        args.filter_stride[1], args.filter_stride[1],1))
    network:add(args.nl())

    local last_num_features = 0

    -- Add convolutional layers
    for i=1,(#args.n_units-1) do
        -- second convolutional layer
        network:add(convLayer(args.n_units[i], args.n_units[i+1],
                            args.filter_size[i+1], args.filter_size[i+1],
                            args.filter_stride[i+1], args.filter_stride[i+1]))
        network:add(args.nl())
        last_num_features = args.n_units[i+1]
    end

    if args.verbose >= 2 then
        print(network)
    end

    -- batch (32) x features (64) x height (7) x width (7)

    local n_hiddens = last_num_features

    local x = nn.Identity()()
    local x1 = nn.Identity()(network(x))
    local x2 = nn.Reshape(last_num_features, 49)(x1)
    -- batch (32) x features (64) x annotations (49)
    local x3 = convLayer(last_num_features, n_hiddens, 1, 1, 1, 1)(x1)
    -- batch (32) x hiddens (42) x height (7) x width (7)

    local h = nn.Identity()()
    local h1 = nn.LinearWithoutBias(args.rnn_size, n_hiddens)(h)
    -- batch (32) x hiddens (42)
    local h2 = nn.Replicate(49, 3)(h1)
    -- batch (32) x hiddens (42) x annotations (49)
    local h3 = nn.Reshape(n_hiddens, 7, 7)(h2)
    -- batch (32) x hiddens (42) x height (7) x width (7)

    local a1 = nn.Tanh()(nn.CAddTable()({h3, x3}))
    local a2 = convLayer(n_hiddens, 1, 1, 1, 1, 1)(a1)
    -- batch (32) x softmax (1) x height (7) x width (7)
    local a3 = nn.SoftMax()(nn.Reshape(49)(a2))
    -- batch (32) x annotations (49)
    
    local a4 = nn.Replicate(last_num_features, 2)(a3)
    -- batch (32) x features (64) x annotations (49)

    local context = nn.Sum(3)(nn.CMulTable()({a4, x2}))
    -- batch (32) x features (64)

    local g = nn.gModule({h,x}, {a3,context})

    if args.gpu >=0 then
        g:cuda()
    end

    --if args.verbose >= 100 then
    --    graph.dot(g.fg, 'Forward Graph', 'fg')
    --    graph.dot(g.bg, 'Backward Graph', 'bg')
    --end

    return g
end

return CNN
