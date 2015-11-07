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

    local nel
    if args.gpu >= 0 then
        nel = network:cuda():forward(torch.zeros(1,unpack(args.input_dims))
                :cuda()):nElement()
    else
        nel = network:forward(torch.zeros(1,unpack(args.input_dims))):nElement()
    end

    -- reshape all feature planes into a vector per example
    network:add(nn.Reshape(nel))

    network:add(nn.Linear(nel, args.rnn_size))
    network:add(args.nl())

    if args.verbose >= 2 then
        print(network)
    end

    return network
end

return CNN
