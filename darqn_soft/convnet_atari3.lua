--[[
Copyright (c) 2014 Google Inc.

See LICENSE file for full terms of limited license.
]]

require 'DARQN'

return function(args)
    args.n_units        = {32, 64, args.rnn_size}
    args.filter_size    = {8, 4, 3}
    args.filter_stride  = {4, 2, 1}
    args.nl             = nn.Rectifier
    return DARQN(args)
end

