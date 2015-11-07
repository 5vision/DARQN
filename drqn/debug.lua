require 'DRQN'
require 'optim'
require 'image'

local CNN = require 'CNN'

local opt = {}
opt.gpu				= -1 
opt.verbose			= 200
opt.minibatch_size	= 2
opt.hist_len		= 4
opt.ncols			= 1
opt.n_actions		= 18
opt.input_dims		= {1, 84, 84}
opt.n_units        	= {32, 64, 16}
opt.filter_size    	= {8, 4, 3}
opt.filter_stride  	= {4, 2, 1}
opt.rnn_size		= 256
opt.nl             	= nn.Rectifier
opt.attention 		= 'hard'
opt.dropout 		= 0.5


local s = torch.Tensor(opt.minibatch_size, opt.hist_len, 84, 84)
local game = 'seaquest'
--[[local f = 98
for e = 1,opt.minibatch_size do
	for k = 1,opt.hist_len do
		local file = string.format('/home/lytic/%s/1_%d.png',game,f)
		s[{e,{k,{{}}}}] = image.scale(image.load(file),84,84,'bilinear')[1]
		f = f + 1
	end
end
]]
if opt.gpu >=0 then
	require 'cunn'
	require 'cutorch'
	cutorch.setDevice(opt.gpu)
	print('GPU: '..cutorch.getDevice())
	s = s:cuda()
end

local network2 = DRQN(opt)

local target_network = network2:clone()

network2:forward(s)

local w, dw = network2:getParameters()
local tw, tdw = target_network:getParameters()

print(w:min())
print(tw:min())

local targets = {}
targets[1] = torch.randn(2,opt.n_actions)
targets[2] = torch.randn(2,opt.n_actions)
targets[3] = torch.randn(2,opt.n_actions)
targets[4] = torch.randn(2,opt.n_actions)

network2:backward(s,targets)

local function opfunc2(x)
	return network2:getParameters()
end

state          = {}
config         = {eps=0.1,rho=0.95}

optim.adadelta(opfunc2, w, config, state)

print(w:min())
print(tw:min())

print(state)

--local attention, q = network2:predict(s[1][1])
--print('Attention:')
--print(attention)
--print('Q')
--print(q)