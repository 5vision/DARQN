require 'xlua'
require 'initenv'
require 'cutorch'
require 'cudnn'
require 'image'
require 'DARQN'

op = xlua.OptionParser('filters.lua -m <model.t7>')
op:option{'-m', '--model', action='store', dest='model', help='file name of model', default=''}
opt = op:parse()
op:summarize()
file = torch.load(opt.model)

local cnn = file.model.protos.cnn.forwardnodes[10].data.module
local layer = cnn:get(2)
local weight = layer.weight:float()

print('DQN')
print(get_weight_norms(cnn))
print(get_grad_norms(cnn))

weight = weight:reshape(32,8,8)
map = image.toDisplayTensor{input=weight, padding=1, nrow=4, symmetric=true}
image.save('weights.png', map)
