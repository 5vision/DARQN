require 'xlua'
require 'initenv'
require 'cutorch'
require 'cudnn'
require 'DARQN'

op = xlua.OptionParser('pretrain.lua -m <model.t7>')
op:option{'-m', '--model', action='store', dest='model', help='file name of model', default=''}
opt = op:parse()
op:summarize()
file = torch.load(opt.model)
local pretrained_network = file.model.protos.cnn.forwardnodes[10].data.module
local W = pretrained_network:getParameters()
W = W:float()
torch.save('pretrain.t7',{W=W})