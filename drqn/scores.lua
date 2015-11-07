require 'xlua'
require 'initenv'
require 'cutorch'
require 'cudnn'
require 'DRQN'

op = xlua.OptionParser('scores.lua -m <model.t7>')
op:option{'-m', '--model', action='store', dest='model', help='file name of model', default=''}
opt = op:parse()
op:summarize()
file = torch.load(opt.model)

h = file.reward_history
for i = 1,#h do
  print(string.format('%02d: %.3f',i,h[i]))
end
