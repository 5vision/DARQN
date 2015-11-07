    
require 'nnx'
require 'image'

local screen = image.load('sample.png')
screen = image.scale(screen, 160, 210, 'bilinear')

local upsample = nn.SpatialReSampling{owidth=84,oheight=84,mode='bilinear'}
upsample:float()

local attention = torch.FloatTensor(7, 7):zero()
attention[{{7},{7}}] = 1

local attention_mask = torch.FloatTensor(1, 9, 9):zero()
attention_mask[{{1},{2,8},{2,8}}] = attention
attention = upsample:forward(attention_mask)

attention = image.scale(attention, 160, 210, 'bilinear')[1]

attention = attention * 2.5

local photo = screen[1]:float() + attention
--photo[1] = photo[1] + attention
--photo[2] = photo[2] + attention
--photo[3] = photo[3] + attention

photo:clamp(0,1)
image.save('upsample.png', photo)