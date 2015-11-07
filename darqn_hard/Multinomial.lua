local Multinomial, Parent = torch.class('nn.Multinomial', 'nn.Module')

function Multinomial:__init()
   Parent.__init(self)
   self._index = nil
   self.train = true
   self.gradEntropyTerm = torch.CudaTensor()
   self.h_sampling_mask = torch.CudaTensor()
end

function Multinomial:updateOutput(input)

--   self.output:resizeAs(input):fill(0)
--   self.output:scatter(2,torch.multinomial(input, 1),1)

   self._index = self._index or ((torch.type(input) == 'torch.CudaTensor') and torch.CudaTensor() or torch.LongTensor())

   input.multinomial(self._index, input, 2, true)

   self.output:resizeAs(input):fill(0)
   self.output:scatter(2, self._index[{{},{1}}], 1)

   if self.train then
      self.h_sampling_mask:resize(input:size(1), 1):bernoulli(0.5)
      local h_sampling_mask = self.h_sampling_mask:expandAs(input)
      self.output:cmul(h_sampling_mask)
      self.h_sampling_mask:mul(-1):add(1)
      self.output:addcmul(h_sampling_mask, input)
   end

   return self.output
end

function Multinomial:updateGradInput(input, gradOutput)

   self.gradInput:resizeAs(gradOutput):copy(gradOutput)
   self.gradInput:div(10)

   self.gradInput:cmul(self.output)
   self.gradInput:cdiv(input)

   --self.gradEntropyTerm:resizeAs(input):copy(input):log():add(1)
   --self.gradInput:add(self.gradEntropyTerm:mul(0.0005))

   self.gradInput:mul(-1)
   return self.gradInput
end

function Multinomial:__tostring__()
  return string.format('%s', torch.type(self))
end
