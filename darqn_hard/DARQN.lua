require 'nn'
require 'nngraph'

require 'Constant'
require 'Rectifier'
require 'Multinomial'
require 'CMulTableHard'
require 'LinearWithoutBias'

local CNN = require 'CNN'
local LSTM = require 'LSTM'
local model_utils = require 'model_utils'


local darqn = torch.class('DARQN')


function darqn:__init(args)

	self.minibatch_size = args.minibatch_size
	self.hist_len 		= args.hist_len
	self.height			= args.input_dims[2]
	self.width 			= args.input_dims[3]
    self.dropout        = args.dropout
    self.gpu            = args.gpu
    self.attention      = args.attention

    self.magicNodeHard = 19
    self.magicNodeDropout = 22


	self.protos = {}

	self.protos.cnn = CNN.cnn(args)

	-- lstm timestep's input: {x, prev_c, prev_h}, output: {next_c, next_h}
	self.protos.lstm = LSTM.lstm(args)

    --self.protos.dqn = nn.Sequential()
    --if self.dropout > 0 then
    --    self.protos.dqn:add(nn.Dropout(self.dropout))
    --end
    self.protos.dqn = nn.Linear(args.rnn_size, args.n_actions)

    self.protos.base = nn.Linear(args.rnn_size, 1)
    self.criterion = nn.MSECriterion() -- baseline criterion
    self.baseline = {}

	if self.gpu >=0 then
		self.protos.cnn:cuda()
		self.protos.lstm:cuda()
		self.protos.dqn:cuda()
		self.protos.base:cuda()
        self.criterion:cuda()
	end

	-- put the above things into one flattened parameters tensor
	self.w, self.dw = model_utils.combine_all_parameters(self.protos.cnn, self.protos.lstm, self.protos.dqn, self.protos.base)
	self.dw:zero()

	-- make a bunch of clones, AFTER flattening, as that reallocates memory
	self.clones = {}
	for name,proto in pairs(self.protos) do
	    print('cloning '..name)
	    self.clones[name] = model_utils.clone_many_times(proto, args.hist_len, not proto.parameters)
	end

    -- disable bernoulli sampling for evaluation
    if self.attention == 'hard' then
        self.protos.cnn.forwardnodes[self.magicNodeHard].data.module.train = false
        if self.clones.cnn[1].forwardnodes[self.magicNodeHard].data.module.train then
            print('Multinomial of CNN clones is set!')
        end
        self.magicNodeDropout = self.magicNodeDropout + 2
    end

    -- disable dropout for evaluation
    if self.dropout > 0 then
        self.protos.cnn.forwardnodes[self.magicNodeDropout].data.module.train = false
        self.protos.dqn:get(1).train = false
        if self.clones.cnn[1].forwardnodes[self.magicNodeDropout].data.module.train then
            print('Dropout of CNN clones is set!')
        end
        if self.clones.dqn[1]:get(1).train then
            print('Dropout of DQN clones is set!')
        end
    end

	-- LSTM initial state (zero initially, but final state gets sent to initial state when we do BPTT)
	self.initstate_c = torch.zeros(args.minibatch_size, args.rnn_size)
	self.initstate_h = self.initstate_c:clone()
	if self.gpu >=0 then
		self.initstate_c = self.initstate_c:cuda()
		self.initstate_h = self.initstate_h:cuda()
	end

	-- LSTM final state's backward message (dloss/dfinalstate) is 0, since it doesn't influence predictions
	self.dfinalstate_c = self.initstate_c:clone()

    self.dattention = torch.zeros(args.minibatch_size, 49)
    if self.gpu >=0 then
        self.dattention = self.dattention:cuda()
    end

	-- LSTM initial state for prediction, note that we're using minibatches OF SIZE ONE here
	self.prev_c = torch.zeros(1, args.rnn_size)
	self.prev_h = self.prev_c:clone()
	if self.gpu >=0 then
		self.prev_c = self.prev_c:cuda()
		self.prev_h = self.prev_h:cuda()
	end
end

function darqn:getParameters()
	return self.w, self.dw
end

function darqn:clone()
	local mem = torch.MemoryFile("w"):binary()
    mem:writeObject(self)
    local reader = torch.MemoryFile(mem:storage(), "r"):binary()
    local clone = reader:readObject()
    reader:close()
    mem:close()
    return clone
end

function darqn:forward(s)

    ------------------- forward pass -------------------
    self.lstm_c = {[0]=self.initstate_c} -- internal cell states of LSTM
    self.lstm_h = {[0]=self.initstate_h} -- output values of LSTM
    
    local attention
    self.observation = {}         -- input observation
    self.predictions = {}         -- dqn outputs
    self.baseline = {}

    local input = s:reshape(self.minibatch_size, self.hist_len, self.height, self.width)

    for t=1,self.hist_len do
        attention, self.observation[t] = unpack(self.clones.cnn[t]:forward{self.lstm_h[t-1], input[{{},{t},{},{}}]})
        self.lstm_c[t], self.lstm_h[t] = unpack(self.clones.lstm[t]:forward{self.observation[t], self.lstm_c[t-1], self.lstm_h[t-1]})
        self.predictions[t] = self.clones.dqn[t]:forward(self.lstm_h[t])
        self.baseline[t] = self.clones.base[t]:forward(self.lstm_h[t])
    end
end

function darqn:backward(s, targets, deltas)

    local input = s:reshape(self.minibatch_size, self.hist_len, self.height, self.width)

    -- zero gradients of parameters
    self.dw:zero()

    ------------------ backward pass -------------------
    -- complete reverse order of the above
    local dobservation = {}                           		-- d loss / d input observation
    local dlstm_c = {[self.hist_len]=self.dfinalstate_c}   	-- internal cell states of LSTM
    local dlstm_h = {}                                		-- output values of LSTM
    for t=self.hist_len,1,-1 do


        local attention_reward = targets[t]
        local dbaseline = self.criterion:backward(self.baseline[t], attention_reward)
        dbaseline:mul(-1)
        self.clones.base[t]:backward(self.lstm_h[t], dbaseline)

        local dattention = attention_reward:add(-1, self.baseline[t])
        --dattention:clamp(-1, 1)
        --dattention = dattention:expandAs(self.dattention)
        local dattention2 = torch.repeatTensor(dattention,1,49)

        -- backprop through loss/deltas, and DQN/linear
        -- Two cases for dloss/dh_t: 
        --   1. h_T is only used once, sent to the DQN (but not to the next LSTM timestep).
        --   2. h_t is used twice, for the DQN and for the next step. To obey the
        --      multivariate chain rule, we add them.
        if t == self.hist_len then
            assert(dlstm_h[t] == nil)
            dlstm_h[t] = self.clones.dqn[t]:backward(self.lstm_h[t], deltas[t])
        else
            dlstm_h[t]:add(self.clones.dqn[t]:backward(self.lstm_h[t], deltas[t]))
        end

        -- backprop through LSTM timestep
        dobservation[t], dlstm_c[t-1], dlstm_h[t-1] = unpack(self.clones.lstm[t]:backward(
            {self.observation[t], self.lstm_c[t-1], self.lstm_h[t-1]},
            {dlstm_c[t], dlstm_h[t]}
        ))

        -- backprop through CNN
        self.clones.cnn[t]:backward(
            {self.lstm_h[t-1], input[{{},{t},{},{}}]},
            {dattention2, dobservation[t]}
        )
    end

    ------------------------ misc ----------------------
    -- transfer final state to initial state (BPTT)
    --self.initstate_c:copy(self.lstm_c[#self.lstm_c])
    --self.initstate_h:copy(self.lstm_h[#self.lstm_h])

    -- clip gradient element-wise
    self.dw:clamp(-5, 5)
end

function darqn:predict(frame)

    local input = frame:reshape(1, 1, self.height, self.width):cuda()
    -- CNN and LSTM 
    local attention, observation = unpack(self.protos.cnn:forward{self.prev_h, input})
    local next_c, next_h = unpack(self.protos.lstm:forward{observation, self.prev_c, self.prev_h})
    self.prev_c:copy(next_c)
    self.prev_h:copy(next_h)
    -- DQN
    local prediction = self.protos.dqn:forward(next_h)

    return attention, prediction
end
