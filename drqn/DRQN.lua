require 'nn'
require 'nngraph'

require 'Rectifier'

local CNN = require 'CNN'
local LSTM = require 'LSTM'
local model_utils = require 'model_utils'


local drqn = torch.class('DRQN')


function drqn:__init(args)

	self.minibatch_size = args.minibatch_size
	self.hist_len 		= args.hist_len
	self.height			= args.input_dims[2]
	self.width 			= args.input_dims[3]
    self.gpu            = args.gpu

	self.protos = {}

	self.protos.cnn = CNN.cnn(args)

	-- lstm timestep's input: {x, prev_c, prev_h}, output: {next_c, next_h}
	self.protos.lstm = LSTM.lstm(args)

	self.protos.dqn = nn.Linear(args.rnn_size, args.n_actions)

	if self.gpu >=0 then
		self.protos.cnn:cuda()
		self.protos.lstm:cuda()
		self.protos.dqn:cuda()
	end

	-- put the above things into one flattened parameters tensor
	self.w, self.dw = model_utils.combine_all_parameters(self.protos.cnn, self.protos.lstm, self.protos.dqn)
    self.dw:zero()

	-- make a bunch of clones, AFTER flattening, as that reallocates memory
	self.clones = {}
	for name,proto in pairs(self.protos) do
	    print('cloning '..name)
	    self.clones[name] = model_utils.clone_many_times(proto, args.hist_len, not proto.parameters)
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

	-- LSTM initial state for prediction, note that we're using minibatches OF SIZE ONE here
	self.prev_c = torch.zeros(1, args.rnn_size)
	self.prev_h = self.prev_c:clone()
	if self.gpu >=0 then
		self.prev_c = self.prev_c:cuda()
		self.prev_h = self.prev_h:cuda()
	end
end

function drqn:getParameters()
	return self.w, self.dw
end

function drqn:clone()
	local mem = torch.MemoryFile("w"):binary()
    mem:writeObject(self)
    local reader = torch.MemoryFile(mem:storage(), "r"):binary()
    local clone = reader:readObject()
    reader:close()
    mem:close()
    return clone
end

function drqn:forward(s)
    ------------------- forward pass -------------------
    self.lstm_c = {[0]=self.initstate_c} -- internal cell states of LSTM
    self.lstm_h = {[0]=self.initstate_h} -- output values of LSTM
    
    self.observation = {}         -- input observation
    self.predictions = {}         -- dqn outputs

    local input = s:reshape(self.minibatch_size, self.hist_len, self.height, self.width)

    for t=1,self.hist_len do
        self.observation[t] = self.clones.cnn[t]:forward(input[{{},{t},{},{}}])
        self.lstm_c[t], self.lstm_h[t] = unpack(self.clones.lstm[t]:forward{self.observation[t], self.lstm_c[t-1], self.lstm_h[t-1]})
        self.predictions[t] = self.clones.dqn[t]:forward(self.lstm_h[t])
    end
end

function drqn:backward(s, targets)

    local input = s:reshape(self.minibatch_size, self.hist_len, self.height, self.width)

    -- zero gradients of parameters
    self.dw:zero()

    ------------------ backward pass -------------------
    -- complete reverse order of the above
    local dobservation = {}                           		-- d loss / d input observation
    local dlstm_c = {[self.hist_len]=self.dfinalstate_c}   	-- internal cell states of LSTM
    local dlstm_h = {}                                		-- output values of LSTM
    for t=self.hist_len,1,-1 do
        -- backprop through loss/target, and DQN/linear
        -- Two cases for dloss/dh_t: 
        --   1. h_T is only used once, sent to the DQN (but not to the next LSTM timestep).
        --   2. h_t is used twice, for the DQN and for the next step. To obey the
        --      multivariate chain rule, we add them.
        if t == self.hist_len then
            assert(dlstm_h[t] == nil)
            dlstm_h[t] = self.clones.dqn[t]:backward(self.lstm_h[t], targets[t])
        else
            dlstm_h[t]:add(self.clones.dqn[t]:backward(self.lstm_h[t], targets[t]))
        end

        -- backprop through LSTM timestep
        dobservation[t], dlstm_c[t-1], dlstm_h[t-1] = unpack(self.clones.lstm[t]:backward(
            {self.observation[t], self.lstm_c[t-1], self.lstm_h[t-1]},
            {dlstm_c[t], dlstm_h[t]}
        ))

        -- backprop through CNN
        self.clones.cnn[t]:backward(input[{{},{t},{},{}}], dobservation[t])
    end

    ------------------------ misc ----------------------
    -- transfer final state to initial state (BPTT)
    --self.initstate_c:copy(self.lstm_c[#self.lstm_c])
    --self.initstate_h:copy(self.lstm_h[#self.lstm_h])

    -- clip gradient element-wise
    self.dw:clamp(-5, 5)
end

function drqn:predict(frame)

    local input = frame:reshape(1, 1, self.height, self.width):cuda()
    -- CNN and LSTM 
    local observation = self.protos.cnn:forward(input)
    local next_c, next_h = unpack(self.protos.lstm:forward{observation, self.prev_c, self.prev_h})
    self.prev_c:copy(next_c)
    self.prev_h:copy(next_h)
    -- DQN
    return self.protos.dqn:forward(next_h)
end
