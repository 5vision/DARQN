--[[
Copyright (c) 2014 Google Inc.

See LICENSE file for full terms of limited license.
]]

if not dqn then
    require 'initenv'
end

local nql = torch.class('dqn.NeuralQLearner')


function nql:__init(args)
    self.state_dim  = args.state_dim -- State dimensionality.
    self.n_actions  = #args.actions
    self.verbose    = args.verbose
    self.best       = args.best

    --- epsilon annealing
    self.ep_start   = args.ep or 1
    self.ep         = self.ep_start -- Exploration probability.
    self.ep_end     = args.ep_end or self.ep
    self.ep_endt    = args.ep_endt or 1000000

    ---- learning rate annealing
    self.lr_start       = args.lr or 0.01 --Learning rate.
    self.lr             = self.lr_start
    self.lr_end         = args.lr_end or self.lr
    self.lr_endt        = args.lr_endt or 1000000
    self.wc             = args.wc or 0  -- L2 weight cost.
    self.minibatch_size = args.minibatch_size or 1
    self.valid_size     = self.minibatch_size

    --- Q-learning parameters
    self.discount       = args.discount or 0.99 --Discount factor.
    self.update_freq    = args.update_freq or 1
    -- Number of points to replay per learning step.
    self.n_replay       = args.n_replay or 1
    -- Number of steps after which learning starts.
    self.learn_start    = args.learn_start or 0
     -- Size of the transition table.
    self.replay_memory  = args.replay_memory or 1000000
    self.hist_len       = args.hist_len or 1
    self.rescale_r      = args.rescale_r
    self.max_reward     = args.max_reward
    self.min_reward     = args.min_reward
    self.clip_delta     = args.clip_delta
    self.target_q       = args.target_q
    self.bestq          = 0

    self.gpu            = args.gpu

    self.ncols          = args.ncols or 1  -- number of color channels in input
    self.input_dims     = args.input_dims or {self.ncols, 84, 84}
    self.preproc        = args.preproc -- name of preprocessing network
    self.histType       = args.histType or "linear"  -- history type to use
    self.histSpacing    = args.histSpacing or 1
    self.nonTermProb    = args.nonTermProb or 1
    self.bufferSize     = args.bufferSize or 512

    self.network        = args.network
    self.attention      = args.attention
    self.rnn_size       = args.rnn_size
    self.dropout        = args.dropout or 0
    self._mask 		= nil

    -- check whether there is a network file
    if not (type(self.network) == 'string') then
        error("The type of the network provided in NeuralQLearner" ..
              " is not a string!")
    end

    local msg, err = pcall(require, self.network)
    if not msg then
        -- try to load saved agent
        if self.gpu and self.gpu >= 0 then
            require 'cudnn'
        end
        require 'DARQN'
        local err_msg, exp = pcall(torch.load, self.network)
        if not err_msg then
            error("Problem with loading network file "..self.network)
        end
        if self.best and exp.best_model then
            self.network = exp.best_model
        else
            self.network = exp.model
        end
    else
        print('Creating Agent Network from ' .. self.network)
        self.network = err
        self.network = self:network()
    end

    -- Load preprocessing network.
    if not (type(self.preproc == 'string')) then
        error('The preprocessing is not a string')
    end
    msg, err = pcall(require, self.preproc)
    if not msg then
        error("Error loading preprocessing net")
    end
    self.preproc = err
    self.preproc = self:preproc()
    self.preproc:float()

    -- Create transition table.
    ---- assuming the transition table always gets floating point input
    ---- (Foat or Cuda tensors) and always returns one of the two, as required
    ---- internally it always uses ByteTensors for states, scaling and
    ---- converting accordingly
    local transition_args = {
        stateDim = self.state_dim, numActions = self.n_actions,
        histLen = self.hist_len, gpu = self.gpu,
        maxSize = self.replay_memory, histType = self.histType,
        histSpacing = self.histSpacing, nonTermProb = self.nonTermProb,
        bufferSize = self.bufferSize
    }

    self.transitions = dqn.TransitionTable(transition_args)

    self.numSteps = 0 -- Number of perceived states.
    self.lastFrame = nil
    self.lastAction = nil
    self.v_avg = 0 -- V running average.
    self.tderr_avg = 0 -- TD error running average.

    self.q_max = 1
    self.r_max = 1
    
    self.w, self.dw = self.network:getParameters()
    self.dw:zero()
    self.deltas = self.dw:clone():fill(0)
    self.tmp    = self.dw:clone():fill(0)
    self.g      = self.dw:clone():fill(0)
    self.g2     = self.dw:clone():fill(0)

    print('Number of all parameters: '..self.w:nElement())

    if self.target_q then
        self.target_network = self.network:clone()
    end
end


function nql:reset(state)
    if not state then
        return
    end
    self.best_network = state.best_network
    self.network = state.model
    self.w, self.dw = self.network:getParameters()
    self.dw:zero()
    self.numSteps = 0
    print("RESET STATE SUCCESFULLY")
end


function nql:getQUpdate(args)
    local q = args.q
    local a = args.a
    local r = args.r
    local q2 = args.q2
    local term = args.term

    -- q2_max = (1-terminal) * gamma * max_a Q(s2, a)
    -- target = r + q2_max
    -- delta  = target - Q(s, a)

    term = term:clone():mul(-1):add(1)

    local q2_max = q2:clone():max(2):mul(self.discount):cmul(term)

    local target = r:clone()
    if self.rescale_r then
        target:div(self.r_max)
    end
    target:add(q2_max)

    local delta = torch.repeatTensor(target,1,self.n_actions) - q
    
    self._mask = self._mask or q:clone()
    self._mask:fill(0):scatter(2,a,1)
    delta:cmul(self._mask)

    if self.clip_delta then
        delta:clamp(-self.clip_delta, self.clip_delta)
    end

    return target, delta, q2_max
end


function nql:qLearnMinibatch()
    -- Perform a minibatch Q-learning update:
    -- w += alpha * (r + gamma max Q(s2,a2) - Q(s,a)) * dQ(s,a)/dw
    assert(self.transitions:size() > self.minibatch_size)

    local s, a, r, s2, term = self.transitions:sample(self.minibatch_size)

    self.network:forward(s)
    self.target_network:forward(s2)

    local deltas = {}
    local targets = {}
    for t=1,self.hist_len do
        targets[t], deltas[t] = self:getQUpdate{
            q=self.network.predictions[t],
            a=a[{{},{t}}],
            r=r[{{},{t}}],
            q2=self.target_network.predictions[t],
            term=term[{{},{t}}]
        }
    end

    -- get new gradient
    self.network:backward(s, targets, deltas)

    -- add weight cost to gradient
    self.dw:add(-self.wc, self.w)

    -- compute linearly annealed learning rate
    local t = math.max(0, self.numSteps - self.learn_start)
    self.lr = (self.lr_start - self.lr_end) * (self.lr_endt - t)/self.lr_endt + self.lr_end
    self.lr = math.max(self.lr, self.lr_end)

    -- use gradients
    self.g:mul(0.95):add(0.05, self.dw)
    self.tmp:cmul(self.dw, self.dw)
    self.g2:mul(0.95):add(0.05, self.tmp)
    self.tmp:cmul(self.g, self.g)
    self.tmp:mul(-1)
    self.tmp:add(self.g2)
    self.tmp:add(0.01)
    self.tmp:sqrt()

    -- accumulate update
    self.deltas:mul(0):addcdiv(self.lr, self.dw, self.tmp)
    self.w:add(self.deltas)
end


function nql:sample_validation_data()
    local s, a, r, s2, term = self.transitions:sample(self.valid_size)
    self.valid_s    = s:clone()
    self.valid_a    = a:clone()
    self.valid_r    = r:clone()
    self.valid_s2   = s2:clone()
    self.valid_term = term:clone()
end


function nql:compute_validation_statistics()

    self.network:forward(self.valid_s)
    self.target_network:forward(self.valid_s2)

    local targets, deltas, q2_max = self:getQUpdate{
        q=self.network.predictions[self.hist_len],
        a=self.valid_a[{{},{self.hist_len}}],
        r=self.valid_r[{{},{self.hist_len}}],
        q2=self.target_network.predictions[self.hist_len],
        term=self.valid_term[{{},{self.hist_len}}]
    }

    self.v_avg = self.q_max * q2_max:float():mean()
    self.tderr_avg = deltas:float():abs():mean()
end


function nql:perceive(reward, rawstate, terminal, testing, testing_ep)
    -- Preprocess state (will be set to nil if terminal)

    local frame = self.preproc:forward(rawstate:float()):clone():float()

    if self.max_reward then
        reward = math.min(reward, self.max_reward)
    end
    if self.min_reward then
        reward = math.max(reward, self.min_reward)
    end
    if self.rescale_r then
        self.r_max = math.max(self.r_max, reward)
    end

    --Store transition s, a, r, s'
    if self.lastFrame and not testing then
        self.transitions:add(self.lastFrame, self.lastAction, reward, self.lastTerminal)
    end

    if self.numSteps == self.learn_start+1 and not testing then
        self:sample_validation_data()
    end

    -- Select action
    local actionIndex = 1
    if not terminal then
        actionIndex = self:eGreedy(frame, testing_ep)
    end

    --Do some Q-learning updates
    if self.numSteps > self.learn_start and not testing and
        self.numSteps % self.update_freq == 0 then
        for i = 1, self.n_replay do
            self:qLearnMinibatch()
        end
    end

    if not testing then
        self.numSteps = self.numSteps + 1
    end

    self.lastFrame = frame:clone()
    self.lastAction = actionIndex
    self.lastTerminal = terminal

    if self.target_q and self.numSteps % self.target_q == 1 then
        self.target_network = self.network:clone()
    end

    if not terminal then
        return actionIndex
    else
        return 0
    end
end


function nql:eGreedy(frame, testing_ep)
    self.ep = testing_ep or (self.ep_end +
                math.max(0, (self.ep_start - self.ep_end) * (self.ep_endt -
                math.max(0, self.numSteps - self.learn_start))/self.ep_endt))
    -- Epsilon greedy
    if torch.uniform() < self.ep then
        return torch.random(1, self.n_actions)
    else
        return self:greedy(frame)
    end
end


function nql:greedy(frame)

    local attention, q = self.network:predict(frame)

    self.attention = attention

    q = q:float():squeeze()

    local maxq = q[1]
    local besta = {1}

    -- Evaluate all other actions (with random tie-breaking)
    for a = 2, self.n_actions do
        if q[a] > maxq then
            besta = { a }
            maxq = q[a]
        elseif q[a] == maxq then
            besta[#besta+1] = a
        end
    end
    self.bestq = maxq

    local r = torch.random(1, #besta)

    return besta[r]
end


function nql:report()
    local cnn = self.network.protos.cnn.forwardnodes[10].data.module
    print('CNN')
    print(get_weight_norms(cnn))
    print(get_grad_norms(cnn))
    local att = self.network.protos.cnn.forwardnodes[15].data.module
    print('ATT')
    print(get_weight_norms(att))
    print(get_grad_norms(att))
    print('DQN')
    print(get_weight_norms(self.network.protos.dqn))
    print(get_grad_norms(self.network.protos.dqn))
    if #self.network.baseline > 0 then
        print(string.format('BASE: %.3f', self.network.baseline[1]:float():mean()))
    end
end
