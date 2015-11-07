--[[
Copyright (c) 2014 Google Inc.

See LICENSE file for full terms of limited license.
]]

if not dqn then
    require "initenv"
end

require 'image'

local cmd = torch.CmdLine()
cmd:text()
cmd:text('Test Agent in Environment:')
cmd:text()
cmd:text('Options:')

cmd:option('-framework', '', 'name of training framework')
cmd:option('-env', '', 'name of environment to use')
cmd:option('-game_path', '', 'path to environment file (ROM)')
cmd:option('-env_params', '', 'string of environment parameters')
cmd:option('-pool_frms', '',
           'string of frame pooling parameters (e.g.: size=2,type="max")')
cmd:option('-actrep', 1, 'how many times to repeat action')
cmd:option('-random_starts', 0, 'play action 0 between 1 and random_starts ' ..
           'number of times at the start of each training episode')

cmd:option('-name', '', 'filename used for saving network and training history')
cmd:option('-network', '', 'reload pretrained network')
cmd:option('-agent', '', 'name of agent file to use')
cmd:option('-agent_params', '', 'string of agent parameters')
cmd:option('-seed', 1, 'fixed input seed for repeatable experiments')
cmd:option('-saveNetworkParams', false,
           'saves the agent network in a separate file')
cmd:option('-prog_freq', 5*10^3, 'frequency of progress output')
cmd:option('-save_freq', 5*10^4, 'the model is saved every save_freq steps')
cmd:option('-eval_freq', 10^4, 'frequency of greedy evaluation')
cmd:option('-save_versions', 0, '')

cmd:option('-steps', 10^5, 'number of training steps to perform')
cmd:option('-eval_steps', 10^5, 'number of evaluation steps')

cmd:option('-verbose', 2,
           'the higher the level, the more information is printed to screen')
cmd:option('-threads', 1, 'number of BLAS threads')
cmd:option('-gpu', -1, 'gpu flag')
cmd:option('-best', false,'bet network')

cmd:text()

local opt = cmd:parse(arg)

--- General setup.
local game_env, game_actions, agent, opt = setup(opt)

-- override print to always flush the output
local old_print = print
local print = function(...)
    old_print(...)
    io.flush()
end


local total_reward = 0
local nrewards = 0
local nepisodes = 0
local episode_reward = 0


screen, reward, terminal = game_env:getState()


local actions_count = torch.IntTensor(agent.n_actions):zero()

local upsample = nn.Sequential()
upsample:add(nn.SpatialSubSampling(1,8,8,4,4))
upsample:add(nn.SpatialSubSampling(1,4,4,2,2))
upsample:add(nn.SpatialSubSampling(1,3,3,1,1))
upsample:float()
local w, dw = upsample:getParameters()
w:fill(0.33)
dw:zero()
local empty = torch.zeros(1,84,84):float()
upsample:forward(empty)

local eval_time = sys.clock()
for estep=1,opt.eval_steps do

    local action_index = agent:perceive(reward, screen, terminal, true, 0.05)

    local attention = upsample:updateGradInput(empty,agent.attention:float())
    attention = image.scale(attention, 160, 210, 'bilinear')
    local photo = screen[1]:float()
    photo:add(1, attention:expandAs(photo)):clamp(0,1)
    image.save('../images/'..estep..'.png', photo)

    actions_count[agent.lastAction] = actions_count[agent.lastAction] + 1

    -- Play game in test mode (episodes don't end when losing a life)
    screen, reward, terminal = game_env:step(game_actions[action_index])

    if estep%1000 == 0 then collectgarbage() end

    -- record every reward
    episode_reward = episode_reward + reward
    if reward ~= 0 then
       nrewards = nrewards + 1
    end

    if terminal then
        total_reward = total_reward + episode_reward
        episode_reward = 0
        nepisodes = nepisodes + 1
        screen, reward, terminal = game_env:nextRandomGame()
	break
    end
end
eval_time = sys.clock() - eval_time

total_reward = total_reward/math.max(1, nepisodes)

print(actions_count)

print(string.format('Reward: %.2f, last: %.2f, #ep.: %d,  #rewards: %d, testing time/rate: %ds/%dfps',
    total_reward, episode_reward, nepisodes, nrewards, eval_time, opt.actrep*opt.eval_steps/eval_time))
