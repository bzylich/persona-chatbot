# obtained from http://parl.ai/static/docs/basic_tutorial.html

# as of 11/26/2018 (ver. 1) this code is made only to display the current task that we are operating on
# -> in this case, convai2
# future versions will work on implementing actual use of the task

import random

from parlai.core.agents import Agent
from parlai.core.params import ParlaiParser
from parlai.core.worlds import create_task

from parlai.core.dict import DictionaryAgent
import torch

import copy

# actual chatting object
class RepeatLabelAgent(Agent):
    # #
    # initialize by setting id
    # #
    def __init__(self, opt):
        self.id = 'LabelAgent'
        self.dict = DictionaryAgent(opt)
        self.episode_done = False # first observation is not going to be the end of an episode
        # we use EOS markers to break input and output and end our output
    #     self.EOS = self.dict.eos_token
    #     self.observation = {'text': self.EOS, 'episode_done': True}
    #     self.EOS_TENSOR = torch.LongTensor(self.dict.parse(self.EOS))
    
    # # LEARNING OBSERVATION METHOD (sets up observation to be used in batch learning in batch_act method):
    # def observe(self, observation):
    #     observation = copy.deepcopy(observation)
    #     if not self.episode_done:
    #         # if the last example wasn't the end of an episode, then we need to
    #         # recall what was said in that example
    #         prev_dialogue = self.observation['text']
    #         observation['text'] = prev_dialogue + '\n' + observation['text']
    #     self.observation = observation
    #     self.episode_done = observation['episode_done']
    #     return observation
    
    # def act(self):
    #     # call batch_act with this batch of one
    #     return self.batch_act([self.observation])[0]


    # # actual learning act method
    # def batch_act(self, observations):
    #     batchsize = len(observations)
    #     # initialize a table of replies with this agent's id
    #     batch_reply = [{'id': self.getID()} for _ in range(batchsize)]

    #     # convert the observations into batches of inputs and targets
    #     # valid_inds tells us the indices of all valid examples
    #     # e.g. for input [{}, {'text': 'hello'}, {}, {}], valid_inds is [1]
    #     # since the other three elements had no 'text' field
    #     xs, ys, valid_inds = self.batchify(observations)

    #     if len(xs) == 0:
    #         # no valid examples, just return the empty responses we set up
    #         return batch_reply

    #     # produce predictions either way, but use the targets if available
    #     predictions = self.predict(xs, ys)

    #     for i in range(len(predictions)):
    #         # map the predictions back to non-empty examples in the batch
    #         # we join with spaces since we produce tokens one at a time
    #         batch_reply[valid_inds[i]]['text'] = ' '.join(
    #             c for c in predictions[i] if c != self.EOS)

    #     return batch_reply

    # DEFAULT SIMPLE RETURN OBSERVATION METHOD (NO LEARNING INVOLVED)
    # #
    # store observation for later, return it unmodified
    # #
    def observe(self, observation):
        self.observation = observation
        # print(observation)
        return observation
    
    # NON-LEARNING RANDOM LABEL REPLY
    # random label reply because it shouldn't know the labels in the first place
    def act(self):
        reply = {'id': self.id}

        # question = self.observation[]

        if 'labels' in self.observation:
            reply['text'] = ', '.join(self.observation['labels'])
        
        elif 'label_candidates' in self.observation:
            cands = self.observation['label_candidates']
            reply['text'] = random.choice(cands)
        else:
            reply['text'] = "I don't know."
        return reply

# display loop 
parser = ParlaiParser()
opt = parser.parse_args()

# talker bro
agent = RepeatLabelAgent(opt)
# talker bro's home
world = create_task(opt, agent)

for _ in range(10):
    world.parley()
    print(world.display())
    if world.epoch_done():
        print('EPOCH DONE')
        break