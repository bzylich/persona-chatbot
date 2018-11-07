# obtained from http://parl.ai/static/docs/basic_tutorial.html
import random

from parlai.core.agents import Agent
from parlai.core.params import ParlaiParser
from parlai.core.worlds import create_task

# actual chatting object
class RepeatLabelAgent(Agent):
    # #
    # initialize by setting id
    # #
    def __init__(self, opt):
        self.id = 'LabelAgent'
    # #
    # store observation for later, return it unmodified
    # #
    def observe(self, observation):
        self.observation = observation
        # print(observation)
        return observation
    # #
    # return label from before if available
    # #
    # def act(self):
    #     reply = {'id': self.id}
    #     if 'labels' in self.observation:
    #         reply['text'] = ', '.join(self.observation['labels'])
    #     else:
    #         reply['text'] = "I don't know."
    #     return reply
    
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