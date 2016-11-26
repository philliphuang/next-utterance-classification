import editdistance
from operator import itemgetter


train = '''Hello
Hi!
How are you?
I am doing well, how about you?
Great!
How are you today?
I am pretty good
What's up?
I need to poop
'''

train_data = train.split('\n')

while True:
	s = raw_input('Message: ')

	dists = []

	for past_m in train_data:
		dists.append(editdistance.eval(past_m, s))

	best_ind = min(enumerate(dists), key=itemgetter(1))[0] 

	if best_ind == len(train_data):
		print 'Pizzabot: ' + train_data[0]
	else: 
		print 'Pizzabot: ' + train_data[best_ind + 1]
