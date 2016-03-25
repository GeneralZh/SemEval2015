import random
r = [str(random.randint(0,5)) for _ in range(2000)]
with open('sys_random', 'w') as thefile:
	thefile.write("\n".join(r))
