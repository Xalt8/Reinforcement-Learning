gp = (x for x in range(10,15))
reward = 0

def done(reward):
    #observation, reward, done, info
    done = False
    try:
        observation = next(gp)
        reward += observation
        return observation, reward, done
    except:
        done = True
        return observation, reward, done
    
for i in range(5):
    print(done(reward))