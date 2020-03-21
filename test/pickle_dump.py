import pickle

a = {11: 11, 22: 'asdsad'}
history_file = open('history', 'wb')
pickle.dump(a, history_file)
history_file.close()
