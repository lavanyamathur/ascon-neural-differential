lines = open('data_generator.py').readlines()
clean = lines[:308]
open('data_generator.py', 'w').writelines(clean)
print('Trimmed to', len(clean), 'lines')