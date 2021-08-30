import yaml

path = 'uniformv1.yaml'
fs = yaml.safe_load(open(path))
print(fs)