import sklearn.datasets as ds

dataset_loaders = [name for name in dir(ds) if name.startswith("load_")]

print(dataset_loaders)

