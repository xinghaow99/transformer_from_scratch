from datasets import load_dataset
dataset = load_dataset('news_commentary', 'en-zh', cache_dir='data')
dataset = dataset['train']
dataset = dataset.train_test_split(0.1)
print(dataset)
print(dataset['train'][0])