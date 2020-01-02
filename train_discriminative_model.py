from wiser.discriminative import train_discriminative_model
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-c', '--cuda', type=int, default=-1,
                   help='An integer specifying the cuda device (default: -1)')

parser.add_argument('-d', '--task', default='BC5CDR',
                   help='A string specifying the dataset')

parser.add_argument('-g', '--model', default='nb',
                   help='A string specifying the model to run')

parser.add_argument('-t', '--tags', type=bool, default='False',
                   help = 'A bool specifying whether to train on the true tags')

device = parser.parse_args().cuda
task = parser.parse_args().task
model = parser.parse_args().model
tags = parser.parse_args().tags

if task not in {'BC5CDR', 'NCBI-Disease', 'LaptopReview'}:
    raise ValueError("Task must be one of the following: BC5CDR, NCBI-Disease, LaptopReview")

if model not in {'mb', 'unweighted', 'nb', 'hmm', 'link_hmm'}:
    raise ValueError("Generative model must be one of the following: mv, unweighted, nb, hmm, link_hmm")

train_discriminative_model(
    task + '/output/train_data_%s.p' %(model),
    task + '/output/dev_data.p',
    task + '/output/test_data.p',
    'training_config/%s_config.jsonnet' % (task),
    use_tags=tags,
    cuda_device=device)
