from wiser.discriminative import train_discriminative_model
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-c', '--cuda', type=int, default=-1,
                   help='An integer specifying the cuda device (default: -1)')

parser.add_argument('-t', '--task', default='BC5CDR',
                   help='An string specifying the task')

parser.add_argument('-g', '--model', default='nb',
                   help='An string specifying the model to run')


device = parser.parse_args().cuda
task = parser.parse_args().task
model = parser.parse_args().model

if task not in {'BC5CDR', 'NCBI', 'LaptopReview'}:
    

train_discriminative_model(
    task + '/generative_output/train_data.p',
    task + '/generative_output/dev_data.p',
    task + '/generative_output/test_data.p',
    'training_config/%s_config.jsonnet' % (task),
    cuda_device=device)
