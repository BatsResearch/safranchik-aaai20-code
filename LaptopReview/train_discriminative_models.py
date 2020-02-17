from wiser.discriminative import train_discriminative_model
import argparse

dev_file = 'output/generative/dev_data.p'
test_file = 'output/generative/test_data.p'
config_path = 'LaptopReview_config.jsonnet'

parser = argparse.ArgumentParser()
parser.add_argument("--cuda", default=-1, help="Specifies the cuda device")
parser.add_argument(
    "--iterations",
    default=1,
    help="Specifies the number of runs using different seeds")


args = parser.parse_args()
cuda_device = args.cuda
num_iterations = args.iterations

for iteration in range(num_iterations):
    for model in ['mv', 'unweighted', 'nb', 'hmm', 'link_hmm']:

        train_file = 'output/generative/train_data_%s.p' % model
        output_path = 'output/discriminative/%s_%s' % (model, str(iteration))

        train_discriminative_model(
            train_data_path=train_file,
            dev_data_path=dev_file,
            test_data_path=test_file,
            train_config_path=config_path,
            output_path=output_path,
            cuda_device=cuda_device)

    train_file = 'output/generative/train_data_mv.p'
    output_path = 'output/discriminative/supervised_%s' % iteration

    train_discriminative_model(
        train_data_path=train_file,
        dev_data_path=dev_file,
        test_data_path=test_file,
        train_config_path=config_path,
        output_path=output_path,
        use_tags=True,
        cuda_device=cuda_device)
