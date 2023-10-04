import os
import numpy as np
import argparse
from keras.models import load_model
from tqdm import tqdm

from modules import TransformerBlock, TokenAndPositionEmbedding
from config import Config


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description='Script for making predictions using a pre-trained model',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    run = parser.add_argument_group(title='Prediction Parameters',
                                    description='Parameters for prediction using the model')
    run.add_argument('-i', '--input_folder', help='Input data folder')
    run.add_argument('-m', '--model_folder', help='Model folder')
    args = parser.parse_args()

    # Initialize model parameters
    model_config = Config()
    embedding_layer = TokenAndPositionEmbedding(model_config.wsize,
                                                model_config.vocab_size,
                                                model_config.embed_dim)
    transformer_block1 = TransformerBlock(model_config.embed_dim,
                                          model_config.num_heads,
                                          model_config.mlp_dim)
    transformer_block2 = TransformerBlock(model_config.embed_dim,
                                          model_config.num_heads,
                                          model_config.mlp_dim)

    print("--------------------------------------------------")
    parpath = os.path.dirname(os.getcwd())
    inputpath = os.path.join(parpath, args.input_folder)

    # Load the pre-trained model
    if args.model_folder.endswith('.h5'):
        model = load_model(os.path.join(parpath, 'models', args.model_folder),
                           custom_objects={'TransformerBlock': transformer_block1,
                                           'TokenAndPositionEmbedding': embedding_layer,
                                           },
                           )
    else:
        model = load_model(os.path.join(parpath, 'models', args.model_folder))

    inputpath = os.path.join(parpath, 'datasets', args.input_folder)
    all_files = os.listdir(inputpath)
    xc_files = [f for f in all_files if f.endswith('xc.txt')]

    if len(xc_files) < 1:
        print("Input data not found. Please use the data_processing.py to generate the input dataset.")

    if len(xc_files) > 1:
        print("Multiple input datasets exist.")
    
    # Data loading progress bar
    with tqdm(total=len(xc_files), desc="Data Loading", unit="file") as pbar:
        x_c = None
        for xc_file in xc_files:
            xc_path = os.path.join(inputpath, xc_file)
            xc_data = np.loadtxt(xc_path, delimiter="\t")
            if x_c is None:
                x_c = xc_data
            else:
                x_c = np.concatenate((x_c, xc_data))
            pbar.update(1)

    x_c[:, :40] = x_c[:, 0:40] / 100 - 5
    x_c[:, 40] = x_c[:, 40] / 100

    print("--------------------------------------------------")

    # Model prediction progress bar
    with tqdm(total=len(x_c), desc="Model Prediction", unit="data") as pbar:
        y_pred = []
        batch_size = 200000
        for i in range(0, len(x_c), batch_size):
            batch_x = [x_c[i:i + batch_size, -40:],
                       x_c[i:i + batch_size, :40]]
            batch_pred = model.predict(batch_x)
            y_pred.extend(batch_pred)
            pbar.update(len(batch_x[0]))

    np.savetxt(os.path.join(parpath, 'datasets', args.input_folder, 'model_prediction.txt'),
               y_pred, delimiter="\t")


if __name__ == "__main__":
    main()