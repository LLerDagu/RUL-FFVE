import utils
from models.model_FFVE import Encoder, Decoder, reparameterize
import torch
from data.dataset import RUL_Dataset
import os
import argparse
import numpy as np

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--save_dir', type=str, default='./checkpoint/FD001/FFVE_b1/op_4')
    parser.add_argument('--model_file', type=str, default='/best_model_itr4.pt', help='various model_file may result in different images')
    parser.add_argument('--dataset', type=str, default='FD001')
    parser.add_argument('--test_unit_nr', type=int, default=34, help='Engine unit number in the test set for evaluation')
    parser.add_argument('--test_pred_step', type=int, default=10, help='Prediction intervals / Plotting intervals')
    parser.add_argument('--device', type=str, default='cuda:0')

    parser.add_argument('--e_layers', type=int, default=1, help='num of encoder layers')

    parser.add_argument('--norm', default=False, help='whether to apply LayerNorm')
    parser.add_argument('--fac_C', default=True, help='whether to apply factorized channel interaction')
    parser.add_argument('--short_res', type=int, default=True, help='short range residual operation')
    parser.add_argument('--long_res', type=int, default=False, help='long range high-low level feature fusion operation')

    parser.add_argument('--viz_diy', default=False, help='Manually adjusting visualization angles: True, False')
    parser.add_argument('--viz_view_init', default=(30, 30), help='3D plot rotation angle. Recommended viewing angles: z1:(30, -60), z2:(15, 15)/(30, 30)/(30, 210)')
    parser.add_argument('--viz_box_aspect', default=(1, 3, 1), help='Scaling ratio for each axis')
    parser.add_argument('--viz_major_locator', default=(2, 1, 2), help='Set major ticks for the axes')
    parser.add_argument('--viz_xlim', default=(-3, 1), help='X-axis value range')
    parser.add_argument('--viz_ylim', default=(-1.95, 2.65), help='Y-axis value range')
    parser.add_argument('--viz_zlim', default=(-1.35, 1.75), help='Z-axis value range')
    parser.add_argument('--viz_gif_alpha', default=0.025, help='Transparency of the 3D latent space: Lower value for clearer paths of the objects under evaluation, higher value for clearer 3D latent space')
    args = parser.parse_args()

    # ------------------------------ DATA -----------------------------------
    dataset = args.dataset
    save_folder = args.save_dir
    os.makedirs(save_folder, exist_ok=True)

    # sensors to work with: T30, T50, P30, PS30, phi
    # sensors = ['s_3', 's_4', 's_7', 's_11', 's_12']   # 5
    sensors = ['s_2', 's_3', 's_4', 's_7', 's_8', 's_9', 's_11', 's_12', 's_13', 's_14', 's_15', 's_17', 's_20', 's_21']  # 14
    # windows length
    sequence_length = 30
    # smoothing intensity
    alpha = 0.1
    # max RUL
    threshold = 125
    # Load Dataset
    x_train, y_train, x_val, y_val, x_test, y_test = utils.get_data(dataset, sensors, sequence_length, alpha, threshold)
    x_oneUnit = utils.get_oneUnitData(dataset, sensors, sequence_length, alpha, threshold, step = args.test_pred_step, unit_nr = args.test_unit_nr)
    tr_dataset = RUL_Dataset(x_train, y_train)
    val_dataset = RUL_Dataset(x_val, y_val)
    test_dataset = RUL_Dataset(x_test, y_test)
	# -----------------------------------------------------------------------

	# --------------------------------------- MODEL ----------------------------------------
    device = args.device
    intermediate_dim = 300
    latent_dim = 3

    encoder = Encoder(configs=args, sequence_length = sequence_length, input_dim = len(sensors), latent_dim = latent_dim).to(device)
    decoder = Decoder(latent_dim=latent_dim).to(device)

    new_model = torch.load(save_folder + args.model_file)
    encoder.load_state_dict(new_model['encoder'])
    decoder.load_state_dict(new_model['decoder'])

    encoder.eval()
    decoder.eval()
    # -----------------------------------------------------------------------

    # -------------------------- EVALUATION ---------------------------------
    train_mu = utils.viz_latent_space_3Dtrend(args, encoder, reparameterize, torch.tensor(np.concatenate((x_train, x_val))).to(device), torch.tensor(np.concatenate((y_train, y_val))).to(device), save=True, path=save_folder)
    test_mu = utils.viz_latent_space_3Dtrend_lineGif(args, encoder, reparameterize, torch.tensor(np.concatenate((x_train, x_val))).to(device), torch.tensor(x_oneUnit).to(device), torch.tensor(np.concatenate((y_train, y_val))).to(device), epoch='test', save=True, path=save_folder)
    # -----------------------------------------------------------------------
