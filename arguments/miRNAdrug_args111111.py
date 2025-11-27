import argparse

def get_args():

    parser = argparse.ArgumentParser(description='miRNA–Drug Association Prediction Arguments')

    # Negative sampling
    parser.add_argument('--neg_ratio', default=3, type=int, choices=[1, 2, 3],
                        help='Ratio of negative samples to positive samples')

    # Dataset statistics: miRNA–drug (1-dataset)
    parser.add_argument('--m_d', default=269, type=int, help='Number of miRNAs')
    parser.add_argument('--d_d', default=598, type=int, help='Number of drugs')
    parser.add_argument('--total', default=867, type=int, help='Total number of miRNA–drug pairs')

    parser.add_argument('--n_heads', default=4, type=int, help='Number of attention heads')

    parser.add_argument('--epochs', default=541, type=int, help='Number of training epochs')
    parser.add_argument('--batch_size', default=3276, type=int, help='Mini-batch size used for training')

    parser.add_argument('--k1', default=176, type=int, help='Number of nearest miRNA neighbors')
    parser.add_argument('--k2', default=104, type=int, help='Number of nearest drug neighbors')

    # Diffusion process parameters
    parser.add_argument('--time_steps', type=int, default=574, help='Total diffusion steps')
    parser.add_argument('--scheduler', type=str, default='cosine',
                        choices=['cosine', 'linear'], help='Type of diffusion scheduler')
    parser.add_argument('--s', type=float, default=2.024345720429339e-05,
                        help='Smoothing parameter for cosine scheduler')
    parser.add_argument('--beta_start', type=float, default=1e-4,
                        help='Starting beta value for linear scheduler')
    parser.add_argument('--beta_end', type=float, default=0.02,
                        help='Ending beta value for linear scheduler')

    parser.add_argument('--w1', type=float, default=0.5524637956228948,
                        help='Weight factor for similarity fusion')
    parser.add_argument('--penalty_factor', type=float, default=0.1,
                        help='Penalty factor used in diffusion loss')

    # Feature dimensions
    parser.add_argument('--d_model', type=int, default=768,
                        help='Hidden feature dimension for attention layers')
    parser.add_argument('--x_model', type=int, default=256,
                        help='Feature dimensionality after preprocessing')

    parser.add_argument('--num_layers', type=int, default=3,
                        help='Number of multiscale attention layers')
    parser.add_argument('--dropout', type=float, default=0.08890381147844478,
                        help='Dropout probability')

    parser.add_argument('--lr', type=float, default=0.0004634920174142083,
                        help='Learning rate')

    parser.add_argument('--lambda_noise', type=float, default=0.007953508967864443,
                        help='Loss weight for diffusion noise regularization')
    parser.add_argument('--mlp_hidden', type=int, default=128,
                        help='Hidden dimension of the MLP classifier')

    parser.add_argument('--d_h', type=int, default=256,
                        help='Dimensionality of hidden layers in the network')
    parser.add_argument('--G_weight', default=0.8420099902965978, type=int,
                        help='Weight coefficient used in graph-based fusion')

    # File paths (1-dataset)
    parser.add_argument('--res_dir', default='./results/1-dataset_results3',
                        help='Directory to save experimental results')
    parser.add_argument('--miRNA_sim_dir', default=r"./drugdisease/1-dataset/drug_drug.txt",
                        help='Path to miRNA similarity matrix file')
    parser.add_argument('--drug_sim_dir', default=r"./drugdisease/1-dataset/disease_disease.txt",
                        help='Path to drug similarity matrix file')
    parser.add_argument('--association_m_dir', default=r"./drugdisease/1-dataset/drug_disease.txt",
                        help='Path to miRNA–drug association matrix file')

    parser.add_argument('--device', default='cuda:0', type=str,
                        help='Device identifier for training (e.g., cuda:0)')
    parser.add_argument('--fold', default=5, type=int, help='Number of cross-validation folds')

    args = parser.parse_args()
    return args
