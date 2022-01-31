import torch

START_TRAIN_AT_IMG_SIZE = 4
dataset = "/content/prassanna_data"
CHECKPOINT_GENERATOR = "M:\PRO gan project done\saved-loading models etc/generator.pth"
CHECKPOINT_CRITIC = "/content/drive/MyDrive/proGAN/saved loading models etc/critic.pth"
generation_folder = "/content/drive/MyDrive/proGAN/generated_images_latest"
SAVE_MODEL = True
LOAD_MODEL = True



DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
LEARNING_RATE = 1e-3
batch_sizes = [32, 32, 32, 16, 16, 16, 8, 8, 4]
img_ch  = 3
z_dim = 256
in_ch = 256
CRITIC_ITERATIONS = 1
LAMBDA_GP =10
PROGRESSIVE_EPOCHS =[10] * len(batch_sizes)
FIXED_NOISE =torch.randn(8, z_dim, 1, 1).to(DEVICE)
num_workers = 8
END_STEP = 6