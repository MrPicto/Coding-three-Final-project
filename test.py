
import argparse
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torch.autograd import Variable
from DiscoGAN.models import *
from DiscoGAN.datasets import *
import torch
from torchvision.utils import make_grid


parser = argparse.ArgumentParser()
parser.add_argument("--epoch", type=int, default=200, help="epoch to start training from")
parser.add_argument("--n_epochs", type=int, default=201, help="number of epochs of training")
parser.add_argument("--dataset_name", type=str, default="origin2enhancement", help="name of the dataset")
parser.add_argument("--batch_size", type=int, default=1, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--decay_epoch", type=int, default=100, help="epoch from which to start lr decay")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--img_height", type=int, default=512, help="size of image height")
parser.add_argument("--img_width", type=int, default=512, help="size of image width")
parser.add_argument("--channels", type=int, default=3, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=1000, help="interval between saving generator outputs")
parser.add_argument("--checkpoint_interval", type=int, default=10, help="interval between saving model checkpoints")
parser.add_argument("--n_residual_blocks", type=int, default=9, help="number of residual blocks in generator")
parser.add_argument("--lambda_cyc", type=float, default=10.0, help="cycle loss weight")
parser.add_argument("--lambda_id", type=float, default=5.0, help="identity loss weight")
opt = parser.parse_args()
print(opt)

os.makedirs("results/%s" % opt.dataset_name, exist_ok=True)

# Image transformations
transforms_ = [
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
]

# Test data loader
val_dataloader = DataLoader(
    ImageDataset("../CycleGAN_Data/%s" % opt.dataset_name, transforms_=transforms_, mode="val"),
    batch_size=1,
    shuffle=False,
    num_workers=1,
)

cuda = torch.cuda.is_available()
Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor

input_shape = (opt.channels, opt.img_height, opt.img_width)

# Initialize generator and discriminator
G_BA = GeneratorUNet(input_shape)


if cuda:
    G_BA = G_BA.cuda()

# Load pretrained models
G_BA.load_state_dict(torch.load("saved_models/%s/G_BA_%d.pth" % (opt.dataset_name, opt.epoch)))
G_BA.eval()

for i, batch in enumerate(val_dataloader):
    real_B = Variable(batch["B"].type(Tensor))
    fake_A = G_BA(real_B)
    fake_A = make_grid(fake_A, normalize=True)
    save_image(fake_A, "results/%s/%s.png" % (opt.dataset_name, (i + 1)), normalize=False)
    print('%s.png is done!' % (i + 1))

print("done!")
