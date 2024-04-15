from pythae.models import *
from pythae.models.nn.default_architectures import *
from pythae.models.base.base_utils import ModelOutput


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class Encoder_VAE_TinyMLP(BaseEncoder):
    def __init__(self, args:dict):
        BaseEncoder.__init__(self)
        self.input_dim = args.input_dim
        self.latent_dim = args.latent_dim

        self.layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(np.prod(args.input_dim), 400), nn.ReLU()
            )
        self.embedding = nn.Linear(400, self.latent_dim)
        self.log_var = nn.Linear(400, self.latent_dim)

    def forward(self, x: torch.Tensor) -> ModelOutput:
        out = self.layers(x) #.squeeze())
        output = ModelOutput(
            embedding=self.embedding(out),
            log_covariance=self.log_var(out)
        )
        return output


class Decoder_AE_TinyMLP(BaseDecoder):
    def __init__(self, args:dict):
        BaseDecoder.__init__(self)

        self.input_dim = args.input_dim

        self.layers = nn.Sequential(
            nn.Linear(args.latent_dim, 400), nn.ReLU(),
            nn.Linear(400, int(np.prod(args.input_dim))), nn.Sigmoid()
            )
        
    def forward(self, z: torch.Tensor) -> ModelOutput:
        out = self.layers(z)
        out = out.reshape((z.shape[0],) + self.input_dim)
        output = ModelOutput(reconstruction=out)
        return output