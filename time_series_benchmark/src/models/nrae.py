import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader



def get_encoder(d_input, d_latent, layer_sz=16, alpha=0.5):
    return nn.Sequential(
        nn.Linear(d_input, layer_sz),
        nn.ELU(alpha=alpha),
        nn.Linear(layer_sz, layer_sz),
        nn.ELU(alpha=alpha),
        nn.Linear(layer_sz, layer_sz),
        nn.ELU(alpha=alpha),
        nn.Linear(layer_sz, d_latent)
    )

def get_decoder(d_latent, d_output, layer_sz=16, alpha=0.5):
    return nn.Sequential(
        nn.Linear(d_latent, layer_sz),
        nn.ELU(alpha=alpha),
        nn.Linear(layer_sz, layer_sz),
        nn.ELU(alpha=alpha),
        nn.Linear(layer_sz, layer_sz),
        nn.ELU(alpha=alpha),
        nn.Linear(layer_sz, d_output)
    )


def get_kernel_function(kernel):
    if kernel['type'] == 'binary':
        def kernel_func(x_center, x_neighbors):
            batch_size = x_neighbors.size(0)
            num_neighbors = x_neighbors.size(1)
            x_center = x_center.view(batch_size, -1)
            x_neighbors = x_neighbors.view(batch_size, num_neighbors, -1)
            eps = 1.0e-12
            index = torch.norm(x_center.unsqueeze(1)-x_neighbors, dim=2) > eps
            output = torch.ones(batch_size, num_neighbors).to(x_center)
            output[index] = kernel['lambda']
            return output
    return kernel_func


class AE(nn.Module):
    def __init__(self, d_in, d_out, l_sz):
        super(AE, self).__init__()
        self.encoder = get_encoder(d_input=d_in, d_latent=d_out, layer_sz=l_sz)
        self.decoder = get_decoder(d_latent=l_sz, d_output=d_out)

    def forward(self, x):
        z = self.encoder(x)
        recon = self.decoder(z)
        loss = ((recon - x) ** 2).view(len(x), -1).mean(dim=1).mean()
        mse_loss = F.mse_loss(recon, x)
        return {"recon": recon, "loss": loss, 'mse_loss': mse_loss}


class NRAE(nn.Module):
    def __init__(self, d_in, d_out, l_sz, approx_order=1, kernel=None):
        super(NRAE, self).__init__()
        self.d_in = d_in
        self.d_out = d_out
        self.encoder = get_encoder(d_input=d_in, d_latent=d_out, layer_sz=l_sz)
        self.decoder = get_decoder(d_latent=d_out, d_output=d_in, layer_sz=l_sz)
        self.approx_order = approx_order
        self.kernel_func = get_kernel_function(kernel)

    def jacobian(self, z, dz, create_graph=True):
        batch_size = dz.size(0)
        num_neighbors = dz.size(1)
        z_dim = dz.size(2)

        v = dz.view(-1, z_dim)
        inputs = (z.unsqueeze(1).repeat(1, num_neighbors, 1).view(-1, z_dim))
        jac = torch.autograd.functional.jvp(self.decoder, inputs, v=v, create_graph=create_graph)[1].view(batch_size, num_neighbors, -1)
        return jac

    def jacobian_and_hessian(self, z, dz, create_graph=True):
        batch_size = dz.size(0)
        num_neighbors = dz.size(1)
        z_dim = dz.size(2)

        v = dz.view(-1, z_dim)
        inputs = (z.unsqueeze(1).repeat(1, num_neighbors, 1).view(-1, z_dim))

        def jac_temp(inputs):
            jac = torch.autograd.functional.jvp(self.decoder, inputs, v=v, create_graph=create_graph)[1].view(batch_size, num_neighbors, -1)
            return jac

        temp = torch.autograd.functional.jvp(jac_temp, inputs, v=v, create_graph=create_graph)

        jac = temp[0].view(batch_size, num_neighbors, -1)
        hessian = temp[1].view(batch_size, num_neighbors, -1)
        return jac, hessian
        
    def neighborhood_recon(self, z_center, z_neighbors):
        recon = self.decoder(z_center)
        recon_flat = recon.unsqueeze(1)  # [batch_size, 1, d_in]
        dz = z_neighbors - z_center.unsqueeze(1)  # [batch_size, k_neighbors, d_out]
        
        if self.approx_order == 1:
            jacobian_dz = self.jacobian(z_center, dz)  # [batch_size, k_neighbors, d_in]
            neighbor_recon = recon_flat + jacobian_dz
        elif self.approx_order == 2:
            jacobian_dz, dz_hessian_dz = self.jacobian_and_hessian(z_center, dz)
            neighbor_recon = recon_flat + jacobian_dz + 0.5*dz_hessian_dz
            
        return neighbor_recon  # [batch_size, k_neighbors, d_in]

    def forward(self, x_center, x_neighbors):
        batch_size = x_center.size(0)
        num_neighbors = x_neighbors.size(1)
        
        # Encode
        z_center = self.encoder(x_center)  # [batch_size, d_out]
        z_neighbors = self.encoder(x_neighbors.reshape(-1, self.d_in))  # [batch_size*k_neighbors, d_out]
        z_neighbors = z_neighbors.view(batch_size, num_neighbors, -1)  # [batch_size, k_neighbors, d_out]
        
        # Decode center point
        recon = self.decoder(z_center)  # [batch_size, d_in]
        
        # Compute neighborhood reconstruction
        neighbor_recon = self.neighborhood_recon(z_center, z_neighbors)  # [batch_size, k_neighbors, d_in]
        
        # Compute losses
        neighbor_loss = torch.norm(x_neighbors - neighbor_recon, dim=2)**2  # [batch_size, k_neighbors]
        weights = self.kernel_func(x_center, x_neighbors)  # [batch_size, k_neighbors]
        loss = (weights*neighbor_loss).mean()
        mse_loss = F.mse_loss(recon, x_center)
        
        return {"recon": recon, "loss": loss, "mse_loss": mse_loss}
